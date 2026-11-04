# modulo_simulacion/politicas.py
"""
Políticas de reposición y modelo analítico de costo.

DOS COSAS VIVEN AQUÍ Y CONVIENE NO CONFUNDIRLAS
------------------------------------------------
1. Las FÓRMULAS que fijan los parámetros de la política (ROP, Q, S). Son las
   que el gemelo aplica para decidir cuánto y cuándo comprar.

2. Un MODELO ANALÍTICO de costo total. No decide nada: es un sustituto
   cerrado de la simulación, y existe por una razón concreta — permite
   calcular valores de Shapley EXACTOS sobre los ocho factores de costo sin
   correr 256 simulaciones por SKU. Su fidelidad frente a la simulación se
   mide y se reporta; si se separan demasiado, la explicación no vale y el
   sistema lo dice.

LA FÓRMULA QUE CASI TODOS SIMPLIFICAN
--------------------------------------
El stock de seguridad con tiempo de entrega variable es

    SS = z * raiz( L * sigma_d^2  +  d^2 * sigma_L^2 )

El segundo término —la variabilidad del proveedor— es el que se omite al
asumir tiempo de entrega constante, y para SOAZ suele ser el DOMINANTE: un
proveedor que sube a Otuzco por carretera de sierra es más irregular que la
demanda de arroz. Omitirlo produce un stock de seguridad que se ve razonable
y es sistemáticamente insuficiente.
"""
import numpy as np
from scipy import stats

from .parametros import COSTOS, NEGOCIO, NIVELES_SERVICIO

DIAS_ANIO = NEGOCIO["dias_habiles_semana"] * 52


# --------------------------------------------------------------------------
# Componentes de la demanda durante el tiempo de entrega
# --------------------------------------------------------------------------
def desviacion_ddlt(demanda_media, demanda_desv, lead_time_medio, lead_time_desv,
                    periodo_proteccion_extra=0.0):
    """
    Desviación estándar de la demanda durante el período de protección.

    Para (s,Q) el período de protección es L. Para (R,S) es L + R, porque
    entre dos revisiones el inventario queda sin vigilancia y hay que cubrir
    también ese tramo. Es la razón por la que la revisión periódica siempre
    exige más stock de seguridad que la continua: no es una ineficiencia del
    modelo, es el precio de mirar menos seguido.
    """
    L = lead_time_medio + periodo_proteccion_extra
    return float(np.sqrt(max(L, 1e-9) * demanda_desv ** 2
                         + demanda_media ** 2 * lead_time_desv ** 2))


def stock_seguridad(z, demanda_media, demanda_desv, lead_time_medio, lead_time_desv,
                    periodo_proteccion_extra=0.0):
    """SS = z * sigma_DDLT. Ver nota de cabecera sobre el segundo término."""
    return float(z * desviacion_ddlt(demanda_media, demanda_desv,
                                     lead_time_medio, lead_time_desv,
                                     periodo_proteccion_extra))


def stock_seguridad_simplificado(z, demanda_desv, lead_time_medio):
    """
    Versión que asume tiempo de entrega constante. NO la usa el motor.

    Existe solo para poder mostrar en el informe cuánto se subestima el stock
    de seguridad al omitir la variabilidad del proveedor.
    """
    return float(z * demanda_desv * np.sqrt(max(lead_time_medio, 1e-9)))


def cantidad_economica(demanda_media, costo_unitario, costo_pedido=None,
                       tasa_mantenimiento=None):
    """
    Cantidad económica de pedido (EOQ). Es el punto de partida de Q; la
    simulación después la contrasta contra el costo total observado, que
    incluye rupturas y merma que el EOQ no ve.
    """
    costo_pedido = costo_pedido if costo_pedido is not None else COSTOS["costo_pedido"]
    tasa = tasa_mantenimiento if tasa_mantenimiento is not None else COSTOS["tasa_mantenimiento_anual"]
    demanda_anual = demanda_media * DIAS_ANIO
    costo_mantener_unidad_anio = max(tasa * costo_unitario, 1e-9)
    return float(np.sqrt(2.0 * demanda_anual * costo_pedido / costo_mantener_unidad_anio))


# --------------------------------------------------------------------------
# Construcción de la política de un SKU
# --------------------------------------------------------------------------
def construir_politica(sku_estado, tipo="(s,Q)", periodo_revision=7,
                       nivel_servicio=None, costo_pedido=None):
    """
    Calcula los parámetros de la política para un SKU a partir de sus
    parámetros estimados. Devuelve un diccionario listo para el motor.
    """
    nivel = nivel_servicio if nivel_servicio is not None else sku_estado.get("nivel_servicio", 0.95)
    z = NIVELES_SERVICIO.get(round(nivel, 3), sku_estado.get("z", 1.6449))
    d, sd = sku_estado["demanda_media"], sku_estado["demanda_desv"]
    L, sL = sku_estado["lead_time_medio"], sku_estado["lead_time_desv"]
    extra = periodo_revision if tipo == "(R,S)" else 0.0

    ss = stock_seguridad(z, d, sd, L, sL, extra)
    q = max(sku_estado.get("lote_minimo", 1),
            cantidad_economica(d, sku_estado["costo_unitario"], costo_pedido))

    politica = {
        "tipo": tipo, "nivel_servicio": nivel, "z": z,
        "stock_seguridad": ss,
        "sigma_ddlt": desviacion_ddlt(d, sd, L, sL, extra),
        "periodo_revision": periodo_revision if tipo == "(R,S)" else None,
        "cantidad_pedido": q,
        "ss_simplificado": stock_seguridad_simplificado(z, sd, L),
    }
    if tipo == "(R,S)":
        politica["nivel_objetivo_S"] = d * (L + periodo_revision) + ss
        politica["punto_reorden"] = None
    else:
        politica["punto_reorden"] = d * L + ss
        politica["nivel_objetivo_S"] = None
    return politica


def construir_politica_practica_actual(sku_estado, dias_reorden=10, dias_pedido=20,
                                       periodo_revision=7):
    """
    La política que SOAZ aplica hoy: revisión semanal, punto de reorden por
    regla del pulgar (unos 10 días de cobertura) y compra por unos 20 días.

    No tiene stock de seguridad explícito ni considera la variabilidad del
    proveedor. Reproducirla es indispensable para la validación de fidelidad:
    el gemelo debe recrear lo que el negocio HIZO antes de que se le crea lo
    que el negocio DEBERÍA hacer.
    """
    # Punto fino de la validación de fidelidad: para reproducir lo que SOAZ
    # HIZO hay que usar el número con el que SOAZ decidía, que es su promedio
    # de ventas —sesgado a la baja por la censura—, no la demanda corregida.
    # Validar la fidelidad de un gemelo exige modelar al que decide, no solo
    # la física del almacén.
    d = sku_estado.get("demanda_percibida", sku_estado["demanda_media"])
    return {
        "tipo": "(R,s,Q)",
        "nivel_servicio": None,
        "z": 0.0,
        "stock_seguridad": 0.0,
        "sigma_ddlt": desviacion_ddlt(d, sku_estado["demanda_desv"],
                                      sku_estado["lead_time_medio"],
                                      sku_estado["lead_time_desv"]),
        "periodo_revision": periodo_revision,
        "punto_reorden": d * dias_reorden,
        "cantidad_pedido": max(sku_estado.get("lote_minimo", 1), d * dias_pedido),
        "nivel_objetivo_S": None,
        "ss_simplificado": 0.0,
    }


# --------------------------------------------------------------------------
# Modelo analítico de costo — base de la explicabilidad exacta
# --------------------------------------------------------------------------
def funcion_perdida_normal(z):
    """
    G(z) = phi(z) - z * (1 - Phi(z)).

    Es el número esperado de unidades faltantes por ciclo, en desviaciones
    estándar. Sin ella no se puede poner precio a la ruptura, y sin precio a
    la ruptura cualquier optimización recomienda no tener stock.
    """
    return float(stats.norm.pdf(z) - z * (1.0 - stats.norm.cdf(z)))


def costo_total_anual(demanda_media, demanda_desv, lead_time_medio, lead_time_desv,
                      costo_unitario, costo_pedido, nivel_servicio_z, cantidad_pedido,
                      vida_util=3650.0, tasa_mantenimiento=None, penalidad_factor=None,
                      periodo_proteccion_extra=0.0):
    """
    Costo total anual del SKU bajo política (s,Q), en forma cerrada.

    Costo = pedidos + mantenimiento (ciclo + seguridad) + ruptura + MERMA

    El término de merma se añadió después de medir la fidelidad del sustituto
    contra la simulación: sin él, el modelo analítico erraba en más del 90% en
    los lácteos y perecibles, porque en esa familia la merma es la mayor parte
    del costo. Un sustituto que no incluye el costo dominante produce
    explicaciones bien formadas y equivocadas.

    Los NUEVE argumentos son exactamente los factores sobre los que se calcula
    la descomposición de Shapley. Que la función sea analítica es lo que
    permite evaluarla 512 veces sin costo y obtener valores exactos.
    """
    tasa = tasa_mantenimiento if tasa_mantenimiento is not None else COSTOS["tasa_mantenimiento_anual"]
    penalidad = penalidad_factor if penalidad_factor is not None else COSTOS["penalidad_ruptura_factor"]

    demanda_anual = max(demanda_media, 1e-9) * DIAS_ANIO
    q = max(cantidad_pedido, 1e-6)
    sigma = desviacion_ddlt(demanda_media, demanda_desv, lead_time_medio,
                            lead_time_desv, periodo_proteccion_extra)
    ss = nivel_servicio_z * sigma
    h = tasa * costo_unitario                       # costo de mantener 1 unidad 1 año
    ciclos = demanda_anual / q

    costo_ordenar = costo_pedido * ciclos
    costo_mantener = h * (q / 2.0 + ss)
    faltante_por_ciclo = sigma * funcion_perdida_normal(nivel_servicio_z)
    costo_ruptura = penalidad * costo_unitario * ciclos * faltante_por_ciclo

    # Merma: lo que el ciclo no alcanza a vender antes de vencer. Un lote de
    # tamaño Q más el stock de seguridad se consume en (Q+SS)/d días; lo que
    # exceda la vida útil se pierde.
    unidades_en_riesgo = max(0.0, (q + ss) - demanda_media * vida_util)
    costo_merma = (COSTOS["valor_merma_factor"] * costo_unitario
                   * ciclos * unidades_en_riesgo)

    return {
        "costo_total": costo_ordenar + costo_mantener + costo_ruptura + costo_merma,
        "costo_merma": costo_merma,
        "unidades_merma_ciclo": unidades_en_riesgo,
        "costo_ordenar": costo_ordenar,
        "costo_mantener": costo_mantener,
        "costo_ruptura": costo_ruptura,
        "stock_seguridad": ss,
        "sigma_ddlt": sigma,
        "ciclos_anuales": ciclos,
        "faltante_por_ciclo": faltante_por_ciclo,
    }


def vector_factores(sku_estado, politica, costo_pedido=None):
    """Extrae del SKU los ocho factores del modelo analítico."""
    return {
        "demanda_media": sku_estado["demanda_media"],
        "demanda_desv": sku_estado["demanda_desv"],
        "lead_time_medio": sku_estado["lead_time_medio"],
        "lead_time_desv": sku_estado["lead_time_desv"],
        "costo_unitario": sku_estado["costo_unitario"],
        "costo_pedido": costo_pedido if costo_pedido is not None else COSTOS["costo_pedido"],
        "nivel_servicio_z": politica["z"],
        "cantidad_pedido": politica["cantidad_pedido"],
        "vida_util": float(sku_estado["vida_util_dias"]),
    }


def costo_desde_vector(v, periodo_proteccion_extra=0.0):
    """Evalúa el modelo analítico desde el vector de factores."""
    return costo_total_anual(
        v["demanda_media"], v["demanda_desv"], v["lead_time_medio"], v["lead_time_desv"],
        v["costo_unitario"], v["costo_pedido"], v["nivel_servicio_z"], v["cantidad_pedido"],
        vida_util=v.get("vida_util", 3650.0),
        periodo_proteccion_extra=periodo_proteccion_extra,
    )["costo_total"]
