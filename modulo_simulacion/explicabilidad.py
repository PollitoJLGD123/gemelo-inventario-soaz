# modulo_simulacion/explicabilidad.py
"""
Capa de explicabilidad del gemelo — por qué sale lo que sale.

POR QUÉ SHAPLEY EXACTO Y NO SHAP
---------------------------------
SHAP aproxima los valores de Shapley porque los modelos que explica —bosques,
redes— son cajas negras costosas de evaluar. Aquí el costo total de un SKU
tiene forma cerrada en ocho factores, así que evaluarlo 2^8 = 256 veces cuesta
microsegundos y los valores salen EXACTOS: sin muestreo y sin error de
aproximación.

Eso permite algo que con SHAP no se puede hacer: **verificar el axioma de
eficiencia en pantalla**. La suma de las contribuciones tiene que ser
exactamente la diferencia entre el SKU y la referencia. Si no cuadra, la
explicación está mal y el sistema lo dice.

LA HONESTIDAD QUE ESTO EXIGE
-----------------------------
La función analítica NO es el simulador: es un sustituto cerrado del
simulador. Explicar el sustituto y presentarlo como si se explicara el motor
sería un fraude silencioso. Por eso `fidelidad_sustituto()` mide cuánto se
parecen los dos, y ese número acompaña a toda explicación. Si el sustituto se
aparta demasiado, la explicación no vale.

TRES NIVELES
-------------
  Local    → por qué ESTE SKU cuesta lo que cuesta         (Shapley exacto)
  Global   → qué factores mueven el costo del almacén      (tornado / OAT)
  Política → por qué un escenario cambia el resultado      (palanca a palanca)
"""
from itertools import combinations
from math import factorial

import numpy as np

from .motor_des import simular
from .parametros import (
    FACTORES_SHAPLEY, ETIQUETAS_FACTORES, ESCENARIOS, COSTOS, EJECUCION,
)
from .politicas import (
    construir_politica, vector_factores, costo_desde_vector, costo_total_anual,
)


# --------------------------------------------------------------------------
# Referencia
# --------------------------------------------------------------------------
def vector_referencia(estado, nivel_servicio=0.95):
    """
    El «SKU promedio» del almacén: la referencia contra la que se explica.

    La elección de referencia NO es neutral. Los valores de Shapley responden
    «por qué este SKU y no el promedio», así que cambiar la referencia cambia
    toda la explicación. Por eso se devuelve junto con ella y se declara en la
    interpretación de la figura.
    """
    vectores = []
    for e in estado:
        pol = construir_politica(e, nivel_servicio=nivel_servicio)
        vectores.append(vector_factores(e, pol))
    return {k: float(np.mean([v[k] for v in vectores])) for k in FACTORES_SHAPLEY}


# --------------------------------------------------------------------------
# Shapley exacto
# --------------------------------------------------------------------------
def shapley_exacto(vector_sku, vector_base):
    """
    Valores de Shapley exactos sobre los ocho factores del costo.

    Enumera las 2^8 = 256 coaliciones. No hay muestreo: el resultado ES el
    valor de Shapley, no una estimación suya.
    """
    n = len(FACTORES_SHAPLEY)
    idx = {nombre: i for i, nombre in enumerate(FACTORES_SHAPLEY)}

    valor = {}
    for mascara in range(1 << n):
        v = dict(vector_base)
        for nombre, i in idx.items():
            if mascara & (1 << i):
                v[nombre] = vector_sku[nombre]
        valor[mascara] = costo_desde_vector(v)

    contribuciones = {}
    for nombre, i in idx.items():
        bit = 1 << i
        otros = [j for j in range(n) if j != i]
        phi = 0.0
        for tam in range(n):
            peso = factorial(tam) * factorial(n - tam - 1) / factorial(n)
            for sub in combinations(otros, tam):
                mascara = 0
                for j in sub:
                    mascara |= (1 << j)
                phi += peso * (valor[mascara | bit] - valor[mascara])
        contribuciones[nombre] = phi

    v_base, v_total = valor[0], valor[(1 << n) - 1]
    error = abs(sum(contribuciones.values()) - (v_total - v_base))
    return {
        "contribuciones": contribuciones,
        "valor_base": v_base,
        "valor_sku": v_total,
        "suma_contribuciones": sum(contribuciones.values()),
        "error_eficiencia": error,
        "eficiencia_ok": error < 1e-6,
        "n_coaliciones": 1 << n,
    }


def explicar_sku(sku_estado, estado, nivel_servicio=0.95):
    """Explicación local completa de un SKU, lista para graficar."""
    pol = construir_politica(sku_estado, nivel_servicio=nivel_servicio)
    v_sku = vector_factores(sku_estado, pol)
    v_base = vector_referencia(estado, nivel_servicio)
    sh = shapley_exacto(v_sku, v_base)

    unidades = {
        "demanda_media": "u/día", "demanda_desv": "u/día",
        "lead_time_medio": "días", "lead_time_desv": "días",
        "costo_unitario": "S/.", "costo_pedido": "S/.",
        "nivel_servicio_z": "z", "cantidad_pedido": "u", "vida_util": "días",
    }
    filas = []
    for nombre, phi in sorted(sh["contribuciones"].items(), key=lambda kv: -abs(kv[1])):
        filas.append({
            "Factor": ETIQUETAS_FACTORES[nombre],
            "clave": nombre,
            "Valor del SKU": round(v_sku[nombre], 2),
            "Valor de referencia": round(v_base[nombre], 2),
            "Unidad": unidades[nombre],
            "Contribución (S/.)": round(phi, 0),
            "Dirección": "Encarece" if phi > 0 else "Abarata",
        })

    desglose = costo_total_anual(**v_sku)
    return {
        **sh, "tabla": filas, "vector_sku": v_sku, "vector_referencia": v_base,
        "desglose": desglose, "politica": pol, "sku": sku_estado["sku"],
    }


# --------------------------------------------------------------------------
# Fidelidad del sustituto analítico
# --------------------------------------------------------------------------
def fidelidad_sustituto(estado, dias=None, n_replicas=8, semilla_base=64000):
    """
    ¿Se parece el modelo analítico al simulador que sí decide?

    Compara el costo anualizado que predice la fórmula cerrada contra el que
    produce la simulación de eventos discretos. Si se separan demasiado, la
    explicación de Shapley describe una función que no es el motor, y hay que
    decirlo antes de que alguien lo descubra.
    """
    dias = dias or EJECUCION["dias_simulacion"]
    corridas = [simular(estado, {}, dias=dias, semilla=semilla_base + i * 401)
                for i in range(n_replicas)]
    factor_anual = 312.0 / dias

    sim_por_sku = {}
    for c in corridas:
        for s in c["por_sku"]:
            sim_por_sku.setdefault(s["sku"], []).append(s["costo_total"] * factor_anual)

    filas = []
    for e in estado:
        pol = construir_politica(e)
        analitico = costo_total_anual(**vector_factores(e, pol))["costo_total"]
        simulado = float(np.mean(sim_por_sku[e["sku"]]))
        filas.append({
            "SKU": e["sku"],
            "Costo analítico (S/.)": round(analitico, 0),
            "Costo simulado (S/.)": round(simulado, 0),
            "Error relativo (%)": round((analitico - simulado) / max(simulado, 1e-9) * 100, 1),
        })

    errores = np.array([f["Error relativo (%)"] for f in filas])
    anal = np.array([f["Costo analítico (S/.)"] for f in filas], dtype=float)
    sim = np.array([f["Costo simulado (S/.)"] for f in filas], dtype=float)
    correlacion = float(np.corrcoef(anal, sim)[0, 1]) if len(anal) > 2 else 1.0
    return {
        "tabla": sorted(filas, key=lambda f: -abs(f["Error relativo (%)"])),
        "error_absoluto_medio": float(np.mean(np.abs(errores))),
        "sesgo_medio": float(np.mean(errores)),
        "correlacion": correlacion,
        "n_replicas": n_replicas,
        "apto": float(np.mean(np.abs(errores))) < 40.0 and correlacion > 0.85,
    }


# --------------------------------------------------------------------------
# Explicabilidad global — tornado
# --------------------------------------------------------------------------
def tornado_sensibilidad(estado, nivel_servicio=0.95, p_bajo=10, p_alto=90):
    """
    Cuánto mueve el costo cada factor POR SÍ SOLO, del percentil 10 al 90.

    LIMITACIÓN QUE HAY QUE DECLARAR: el análisis mueve un factor a la vez, así
    que no captura interacciones. Si dos factores solo importan juntos, el
    tornado los subestima a ambos. Para el reparto que sí incluye interacciones
    está el Shapley local.
    """
    base = vector_referencia(estado, nivel_servicio)
    vectores = [vector_factores(e, construir_politica(e, nivel_servicio=nivel_servicio))
                for e in estado]
    rangos = {k: (float(np.percentile([v[k] for v in vectores], p_bajo)),
                  float(np.percentile([v[k] for v in vectores], p_alto)))
              for k in FACTORES_SHAPLEY}

    valor_base = costo_desde_vector(base)
    filas = []
    for k in FACTORES_SHAPLEY:
        lo, hi = rangos[k]
        v_lo, v_hi = dict(base), dict(base)
        v_lo[k], v_hi[k] = lo, hi
        r_lo, r_hi = costo_desde_vector(v_lo), costo_desde_vector(v_hi)
        filas.append({
            "Factor": ETIQUETAS_FACTORES[k],
            "clave": k,
            "Valor p10": round(lo, 2),
            "Valor p90": round(hi, 2),
            "Costo en p10 (S/.)": round(r_lo, 0),
            "Costo en p90 (S/.)": round(r_hi, 0),
            "Amplitud (S/.)": round(abs(r_hi - r_lo), 0),
            "efecto_bajo": r_lo - valor_base,
            "efecto_alto": r_hi - valor_base,
        })
    filas.sort(key=lambda d: -d["Amplitud (S/.)"])
    return {"filas": filas, "valor_base": valor_base}


# --------------------------------------------------------------------------
# Explicabilidad de escenario — palanca a palanca
# --------------------------------------------------------------------------
ETIQUETAS_PALANCA = {
    "factor_demanda": "Sube la demanda",
    "factor_lead_time": "Alarga el tiempo de entrega",
    "factor_desv_lead_time": "Vuelve al proveedor más irregular",
    "nivel_servicio": "Exige más nivel de servicio",
    "tipo_politica": "Cambia el tipo de política",
    "periodo_revision_dias": "Cambia el período de revisión",
    "fraccion_presupuesto": "Limita el presupuesto",
}


def descomponer_escenario(estado, codigo, dias=None, n_replicas=8, semilla_base=52000):
    """
    Activa las palancas de un escenario de a una para ver cuánto aporta cada
    una, y compara la suma de las partes contra el efecto del paquete.

    Responde lo que ninguna tabla de resultados responde sola: el escenario
    «el proveedor se demora el doble» empeora todo, ¿pero por el retraso medio
    o por la irregularidad? Son dos problemas con dos soluciones distintas —
    uno se arregla pidiendo antes, el otro solo con más stock de seguridad.
    """
    dias = dias or EJECUCION["dias_simulacion"]
    cfg = ESCENARIOS.get(codigo)
    if not cfg or not cfg["palancas"]:
        return None
    palancas = cfg["palancas"]

    def costo_medio(activas):
        corridas = [simular(estado, activas, dias=dias, semilla=semilla_base + i * 613)
                    for i in range(n_replicas)]
        return (float(np.mean([c["costo_total"] for c in corridas])),
                float(np.mean([c["fill_rate"] for c in corridas])))

    costo_base, fill_base = costo_medio({})
    filas, suma = [], 0.0
    for nombre, valor in palancas.items():
        costo, fill = costo_medio({nombre: valor})
        efecto = costo - costo_base
        suma += efecto
        filas.append({
            "Palanca": ETIQUETAS_PALANCA.get(nombre, nombre),
            "clave": nombre,
            "Valor": valor,
            "Costo con solo esta palanca (S/.)": round(costo, 0),
            "Efecto sobre el costo (S/.)": round(efecto, 0),
            "Fill rate (%)": round(fill, 2),
        })

    costo_conjunto, fill_conjunto = costo_medio(palancas)
    efecto_conjunto = costo_conjunto - costo_base
    interaccion = efecto_conjunto - suma
    for f in filas:
        f["Participación (%)"] = (round(f["Efecto sobre el costo (S/.)"] / suma * 100, 1)
                                  if abs(suma) > 1e-9 else 0.0)
    filas.sort(key=lambda d: -abs(d["Efecto sobre el costo (S/.)"]))

    return {
        "codigo": codigo, "nombre": cfg["nombre"], "pregunta": cfg["pregunta"],
        "costo_base": costo_base, "fill_base": fill_base,
        "costo_conjunto": costo_conjunto, "fill_conjunto": fill_conjunto,
        "efecto_conjunto": efecto_conjunto, "suma_individual": suma,
        "interaccion": interaccion,
        "tipo_interaccion": ("Se amplifican entre sí" if interaccion > abs(suma) * 0.10
                             else "Se solapan" if interaccion < -abs(suma) * 0.10
                             else "Aproximadamente aditivas"),
        "filas": filas, "n_replicas": n_replicas,
    }
