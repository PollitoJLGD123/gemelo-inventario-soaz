# modulo_simulacion/narrativa.py
"""
Interpretabilidad y explicabilidad de cada figura y tabla del motor.

REGLA DEL MÓDULO: ninguna figura ni tabla se muestra sola. Toda salida del
gemelo va acompañada de un bloque de tres partes, y las tres se GENERAN a
partir de los números de esa corrida concreta — no son texto fijo:

  QUÉ MUESTRA        interpretabilidad: cómo se lee, en qué unidades, qué
                     significan los ejes y los umbrales.
  POR QUÉ SALE ASÍ   explicabilidad: qué mecanismo del modelo produjo ese
                     resultado y no otro.
  QUÉ NO SE PUEDE    el límite honesto de la evidencia: qué pregunta parece
  CONCLUIR           responder la figura pero no responde.

La tercera parte es la que suele faltar y la que más protege en una
sustentación: adelantarse a la objeción vale más que esperarla.
"""
import numpy as np
import streamlit as st

from .parametros import (
    FIDELIDAD, NEGOCIO, COSTOS, PROCEDENCIA_RESUMEN, CATALOGO, EJECUCION,
)

MONEDA = NEGOCIO["moneda"]


# ==========================================================================
# RENDERIZADO ESTÁNDAR
# ==========================================================================
def bloque_interpretacion(interp, expandido=False):
    with st.expander("🔎 Interpretación y explicabilidad", expanded=expandido):
        st.markdown(f"**Qué muestra.** {interp['que_muestra']}")
        st.markdown(f"**Por qué sale así.** {interp['por_que']}")
        st.markdown(f"**Qué NO se puede concluir.** {interp['limitacion']}")
        if interp.get("glosario"):
            st.markdown("**Glosario**")
            for termino, definicion in interp["glosario"].items():
                st.markdown(f"- `{termino}` — {definicion}")
        if interp.get("nota_metodo"):
            st.caption(f"Método: {interp['nota_metodo']}")


def mostrar_figura(fig, titulo, interp, expandido=False, clave=None):
    """Figura + su interpretación. Es la única forma de mostrar una figura."""
    st.markdown(f"#### {titulo}")
    st.plotly_chart(fig, use_container_width=True, key=clave)
    bloque_interpretacion(interp, expandido)


def mostrar_tabla(df, titulo, interp, expandido=False, **kwargs):
    """Tabla + su interpretación. Es la única forma de mostrar una tabla."""
    st.markdown(f"#### {titulo}")
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
    bloque_interpretacion(interp, expandido)


def _s(x):
    return f"{MONEDA} {x:,.0f}"


# ==========================================================================
# INTERPRETACIONES
# ==========================================================================
def interpretar_abc(filas, n_total):
    a = next((f for f in filas if f["Clase"] == "A"), None)
    if not a:
        return {"que_muestra": "Sin datos.", "por_que": "—", "limitacion": "—"}
    return {
        "que_muestra": (
            f"La clasificación ABC del catálogo por valor de movimiento anual. "
            f"La clase A son **{a['SKU']} SKU** ({a['% del catálogo']}% del catálogo) "
            f"y concentran el **{a['% del movimiento']}%** del movimiento. Son los "
            f"únicos que entran al gemelo."
        ),
        "por_que": (
            f"Modelar los {n_total} SKU multiplicaría el trabajo sin cambiar ninguna "
            f"conclusión: el {100 - a['% del movimiento']:.0f}% restante del movimiento "
            f"se reparte entre productos cuyo costo de ruptura individual es marginal. "
            f"El criterio de corte es el {CATALOGO['umbral_clase_a'] * 100:.0f}% de "
            f"participación acumulada, declarado antes de mirar los datos."
        ),
        "limitacion": (
            "El ABC clasifica por valor, no por criticidad. Un producto barato pero "
            "indispensable —que hace que el cliente se lleve toda la compra a otro "
            "lado si falta— puede ser clase C y merecer igual atención. Esa dimensión "
            "no está en el modelo y debe consultarse con la representante del negocio."
        ),
    }


def interpretar_censura(comparacion, resumen_hist):
    r = comparacion["resumen"]
    ing, em = r["ingenuo"], r["em"]
    mejora = abs(ing["sesgo_medio"]) - abs(em["sesgo_medio"])
    perdida = resumen_hist.get("Demanda no observada (%)", 0)
    return {
        "que_muestra": (
            f"El sesgo de tres estimadores de demanda frente a la demanda verdadera. "
            f"El promedio simple de ventas subestima la demanda en "
            f"**{abs(ing['sesgo_medio']):.1f}%** en promedio y hasta "
            f"**{abs(ing['peor_sesgo']):.1f}%** en el peor SKU. La corrección por "
            f"censura (EM) baja el sesgo a **{abs(em['sesgo_medio']):.1f}%**."
        ),
        "por_que": (
            f"Las ventas no son la demanda. En el histórico, **{perdida}% de la demanda "
            f"nunca se registró**: los días con stock en cero, el cliente que no fue "
            f"atendido no dejó rastro. Ajustar sobre ventas crudas propaga ese hueco a "
            f"la media, el gemelo calcula un stock de seguridad bajo y termina "
            f"recomendando comprar de menos — reproduciendo el problema que vino a "
            f"resolver. EM trata esos días como observaciones censuradas (se sabe que "
            f"la demanda fue *al menos* lo vendido) y reconstruye su esperanza."
        ),
        "limitacion": (
            "La comparación es posible porque aquí la demanda verdadera se conoce. "
            "En el proyecto real no se conoce, así que la corrección no se puede "
            "verificar de esta forma: se justifica por el método, no por el resultado. "
            "Además, el marcado de censura usa 'stock cerró en cero', que no detecta "
            "una ruptura de media tarde repuesta al día siguiente."
        ),
        "nota_metodo": f"Mejora del sesgo absoluto: {mejora:.1f} puntos porcentuales.",
    }


def interpretar_corrida(res):
    ic = res["costo_total_ic"]
    desglose = [("ruptura", res["costo_ruptura"]), ("mantener", res["costo_mantener"]),
                ("merma", res["costo_merma"]), ("ordenar", res["costo_ordenar"])]
    dominante = max(desglose, key=lambda d: d[1])
    pct = dominante[1] / max(res["costo_total"], 1e-9) * 100
    return {
        "que_muestra": (
            f"Resultado del escenario **{res['nombre']}** a {res['dias']} días sobre "
            f"{res['n_sku']} SKU de clase A, promediado sobre **{res['n_replicas']} "
            f"réplicas**. Fill rate {res['fill_rate']:.2f}%, costo total "
            f"{_s(res['costo_total'])} con intervalo del 95% "
            f"[{ic[0]:,.0f}, {ic[1]:,.0f}]. El fill rate se mide en unidades atendidas "
            f"sobre unidades demandadas, no en ciclos sin ruptura."
        ),
        "por_que": (
            f"El componente dominante del costo es **{dominante[0]}** con "
            f"{_s(dominante[1])} ({pct:.0f}% del total). "
            + ("La ruptura domina porque la penalidad se valora a "
               f"{COSTOS['penalidad_ruptura_factor']:.2f}× el costo del producto: el "
               "margen perdido más el riesgo de que el cliente no vuelva."
               if dominante[0] == "ruptura" else
               "La merma domina porque hay familias perecibles cuyo lote se compra por "
               "más días de los que el producto aguanta; ahí el problema no es comprar "
               "poco, es comprar demasiado de una vez."
               if dominante[0] == "merma" else
               "El costo de mantener domina, señal de que el inventario está por encima "
               "de lo que la variabilidad exige."
               if dominante[0] == "mantener" else
               "El costo de ordenar domina, señal de lotes demasiado pequeños y "
               "compras demasiado frecuentes.")
        ),
        "limitacion": (
            "La cohorte de SKU y el histórico son sintéticos: describen el "
            "comportamiento del **modelo**, no el de SOAZ. Comparar escenarios entre sí "
            "es legítimo; leer estos soles como el costo real del negocio no lo es "
            "hasta reemplazar el histórico por los movimientos del sistema."
        ),
        "nota_metodo": f"{res['n_replicas']} réplicas de {res['dias']} días, "
                       f"política {res['politica']['tipo']}.",
    }


def interpretar_diente_sierra(sku, serie):
    stocks = [p["stock"] for p in serie]
    rop = serie[0]["punto_reorden"]
    ss = serie[0]["stock_seguridad"]
    bajo_ss = sum(1 for s in stocks if s < ss)
    en_cero = sum(1 for s in stocks if s <= 0)
    return {
        "que_muestra": (
            f"Trayectoria del stock de **{sku}** en una réplica. La línea baja con la "
            f"venta diaria y salta cuando llega el pedido: es el diente de sierra "
            f"clásico del inventario. Las dos referencias horizontales son el punto de "
            f"reorden ({rop:,.0f} u) —donde se dispara la compra— y el stock de "
            f"seguridad ({ss:,.0f} u) —el colchón contra la variabilidad."
        ),
        "por_que": (
            f"El stock cayó por debajo del stock de seguridad en **{bajo_ss} de "
            f"{len(stocks)} días** y llegó a cero en **{en_cero}**. Que toque el colchón "
            f"es normal y es para lo que existe; que llegue a cero significa que la "
            f"combinación de demanda alta y entrega lenta superó lo que el colchón "
            f"cubría. La distancia entre el punto de reorden y el cero es el tiempo de "
            f"reacción real que tiene el negocio."
        ),
        "limitacion": (
            "Es **una** réplica. Con otra semilla, el mismo SKU con los mismos "
            "parámetros produce otro diente: los días exactos de quiebre no son "
            "predicciones. Para conclusiones hay que mirar los agregados sobre "
            "réplicas, no esta curva."
        ),
    }


def interpretar_comparacion(comp):
    filas = comp["tabla"]
    base = next((f for f in filas if f["Código"] == "Base"), None)
    otros = [f for f in filas if f["Código"] != "Base"]
    distinguibles = [f for f in otros if f["¿Distinguible del Base?"] == "Sí"]
    no_dist = [f for f in otros if f["¿Distinguible del Base?"] == "No"]
    peor = max(otros, key=lambda f: f["Δ costo vs Base (S/.)"]) if otros else None
    mejor = min(otros, key=lambda f: f["Δ costo vs Base (S/.)"]) if otros else None

    partes = []
    if peor:
        partes.append(
            f"El escenario más caro es **{peor['Escenario']}**, con "
            f"{_s(peor['Δ costo vs Base (S/.)'])} más que la política actual y "
            f"{abs(peor['Δ fill vs Base (pp)']):.2f} pp de fill rate perdido.")
    if mejor and mejor["Δ costo vs Base (S/.)"] < 0:
        partes.append(
            f"**{mejor['Escenario']}** es el único que ahorra "
            f"({_s(abs(mejor['Δ costo vs Base (S/.)']))}), y lo hace subiendo el "
            f"inventario: el costo de la ruptura evitada supera al de mantener más "
            f"stock. Es contraintuitivo y es el tipo de hallazgo que justifica "
            f"tener un gemelo.")
    if no_dist:
        partes.append(
            f"En cambio {', '.join(f['Escenario'] for f in no_dist)} produce diferencias "
            f"que **caben dentro del ruido del modelo**: sus intervalos se solapan con "
            f"el del Base, así que con {EJECUCION['n_replicas']} réplicas no se puede "
            f"afirmar que cambien algo.")

    return {
        "que_muestra": (
            f"Los {len(filas)} escenarios comparados. La columna decisiva no es el costo "
            f"sino **«¿Distinguible del Base?»**: dice si la diferencia sobrevive a la "
            f"incertidumbre del modelo. {len(distinguibles)} de {len(otros)} escenarios "
            f"la superan."
        ),
        "por_que": " ".join(partes) if partes else "Sin escenarios comparables.",
        "limitacion": (
            "Cada escenario cambia UNA condición y deja el resto igual. En la realidad "
            "las condiciones se mueven juntas —una campaña sube la demanda y también "
            "satura al proveedor— y el gemelo no lo captura salvo que se declare un "
            "escenario combinado. Los costos tampoco incluyen la gestión de compras "
            "ni el transporte, que están fuera del alcance."
        ),
        "glosario": {
            "Fill rate": "Unidades atendidas sobre unidades demandadas.",
            "IC 95% costo": "Intervalo donde estaría el costo en 95 de cada 100 conjuntos de réplicas.",
            "¿Distinguible del Base?": "Sí solo si el intervalo del escenario no se solapa con el del Base.",
            "Capital inmovilizado": "Valor promedio del stock a la mano durante el horizonte.",
        },
    }


def interpretar_servicio_costo(filas):
    por_pol = {}
    for f in filas:
        por_pol.setdefault(f["Política"], []).append(f)
    partes = []
    for pol, fs in por_pol.items():
        mejor = min(fs, key=lambda f: f["Costo total (S/.)"])
        partes.append(f"en **{pol}** el mínimo de costo está en un objetivo de "
                      f"{mejor['Nivel de servicio objetivo']} "
                      f"({_s(mejor['Costo total (S/.)'])})")
    sq = [f for f in filas if f["Política"] == "(s,Q)"]
    rs = [f for f in filas if f["Política"] == "(R,S)"]
    brecha = ""
    if sq and rs:
        m_sq = min(f["Costo total (S/.)"] for f in sq)
        m_rs = min(f["Costo total (S/.)"] for f in rs)
        brecha = (f" La revisión continua sale {_s(abs(m_sq - m_rs))} "
                  f"{'más barata' if m_sq < m_rs else 'más cara'} que la periódica en su "
                  f"mejor punto.")
    return {
        "que_muestra": (
            "Costo total frente al nivel de servicio objetivo, para las dos políticas. "
            "Cada punto es un objetivo distinto de z; el eje X es el fill rate "
            "efectivamente logrado, que no coincide con el objetivo porque el objetivo "
            "es por ciclo y el fill rate se mide en unidades."
        ),
        "por_que": (
            f"La curva no es monótona: {', '.join(partes)}.{brecha} Subir el nivel de "
            f"servicio encarece el inventario pero abarata la ruptura, y hasta cierto "
            f"punto lo segundo pesa más. Ese mínimo es la recomendación del gemelo, y "
            f"depende enteramente de la penalidad de ruptura declarada "
            f"({COSTOS['penalidad_ruptura_factor']:.2f}× el costo del producto): si esa "
            f"penalidad cambia, el mínimo se mueve."
        ),
        "limitacion": (
            "El mínimo de esta curva es óptimo **bajo los costos supuestos**. La "
            "penalidad de ruptura es el parámetro menos observable de todos —nadie "
            "registra al cliente que se fue— y es el que más mueve el resultado. Antes "
            "de llevar esta recomendación a SOAZ hay que acordar ese número con la "
            "representante del negocio."
        ),
    }


def interpretar_fidelidad(val):
    r = val["resumen"]
    cumplidos = [c for c in r["criterios"] if c["Cumple"]]
    fallidos = [c for c in r["criterios"] if not c["Cumple"]]
    pisos = r["pisos"]
    return {
        "que_muestra": (
            f"La retro-prueba: el gemelo se congeló en el día {r['dia_corte']} usando "
            f"solo los movimientos anteriores, corrió {r['dias_prueba']} días hacia "
            f"adelante con la política que SOAZ **realmente aplicó**, y se comparó "
            f"contra el registro. Resultado: **{len(cumplidos)} de "
            f"{r['n_criterios']} criterios cumplidos**, veredicto "
            f"**{'PASA' if r['veredicto'] else 'NO PASA'}**. Los umbrales se fijaron "
            f"antes de correr la prueba."
        ),
        "por_que": (
            f"Dos de los criterios se expresan **relativos a un piso irreducible**, y "
            f"esa decisión metodológica es la más importante de esta pantalla. El piso "
            f"se obtiene comparando el gemelo contra SÍ MISMO con otra semilla: da "
            f"{pisos['mape']:.0f}% de MAPE y {pisos['dias_ruptura']:.1f} días de "
            f"diferencia en rupturas. Los umbrales originales del proyecto —20% de MAPE "
            f"y 2 días— estaban **por debajo de ese piso**, o sea que eran imposibles de "
            f"cumplir incluso para un modelo perfecto. Un umbral que ni el modelo "
            f"verdadero puede alcanzar no es exigente: está mal formulado. "
            f"El gemelo obtiene {r['mape_relativo']:.2f}× el piso de MAPE y "
            f"{r['ruptura_relativa']:.2f}× el de rupturas."
        ),
        "limitacion": (
            "La retro-prueba valida que el gemelo reproduce el pasado **bajo la política "
            "que ya se usaba**. No prueba que acierte bajo una política nueva: "
            "extrapolar a un régimen no observado es un salto que la validación no "
            "cubre. "
            + ("Además, los criterios que fallan deben reportarse en el informe con su "
               "diagnóstico: " + ", ".join(c["Criterio"] for c in fallidos) + "."
               if fallidos else
               "Y sigue apoyada en un histórico sintético: con datos reales de SOAZ el "
               "resultado puede ser otro.")
        ),
        "nota_metodo": (
            f"{r['n_replicas']} réplicas, estimador de demanda «{r['metodo_demanda']}», "
            f"pisos calculados con {pisos['n_pares']} pares de réplicas."
        ),
    }


def interpretar_trayectoria_validacion(fila):
    return {
        "que_muestra": (
            f"Stock real registrado de **{fila['SKU']}** contra la banda que el gemelo "
            f"predijo sin haber visto esos días. La línea es la media de las réplicas y "
            f"la banda es el 95%. Lo real cae dentro de la banda el "
            f"**{fila['Cobertura de banda (%)']}%** de los días."
        ),
        "por_que": (
            f"El nivel medio coincide dentro de {fila['Error nivel medio (%)']}%, pero el "
            f"MAPE punto a punto es {fila['MAPE stock (%)']}%. No es contradictorio: el "
            f"stock es un diente de sierra y el día exacto del salto depende del tiempo "
            f"de entrega, que es aleatorio. Al promediar réplicas los saltos se "
            f"desalinean y la media se suaviza, así que compararla punto a punto contra "
            f"un diente real castiga un desfase que ningún modelo puede eliminar. Por "
            f"eso la métrica que decide es la cobertura de la banda, no el MAPE."
        ),
        "limitacion": (
            "Un SKU aislado puede coincidir por casualidad o fallar por una racha. El "
            "veredicto de fidelidad se toma sobre el promedio de la clase A, no sobre "
            "un producto elegido a mano."
        ),
    }


def interpretar_shapley(ex, fidelidad=None):
    tabla = ex["tabla"]
    principal = tabla[0]
    abaratan = [f for f in tabla if f["Contribución (S/.)"] < 0]
    d = ex["desglose"]
    comp = max([("ordenar", d["costo_ordenar"]), ("mantener", d["costo_mantener"]),
                ("ruptura", d["costo_ruptura"]), ("merma", d["costo_merma"])],
               key=lambda x: x[1])
    nota_fid = ""
    if fidelidad:
        nota_fid = (f" El modelo analítico que se explica se aparta de la simulación en "
                    f"{fidelidad['error_absoluto_medio']:.0f}% en promedio, con "
                    f"correlación {fidelidad['correlacion']:.2f}: "
                    + ("es un sustituto fiel y la explicación es válida."
                       if fidelidad["apto"] else
                       "**la explicación debe tomarse con reservas**."))
    return {
        "que_muestra": (
            f"Por qué **{ex['sku']}** cuesta {_s(ex['valor_sku'])} al año y no los "
            f"{_s(ex['valor_base'])} del SKU promedio. Cada barra es la contribución "
            f"exacta de un factor a esa diferencia de "
            f"{_s(ex['valor_sku'] - ex['valor_base'])}. El que más pesa es "
            f"**{principal['Factor'].lower()}** ({_s(principal['Contribución (S/.)'])}), "
            f"y {len(abaratan)} factores lo abaratan."
        ),
        "por_que": (
            f"Su componente de costo dominante es **{comp[0]}** ({_s(comp[1])}). Las "
            f"contribuciones son valores de Shapley **exactos**: se enumeraron las "
            f"{ex['n_coaliciones']} coaliciones posibles de los 9 factores, sin "
            f"muestreo y sin aproximación. La verificación del axioma de eficiencia da "
            f"un error de {ex['error_eficiencia']:.1e}, es decir, las contribuciones "
            f"suman exactamente la diferencia que explican.{nota_fid}"
        ),
        "limitacion": (
            "Una contribución de Shapley responde «por qué este SKU y no el promedio», "
            "no «qué pasa si cambio este factor». Para lo segundo hay que simular, "
            "porque los factores están acoplados: bajar el lote de compra cambia la "
            "frecuencia de pedido y con ella la exposición a la ruptura. Además, "
            "cambiar el SKU de referencia cambiaría todas las contribuciones."
        ),
        "nota_metodo": (
            f"Shapley exacto sobre {ex['n_coaliciones']} coaliciones del modelo "
            f"analítico de costo. Referencia: SKU promedio del alcance."
        ),
    }


def interpretar_tornado(tn):
    top, cola = tn["filas"][0], tn["filas"][-1]
    segundo = tn["filas"][1] if len(tn["filas"]) > 1 else None
    razon = ""
    if segundo and segundo["Amplitud (S/.)"]:
        razon = (f" — mueve el costo {top['Amplitud (S/.)'] / segundo['Amplitud (S/.)']:.1f} "
                 f"veces más que {segundo['Factor'].lower()}, el siguiente de la lista")
    return {
        "que_muestra": (
            f"Cuánto puede mover el costo cada factor **por sí solo**, al pasar del "
            f"percentil 10 al 90 de la población de SKU mientras los demás quedan en su "
            f"valor promedio. El dominante es **{top['Factor']}** con "
            f"{_s(top['Amplitud (S/.)'])} de amplitud; el menor es "
            f"{cola['Factor'].lower()} con {_s(cola['Amplitud (S/.)'])}."
        ),
        "por_que": (
            f"{top['Factor']}{razon}. Un factor puede tener un peso grande en la fórmula "
            f"y aun así salir corto aquí si casi todos los SKU tienen valores parecidos: "
            f"el tornado mide peso **por** variabilidad, que es lo que importa para "
            f"decidir sobre qué vale la pena actuar. De ahí sale la prioridad: negociar "
            f"con el proveedor rinde si su variabilidad está arriba en esta lista; "
            f"ajustar lotes rinde si el que está arriba es el tamaño de compra."
        ),
        "limitacion": (
            "El análisis mueve un factor a la vez, así que **no captura interacciones**: "
            "si dos factores solo importan combinados, ambos aparecen subestimados. Para "
            "el reparto que sí incluye interacciones está la descomposición de Shapley "
            "por SKU. Y mide sensibilidad del modelo analítico, no evidencia empírica."
        ),
    }


def interpretar_descomposicion(d):
    top = d["filas"][0]
    return {
        "que_muestra": (
            f"De qué está hecho el efecto de **{d['nombre']}**. La pregunta del negocio "
            f"era: *{d['pregunta']}* El escenario completo cambia el costo en "
            f"{_s(d['efecto_conjunto'])} (de {_s(d['costo_base'])} a "
            f"{_s(d['costo_conjunto'])}) y el fill rate de {d['fill_base']:.2f}% a "
            f"{d['fill_conjunto']:.2f}%. Activando las palancas de a una, la que más "
            f"pesa es **{top['Palanca'].lower()}**, con {top['Participación (%)']}% "
            f"del efecto."
        ),
        "por_que": (
            f"La suma de las palancas por separado da {_s(d['suma_individual'])}, pero "
            f"juntas rinden {_s(d['efecto_conjunto'])}: una diferencia de "
            f"{_s(d['interaccion'])} que el modelo clasifica como "
            f"**{d['tipo_interaccion'].lower()}**. "
            + ("Se solapan: cada palanca golpea a los mismos SKU, y una vez que el "
               "primero los puso en ruptura, el segundo ya no encuentra qué empeorar."
               if d["interaccion"] < 0 else
               "Se amplifican: cada palanca hace al almacén más vulnerable a la "
               "siguiente, que es el comportamiento típico cuando el stock de seguridad "
               "ya no alcanza."
               if d["interaccion"] > 0 else
               "Son prácticamente independientes en este rango.")
            + " Saber cuál palanca domina es lo que separa una recomendación útil de un "
              "diagnóstico genérico: si domina el retraso medio, se corrige pidiendo "
              "antes; si domina la irregularidad, solo se corrige con más colchón o "
              "cambiando de proveedor."
        ),
        "limitacion": (
            f"La atribución se hizo con {d['n_replicas']} réplicas por palanca, así que "
            f"diferencias pequeñas frente al intervalo no son distinguibles del ruido. "
            f"Además el reparto «una a la vez» no es un valor de Shapley sobre palancas: "
            f"sería más neutral pero mucho más costoso de calcular."
        ),
    }


def interpretar_recomendaciones(filas):
    urgentes = [f for f in filas if f["Urgencia"] == "Inmediata"]
    semana = [f for f in filas if f["Urgencia"] == "Esta semana"]
    valor = sum(f["Valor de la orden (S/.)"] for f in filas)
    return {
        "que_muestra": (
            f"El lazo de retorno del gemelo hacia el sistema real: qué comprar, cuánto y "
            f"con qué urgencia. **{len(urgentes)} SKU requieren orden inmediata** "
            f"(su cobertura es menor que el tiempo de entrega del proveedor) y "
            f"{len(semana)} pueden esperar a la revisión semanal. El total sugerido es "
            f"{_s(valor)}."
        ),
        "por_que": (
            "Cada fila lleva su razón en la columna «Por qué», y eso no es un adorno: "
            "una recomendación sin razón no se aprueba, se ignora. La urgencia se decide "
            "comparando la cobertura actual contra el tiempo de entrega del proveedor —"
            "no contra una regla fija de días—, porque un producto con proveedor local "
            "de 2 días puede esperar más que uno con distribuidor regional de 7."
        ),
        "limitacion": (
            "Estas cantidades son sugerencias, no órdenes: el encargado aprueba o "
            "rechaza, y su tasa de aceptación es lo que hay que medir para saber si el "
            "gemelo sirve. El gemelo tampoco conoce restricciones que sí conoce el "
            "negocio —una promoción del proveedor, un problema de liquidez, un cambio de "
            "surtido— y por eso el humano se queda en el lazo."
        ),
        "glosario": {
            "Cobertura actual": "Días que alcanza el stock a la demanda media estimada.",
            "Punto de reorden": "Nivel que dispara la compra: demanda durante el lead time más el colchón.",
            "Stock de seguridad": "Colchón contra la variabilidad de demanda Y de tiempo de entrega.",
        },
    }


def interpretar_procedencia(conteo):
    total = sum(conteo.values())
    supuestos = conteo.get("SUPUESTO", 0)
    return {
        "que_muestra": (
            f"El origen declarado de cada parámetro del modelo. De {total} parámetros "
            f"documentados, **{supuestos} son supuestos del autor** todavía sin "
            f"validar con el negocio."
        ),
        "por_que": (
            "Separar teoría, dato estimado y supuesto permite responder «¿de dónde sale "
            "ese número?» sin improvisar. Los TEORIA se derivan de la teoría de "
            "inventarios; los ESTIMADO salen del histórico corrigiendo por censura; los "
            "SUPUESTO son decisiones del autor y son exactamente los que el jurado debe "
            "poder cuestionar sin que se caiga el modelo."
        ),
        "limitacion": (
            "Un parámetro documentado no es un parámetro validado. Mientras predominen "
            "los supuestos —y el más delicado es la penalidad de ruptura— el gemelo "
            "sirve para comparar escenarios bajo reglas explícitas, no para pronosticar "
            "el costo real de SOAZ."
        ),
        "glosario": PROCEDENCIA_RESUMEN,
    }


def interpretar_escenario_adhoc(resultados, propuesta, cerco=None):
    """
    Interpretación de un escenario nacido de una pregunta, no de la lista fija.

    Lleva una parte que ninguna otra interpretación del motor necesita: decir de
    DÓNDE salió el escenario. Los de parametros.ESCENARIOS los escribió una
    persona y quedaron revisados; este lo propuso un modelo de lenguaje a partir
    de una frase y lo aprobó un humano en pantalla. Quien lea el resultado tiene
    derecho a saber eso, y el informe tiene que poder citarlo.

    El resto es lo de siempre: números de esta corrida, nunca texto fijo.
    """
    base, prop = resultados["base"], resultados["propuesto"]
    ic_b, ic_p = base["costo_total_ic"], prop["costo_total_ic"]
    distinguible = resultados["distinguible"]
    delta = prop["costo_total"] - base["costo_total"]
    delta_fill = prop["fill_rate"] - base["fill_rate"]
    palancas = propuesta.get("palancas") or {}
    lista = ", ".join(f"`{k}` = {v}" for k, v in palancas.items()) or "ninguna"

    if distinguible:
        veredicto = (
            f"La diferencia de {_s(abs(delta))} "
            f"{'más caro' if delta > 0 else 'de ahorro'} **sobrevive al ruido del "
            f"modelo**: el intervalo del escenario [{ic_p[0]:,.0f}, {ic_p[1]:,.0f}] no "
            f"se solapa con el del Base [{ic_b[0]:,.0f}, {ic_b[1]:,.0f}]. Con "
            f"{prop['n_replicas']} réplicas se puede afirmar que el escenario cambia "
            f"el costo, no solo que salió distinto esta vez.")
    else:
        veredicto = (
            f"La diferencia de {_s(abs(delta))} **cabe dentro del ruido del modelo**: "
            f"los intervalos [{ic_p[0]:,.0f}, {ic_p[1]:,.0f}] y "
            f"[{ic_b[0]:,.0f}, {ic_b[1]:,.0f}] se solapan. Con {prop['n_replicas']} "
            f"réplicas NO se puede afirmar que este escenario cambie el costo. Subir "
            f"las réplicas estrecha los intervalos; si aun así se solapan, la respuesta "
            f"honesta a la pregunta es «no se nota».")

    if delta > 0 and delta_fill > 0:
        mecanismo = (
            f"El escenario cuesta más y además sirve mejor ({delta_fill:+.2f} pp de fill "
            f"rate): está comprando servicio con capital, que es el intercambio central "
            f"del inventario.")
    elif delta < 0 and delta_fill < 0:
        mecanismo = (
            f"El escenario ahorra pero sirve peor ({delta_fill:+.2f} pp de fill rate): "
            f"el ahorro no es gratis, sale del cliente que no fue atendido.")
    elif delta < 0 and delta_fill >= 0:
        mecanismo = (
            f"El escenario ahorra sin perder servicio ({delta_fill:+.2f} pp). Cuando eso "
            f"pasa suele ser porque la política anterior mantenía stock que la "
            f"variabilidad no exigía, o porque menos inventario significó menos merma.")
    else:
        mecanismo = (
            f"El escenario cuesta más y sirve peor ({delta_fill:+.2f} pp): las dos cosas "
            f"empeoran a la vez, señal de que la palanca movida degrada la operación en "
            f"lugar de intercambiar una cosa por otra.")

    procedencia = (
        "Este escenario **no estaba declarado en el modelo**: un modelo de lenguaje "
        "tradujo una pregunta en palancas y una persona las aprobó en pantalla antes de "
        "simular. Las cifras las produjo el motor de siempre, con las mismas semillas y "
        "las mismas réplicas; el modelo de lenguaje no calculó ninguna.")

    if cerco is None:
        nota_cerco = ""
    elif cerco.get("ok"):
        nota_cerco = (f" El texto de acompañamiento pasó el cerco numérico: sus "
                      f"{cerco['verificados']} cifras están todas en los resultados.")
    else:
        inventadas = ", ".join(i["literal"] for i in cerco.get("intrusos", []))
        nota_cerco = (f" El texto de acompañamiento fue **descartado** por el cerco "
                      f"numérico: las cifras {inventadas} no aparecen en ningún "
                      f"resultado. Lo que se lee arriba es solo esta interpretación.")

    return {
        "que_muestra": (
            f"El escenario **{prop['nombre']}**, construido desde la pregunta "
            f"«{prop['pregunta']}». Mueve {lista}. Se corrió contra el Base a "
            f"{prop['dias']} días sobre {prop['n_sku']} SKU de clase A, con "
            f"**{prop['n_replicas']} réplicas** cada uno: costo {_s(prop['costo_total'])} "
            f"contra {_s(base['costo_total'])} del Base, fill rate "
            f"{prop['fill_rate']:.2f}% contra {base['fill_rate']:.2f}%."
        ),
        "por_que": f"{veredicto} {mecanismo}",
        "limitacion": (
            f"{procedencia} Dicho eso, siguen valiendo todos los límites del motor: el "
            f"histórico es sintético, así que estos soles describen el modelo y no a "
            f"SOAZ; el escenario mueve las palancas declaradas y deja el resto igual, "
            f"cuando en la realidad las condiciones se mueven juntas; y la penalidad de "
            f"ruptura sigue siendo el supuesto más delicado.{nota_cerco}"
        ),
        "glosario": {
            "Escenario ad-hoc": "Uno que no está en parametros.ESCENARIOS: nació de una "
                                "pregunta y fue aprobado por una persona antes de correr.",
            "¿Distinguible del Base?": "Sí solo si el intervalo del escenario no se "
                                       "solapa con el del Base.",
            "Cerco numérico": "Verificación de que toda cifra del texto generado "
                              "aparece en los resultados del motor.",
        },
        "nota_metodo": (f"{prop['n_replicas']} réplicas de {prop['dias']} días por "
                        f"escenario, política {prop['politica']['tipo']}. Palancas: {lista}."),
    }
