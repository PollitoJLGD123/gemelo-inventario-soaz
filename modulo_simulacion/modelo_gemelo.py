# modulo_simulacion/modelo_gemelo.py
"""
Interfaz Streamlit del motor del Gemelo Digital de Inventario — SOAZ.

REGLA DE ESTA INTERFAZ: ninguna figura ni tabla se dibuja sin su bloque de
interpretación y explicabilidad. El renderizado pasa siempre por
narrativa.mostrar_figura() o narrativa.mostrar_tabla(), que lo garantizan por
construcción en lugar de dejarlo a la disciplina de quien escribe la pantalla.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from . import narrativa
from .catalogo import clasificar_abc, catalogo_en_alcance, resumen_abc
from .datos_historicos import generar_historico, resumen_historico, PRACTICA_ACTUAL
from .escenarios import (
    ejecutar_escenario, comparar_escenarios, comparar_politicas,
    recomendaciones_reposicion,
)
from .estimacion import construir_estado_gemelo, comparar_estimadores
from .explicabilidad import (
    explicar_sku, tornado_sensibilidad, descomponer_escenario, fidelidad_sustituto,
)
from .motor_des import simular, gasto_reposicion_mensual
from .parametros import (
    ESCENARIOS, EJECUCION, NEGOCIO, CATALOGO, FIDELIDAD, COSTOS, PROCEDENCIA_RESUMEN,
)
from .politicas import construir_politica
from .validacion import retro_prueba, comparar_fidelidad_por_metodo

# Paleta semántica, no decorativa
COLOR = {
    "stock": "#1d4ed8", "reorden": "#b45309", "seguridad": "#0f766e",
    "riesgo": "#c2410c", "riesgo_suave": "rgba(194,65,12,0.15)",
    "ok": "#15803d", "neutro": "#64748b", "banda": "rgba(29,78,216,0.15)",
}
PLANTILLA = "plotly_white"
MONEDA = NEGOCIO["moneda"]


# ==========================================================================
# CACHÉ — el histórico y el estado se construyen una sola vez
# ==========================================================================
@st.cache_data(show_spinner=False)
def _mundo(semilla, dias_historico, metodo_demanda):
    """
    El catálogo y la demanda salen de Favorita (tienda 44). La semilla solo
    mueve los tiempos de entrega de la reconstrucción, que el dataset no trae.
    """
    from .favorita import cargar_alcance, catalogo_desde_panel, demanda_observada

    panel = cargar_alcance()
    completo = clasificar_abc(catalogo_desde_panel(panel))
    catalogo = catalogo_en_alcance(completo)
    items = {int(s["item_nbr"]) for s in catalogo}
    fechas = sorted(pd.to_datetime(panel["date"].unique()))
    if len(fechas) > dias_historico:
        fechas = fechas[-int(dias_historico):]
    ventana = panel[panel["item_nbr"].isin(items) & panel["date"].isin(fechas)]
    movimientos, verdad = generar_historico(
        catalogo, semilla=semilla,
        demanda_observada=demanda_observada(ventana, fechas))
    estado = construir_estado_gemelo(catalogo, movimientos, metodo_demanda=metodo_demanda)
    return completo, catalogo, movimientos, verdad, estado, panel[panel["item_nbr"].isin(items)].copy()


@st.cache_data(show_spinner=False)
def _escenario(_estado, codigo, dias, replicas):
    return ejecutar_escenario(_estado, codigo, dias, replicas)


@st.cache_data(show_spinner=False)
def _comparacion(_estado, dias, replicas):
    return comparar_escenarios(_estado, None, dias, replicas)


@st.cache_data(show_spinner=False)
def _validacion(_catalogo, _movimientos, metodo, replicas):
    return retro_prueba(_catalogo, _movimientos, metodo_demanda=metodo, n_replicas=replicas)


# ==========================================================================
# PUNTO DE ENTRADA
# ==========================================================================
def ejecutar_simulacion_interfaz():
    with st.sidebar:
        st.markdown("### ⚙️ Fuente de datos")
        st.caption("Ventas reales de **Corporación Favorita**, tienda 44 "
                   "(Quito). El costo unitario y el lead time no vienen en el "
                   "dataset: siguen como supuesto para que el motor corra.")
        semilla = st.number_input("Semilla de los tiempos de entrega",
                                  value=EJECUCION["semilla_base"], step=1)
        dias_hist = st.slider("Días de histórico", 180, 720, EJECUCION["dias_historico"], 60)
        metodo = st.selectbox(
            "Estimador de demanda", ["em", "exclusion", "ingenuo"], index=0,
            format_func=lambda m: {"em": "EM (corrige censura)",
                                   "exclusion": "Excluye días con ruptura",
                                   "ingenuo": "Promedio de ventas (sesgado)"}[m],
            help="El promedio de ventas subestima la demanda porque los días con "
                 "ruptura no registran al cliente que se fue.")
        st.markdown("---")
        dias_sim = st.slider("Días de simulación", 30, 180, EJECUCION["dias_simulacion"], 30)
        replicas = st.slider("Réplicas", 5, 40, 12, 5,
                             help="El motor es estocástico: con una sola réplica no hay "
                                  "intervalos de confianza.")

    with st.spinner("Construyendo el estado del gemelo desde las ventas de Favorita..."):
        completo, catalogo, movimientos, verdad, estado, panel = _mundo(
            int(semilla), dias_hist, metodo)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKU en alcance", f"{len(estado)}", help="Clase A del análisis ABC")
    c2.metric("Días de histórico", f"{dias_hist}")
    c3.metric("Reposición mensual", f"{MONEDA} {gasto_reposicion_mensual(estado):,.0f}")
    c4.metric("Estimador", metodo.upper())

    pestanas = st.tabs([
        "📦 Alcance y datos", "📈 Prophet", "📋 CRISP-DM", "▶️ Corrida", "📊 Escenarios",
        "✅ Fidelidad", "🔬 Por qué este SKU", "🌐 Qué mueve el costo",
        "🔁 Recomendaciones", "🧭 Copiloto",
    ])
    with pestanas[0]:
        _tab_datos(completo, catalogo, movimientos, verdad)
    with pestanas[1]:
        _tab_prophet(panel)
    with pestanas[2]:
        from .vista_crisp import mostrar as mostrar_crisp
        mostrar_crisp(panel)
    with pestanas[3]:
        _tab_corrida(estado, dias_sim, replicas)
    with pestanas[4]:
        _tab_escenarios(estado, dias_sim, replicas)
    with pestanas[5]:
        _tab_fidelidad(catalogo, movimientos, metodo)
    with pestanas[6]:
        _tab_shapley(estado)
    with pestanas[7]:
        _tab_global(estado, dias_sim)
    with pestanas[8]:
        _tab_recomendaciones(estado)
    with pestanas[9]:
        _tab_copiloto(estado, dias_sim, replicas)


# ==========================================================================
# 1b. PROPHET — pronóstico y evaluación sobre la serie real
# ==========================================================================
@st.cache_data(show_spinner=False)
def _evaluacion_prophet(_panel):
    from .prophet_demanda import CACHE, disponible, evaluar_panel

    if not CACHE.exists():
        listo, motivo = disponible()
        if not listo:
            return None, None, motivo
    detalle, metricas = evaluar_panel(_panel)
    return detalle, metricas, None


def _tab_prophet(panel):
    """Gráficos del dataset real y de la evaluación Prophet, aparte del motor."""
    from .favorita import resumen_fuente

    fuente = resumen_fuente(panel)
    st.caption(f"Tienda {fuente['tienda']} · {fuente['ciudad']}, {fuente['provincia']} · "
               f"tipo {fuente['tipo']}, cluster {fuente['cluster']} · "
               f"{fuente['desde']} a {fuente['hasta']}. "
               f"Prophet se ajusta por ítem con la especificación de `models/prophet.pkl` "
               f"y se evalúa en los últimos 28 días, que no entran al ajuste.")

    diario = (panel.assign(u=panel["unit_sales"].clip(lower=0))
              .groupby("date", as_index=False)["u"].sum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diario["date"], y=diario["u"], mode="lines",
        line=dict(color=COLOR["stock"], width=1.4), name="Unidades"))
    fig.update_layout(template=PLANTILLA, height=380,
                      yaxis_title="Unidades vendidas (suma del alcance)",
                      margin=dict(t=30))
    narrativa.mostrar_figura(
        fig, "Ventas diarias reales del alcance",
        {"que_muestra": (f"La suma diaria de unidades de los {fuente['items']} ítems "
                         f"de clase A de la tienda {fuente['tienda']}, del "
                         f"{fuente['desde']} al {fuente['hasta']}."),
         "por_que": "Sale de train.csv. El día sin fila en Favorita entra como 0, "
                    "porque el dataset no registra la venta nula.",
         "limitacion": "Un cero no distingue «no hubo demanda» de «no hubo stock». "
                       "Tampoco es la demanda de SOAZ en Otuzco."},
        clave="fig_favorita_total")

    detalle, metricas, fallo = _evaluacion_prophet(panel)
    if fallo:
        st.info(f"**Prophet inactivo.** {fallo} Las ventas de arriba sí son las reales.")
        return

    utiles = metricas.dropna(subset=["WAPE"])
    wape = float(np.average(
        utiles["WAPE"], weights=utiles["media_historica"].clip(lower=0.01)))
    mase = float(np.nanmedian(metricas["MASE"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Ítems evaluados", f"{len(metricas)}")
    c2.metric("WAPE ponderado", f"{wape:.1f}%")
    c3.metric("MASE mediano", f"{mase:.2f}")

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=metricas["sku"], y=metricas["WAPE"],
        marker_color=COLOR["reorden"], name="WAPE"))
    fig_m.update_layout(template=PLANTILLA, height=420,
                        yaxis_title="WAPE del holdout de 28 días (%)",
                        margin=dict(t=30))
    peor = metricas.sort_values("WAPE", ascending=False).iloc[0]
    narrativa.mostrar_figura(
        fig_m, "Error del pronóstico Prophet, ítem por ítem",
        {"que_muestra": ("WAPE: la suma de errores absolutos dividida por la suma "
                         "de la demanda real del holdout, en porcentaje. "
                         f"El mediano de MASE es {mase:.2f} (1 sería igual de bueno "
                         "que repetir la semana anterior)."),
         "por_que": (f"El peor ítem de este corte es {peor['sku']} "
                     f"({peor['familia']}), con WAPE {peor['WAPE']:.1f}%."),
         "limitacion": "El holdout son 28 días corridos al final de la serie. "
                       "No es un backtest rodante ni una comparación contra los "
                       "otros modelos entrenados (LSTM, TFT, LightGBM)."},
        clave="fig_prophet_wape")

    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(
        x=metricas["sku"], y=metricas["media_historica"],
        name="Media histórica", marker_color=COLOR["neutro"]))
    fig_d.add_trace(go.Bar(
        x=metricas["sku"], y=metricas["media_pronostico"],
        name="Media del pronóstico", marker_color=COLOR["ok"]))
    fig_d.update_layout(template=PLANTILLA, height=420, barmode="group",
                        yaxis_title="Unidades por día", margin=dict(t=30))
    narrativa.mostrar_figura(
        fig_d, "Lo que Prophet cambiaría en la demanda media",
        {"que_muestra": "La media del historial de ajuste contra la media de los "
                        "28 días pronosticados. Esa media es la que entra al stock "
                        "de seguridad si se planifica con el pronóstico.",
         "por_que": "Prophet descompone nivel, semana y año. Si el tramo final "
                    "está en temporada alta, la media pronosticada queda arriba "
                    "de la media larga.",
         "limitacion": "Sustituir la media del gemelo por este número es una "
                       "decisión aparte: el motor sigue estimando desde las "
                       "ventas reconstruidas, no desde yhat."},
        clave="fig_prophet_medias")

    opciones = {f"IT-{int(i)} · {fam}": int(i)
                for i, fam in zip(metricas["item_nbr"], metricas["familia"])}
    etiqueta = st.selectbox("Ítem", list(opciones), key="prophet_item")
    item = opciones[etiqueta]
    serie = panel[panel["item_nbr"] == item].sort_values("date")
    hold = detalle[detalle["item_nbr"] == item]
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=serie["date"], y=serie["unit_sales"].clip(lower=0),
        mode="lines", line=dict(color=COLOR["neutro"], width=1),
        name="Venta real"))
    fig_s.add_trace(go.Scatter(
        x=hold["date"], y=hold["pronostico"],
        mode="lines", line=dict(color=COLOR["riesgo"], width=2.4),
        name="Prophet (28 días)"))
    fig_s.update_layout(template=PLANTILLA, height=420,
                        yaxis_title="Unidades", margin=dict(t=30))
    fila = metricas[metricas["item_nbr"] == item].iloc[0]
    narrativa.mostrar_figura(
        fig_s, f"Serie real y pronóstico de {etiqueta}",
        {"que_muestra": (f"Holdout de 28 días. MAE {fila['MAE']:.2f} unidades, "
                         f"RMSE {fila['RMSE']:.2f}, WAPE {fila['WAPE']:.1f}%, "
                         f"MASE {fila['MASE']:.2f}."),
         "por_que": "El modelo se ajustó solo con los días anteriores al corte, "
                    "con estacionalidad semanal de orden 5 y anual.",
         "limitacion": "Una serie corta o con muchos ceros hace que el MASE "
                       "se vuelva inestable. El pronóstico no ve promociones "
                       "futuras ni el precio del petróleo."},
        clave="fig_prophet_serie")

    promo = serie["onpromotion"].astype(bool)
    unidades = serie["unit_sales"].clip(lower=0)
    medias = [
        float(unidades[promo].mean()) if promo.any() else 0.0,
        float(unidades[~promo].mean()) if (~promo).any() else 0.0,
    ]
    fig_p = go.Figure(go.Bar(
        x=["En promoción", "Sin promoción"], y=medias,
        marker_color=[COLOR["reorden"], COLOR["stock"]]))
    fig_p.update_layout(template=PLANTILLA, height=360,
                        yaxis_title="Unidades medias por día", margin=dict(t=30))
    narrativa.mostrar_figura(
        fig_p, f"Promoción y venta de {etiqueta}",
        {"que_muestra": (f"Media de unidades en días con onpromotion "
                         f"({medias[0]:.2f}) y sin ella ({medias[1]:.2f}). "
                         f"La tasa de promoción del ítem es "
                         f"{float(promo.mean()) * 100:.1f}%."),
         "por_que": "onpromotion es un campo del dataset, no un resultado del motor.",
         "limitacion": "No es un efecto causal de la promoción: los ítems se "
                       "promocionan justo cuando ya se espera que vendan más."},
        clave="fig_prophet_promo")

    narrativa.mostrar_tabla(
        metricas.round({"MAE": 2, "RMSE": 2, "WAPE": 1, "MASE": 2,
                        "media_historica": 2, "media_pronostico": 2}),
        "Métricas de evaluación Prophet",
        {"que_muestra": "MAE y RMSE en unidades por día. WAPE en porcentaje. "
                        "MASE contra el pronóstico ingenuo de hace 7 días.",
         "por_que": "Es el mismo protocolo de 28 días del proyecto de proyección "
                    "de demanda, aplicado solo a Prophet y a estos ítems.",
         "limitacion": "No incluye Friedman ni Diebold-Mariano: esas pruebas "
                       "comparan varios modelos, y aquí el modelo es uno."})


# ==========================================================================
# 1. ALCANCE Y DATOS
# ==========================================================================
def _tab_datos(completo, catalogo, movimientos, verdad):
    st.caption("Las unidades son las de `train.csv` (tienda 44, Quito). "
               "El valor en soles usa un costo unitario supuesto: Favorita no "
               "publica precios. Los días sin registro entran como venta 0.")
    st.markdown("##### Delimitación del alcance")
    filas = resumen_abc(completo)
    narrativa.mostrar_tabla(pd.DataFrame(filas), "Clasificación ABC del catálogo",
                            narrativa.interpretar_abc(filas, len(completo)), expandido=True)

    resumen = resumen_historico(movimientos, verdad)
    st.markdown("##### Histórico de movimientos")
    st.caption(f"**Práctica actual de compra:** {PRACTICA_ACTUAL['descripcion']}")
    cols = st.columns(4)
    for i, (k, v) in enumerate(list(resumen.items())[:8]):
        cols[i % 4].metric(k, f"{v:,}" if isinstance(v, (int, float)) else v)

    st.markdown("---")
    st.markdown("##### El problema que decide la calidad del gemelo: demanda censurada")
    comp = comparar_estimadores(catalogo, movimientos, verdad)
    r = comp["resumen"]

    fig = go.Figure()
    nombres = {"ingenuo": "Promedio de ventas", "exclusion": "Excluye rupturas",
               "em": "EM (corrige censura)"}
    for m, color in [("ingenuo", COLOR["riesgo"]), ("exclusion", COLOR["reorden"]),
                     ("em", COLOR["ok"])]:
        fig.add_trace(go.Bar(x=[nombres[m]], y=[r[m]["sesgo_medio"]],
                             marker_color=color, name=nombres[m], showlegend=False,
                             text=[f"{r[m]['sesgo_medio']:+.1f}%"], textposition="outside",
                             hovertemplate=f"{nombres[m]}<br>sesgo medio %{{y:+.2f}}%<extra></extra>"))
    fig.add_hline(y=0, line_color=COLOR["neutro"])
    fig.update_layout(template=PLANTILLA, height=360,
                      yaxis_title="Sesgo frente a la demanda verdadera (%)", margin=dict(t=40))
    narrativa.mostrar_figura(fig, "Cuánto subestima la demanda cada estimador",
                             narrativa.interpretar_censura(comp, resumen), expandido=True,
                             clave="fig_censura")

    df = pd.DataFrame(comp["tabla"]).sort_values("% días censurados", ascending=False).head(12)
    narrativa.mostrar_tabla(df, "SKU más afectados por la censura",
                            narrativa.interpretar_censura(comp, resumen))

    st.markdown("---")
    conteo = {"TEORIA": 0, "ESTIMADO": 0, "SUPUESTO": 0}
    import inspect
    from . import parametros as P
    for linea in inspect.getsource(P).splitlines():
        for k in conteo:
            if f"# {k}" in linea:
                conteo[k] += 1
    fig2 = go.Figure(go.Bar(x=list(conteo.keys()), y=list(conteo.values()),
                            marker_color=[COLOR["ok"], COLOR["stock"], COLOR["reorden"]],
                            text=list(conteo.values()), textposition="outside"))
    fig2.update_layout(template=PLANTILLA, height=320, yaxis_title="Parámetros", margin=dict(t=40))
    narrativa.mostrar_figura(fig2, "De dónde sale cada parámetro del modelo",
                             narrativa.interpretar_procedencia(conteo), clave="fig_proc")


# ==========================================================================
# 2. CORRIDA
# ==========================================================================
def _tab_corrida(estado, dias, replicas):
    codigo = st.selectbox("🎯 Escenario", list(ESCENARIOS.keys()),
                          format_func=lambda c: f"{c} — {ESCENARIOS[c]['nombre']}",
                          key="esc_corrida")
    cfg = ESCENARIOS[codigo]
    st.caption(f"**Pregunta del negocio:** {cfg['pregunta']}  \n"
               f"**Mecanismo:** {cfg['mecanismo']}")

    if not st.button("▶️ Ejecutar", type="primary", use_container_width=True, key="btn_corrida"):
        st.info("Configure el escenario y ejecute la simulación.")
        return

    with st.spinner(f"Ejecutando {replicas} réplicas de {dias} días..."):
        res = _escenario(estado, codigo, dias, replicas)
    st.session_state["ultima_corrida"] = res

    c = st.columns(5)
    c[0].metric("Fill rate", f"{res['fill_rate']:.2f}%",
                help=f"IC 95%: [{res['fill_rate_ic'][0]:.2f}, {res['fill_rate_ic'][1]:.2f}]")
    c[1].metric("Costo total", f"{MONEDA} {res['costo_total']:,.0f}",
                help=f"IC 95%: [{res['costo_total_ic'][0]:,.0f}, {res['costo_total_ic'][1]:,.0f}]")
    c[2].metric("Capital inmovilizado", f"{MONEDA} {res['capital_inmovilizado']:,.0f}")
    c[3].metric("Órdenes", f"{res['ordenes']:.0f}")
    c[4].metric("SKU con ruptura", f"{res['sku_con_ruptura']:.0f} / {res['n_sku']}")

    # --- Composición del costo ---
    partes = [("Ruptura", res["costo_ruptura"]), ("Mantener", res["costo_mantener"]),
              ("Merma", res["costo_merma"]), ("Ordenar", res["costo_ordenar"])]
    fig = go.Figure(go.Bar(x=[p[1] for p in partes], y=[p[0] for p in partes],
                           orientation="h",
                           marker_color=[COLOR["riesgo"], COLOR["stock"],
                                         COLOR["reorden"], COLOR["neutro"]],
                           text=[f"{MONEDA} {p[1]:,.0f}" for p in partes],
                           textposition="auto",
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    fig.update_layout(template=PLANTILLA, height=330,
                      xaxis_title=f"Costo en {dias} días ({MONEDA})", margin=dict(t=40))
    narrativa.mostrar_figura(fig, "De qué está hecho el costo",
                             narrativa.interpretar_corrida(res), expandido=True,
                             clave=f"fig_costo_{codigo}")

    # --- Diente de sierra de un SKU ---
    muestra = res["_replica_muestra"]["por_sku"]
    opciones = {s["sku"]: s for s in muestra}
    sku = st.selectbox("SKU a inspeccionar", list(opciones.keys()), key="sku_sierra")
    serie = opciones[sku]["serie"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[p["dia"] for p in serie], y=[p["stock"] for p in serie],
                              mode="lines", name="Stock a la mano",
                              line=dict(color=COLOR["stock"], width=2),
                              hovertemplate="Día %{x}<br>%{y:,.0f} u<extra></extra>"))
    fig2.add_trace(go.Scatter(x=[p["dia"] for p in serie], y=[p["posicion"] for p in serie],
                              mode="lines", name="Posición de inventario",
                              line=dict(color=COLOR["neutro"], width=1, dash="dot")))
    if serie[0]["punto_reorden"]:
        fig2.add_hline(y=serie[0]["punto_reorden"], line_dash="dash", line_color=COLOR["reorden"],
                       annotation_text="Punto de reorden")
    fig2.add_hline(y=serie[0]["stock_seguridad"], line_dash="dot", line_color=COLOR["seguridad"],
                   annotation_text="Stock de seguridad")
    fig2.add_hline(y=0, line_color=COLOR["riesgo"])
    fig2.update_layout(template=PLANTILLA, height=380, xaxis_title="Día",
                       yaxis_title="Unidades",
                       legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=45))
    narrativa.mostrar_figura(fig2, f"Diente de sierra de {sku}",
                             narrativa.interpretar_diente_sierra(sku, serie),
                             clave=f"fig_sierra_{sku}")

    narrativa.mostrar_tabla(pd.DataFrame(res["por_sku"]), "Resultado por SKU",
                            narrativa.interpretar_corrida(res))


# ==========================================================================
# 3. ESCENARIOS
# ==========================================================================
def _tab_escenarios(estado, dias, replicas):
    if not st.button("📊 Comparar los 6 escenarios", type="primary",
                     use_container_width=True, key="btn_esc"):
        st.info("Se ejecutan los 6 escenarios con el mismo número de réplicas.")
        return
    with st.spinner(f"Ejecutando 6 escenarios × {replicas} réplicas..."):
        comp = _comparacion(estado, dias, replicas)

    base = comp["resultados"]["Base"]
    fig = go.Figure()
    for f in comp["tabla"]:
        r = comp["resultados"][f["Código"]]
        dist = f["¿Distinguible del Base?"] == "Sí"
        fig.add_trace(go.Bar(
            x=[f"{f['Código']} — {f['Escenario'][:24]}"], y=[r["costo_total"]],
            error_y=dict(type="data", symmetric=False,
                         array=[r["costo_total_ic"][1] - r["costo_total"]],
                         arrayminus=[r["costo_total"] - r["costo_total_ic"][0]],
                         color=COLOR["neutro"], thickness=1.5, width=6),
            marker_color=(COLOR["neutro"] if f["Código"] == "Base"
                          else COLOR["riesgo"] if dist and r["costo_total"] > base["costo_total"]
                          else COLOR["ok"] if dist else COLOR["reorden"]),
            showlegend=False,
            hovertemplate=f"{f['Escenario']}<br>%{{y:,.0f}}<br>"
                          f"{'Distinguible del Base' if dist else 'Dentro del ruido'}<extra></extra>"))
    fig.add_hline(y=base["costo_total"], line_dash="dot", line_color=COLOR["neutro"],
                  annotation_text="Política actual")
    fig.update_layout(template=PLANTILLA, height=420,
                      yaxis_title=f"Costo total en {dias} días ({MONEDA})", margin=dict(t=40))
    narrativa.mostrar_figura(fig, "Costo de cada escenario, con su incertidumbre",
                             narrativa.interpretar_comparacion(comp), expandido=True,
                             clave="fig_esc")
    narrativa.mostrar_tabla(pd.DataFrame(comp["tabla"]), "Comparación de escenarios",
                            narrativa.interpretar_comparacion(comp))

    st.markdown("---")
    with st.spinner("Trazando la curva servicio-costo..."):
        curva = comparar_politicas(estado, dias=dias, n_replicas=max(6, replicas // 2))
    dfc = pd.DataFrame(curva)
    fig2 = px.line(dfc, x="Fill rate logrado (%)", y="Costo total (S/.)",
                   color="Política", markers=True, text="Nivel de servicio objetivo",
                   color_discrete_sequence=[COLOR["stock"], COLOR["reorden"]])
    fig2.update_traces(textposition="top center")
    fig2.update_layout(template=PLANTILLA, height=420,
                       yaxis_title=f"Costo total ({MONEDA})", margin=dict(t=40))
    narrativa.mostrar_figura(fig2, "Cuánto cuesta prometer más servicio",
                             narrativa.interpretar_servicio_costo(curva), clave="fig_curva")
    narrativa.mostrar_tabla(dfc.drop(columns=["nivel", "costo_lo", "costo_hi"]),
                            "Servicio contra costo",
                            narrativa.interpretar_servicio_costo(curva))


# ==========================================================================
# 4. FIDELIDAD
# ==========================================================================
def _tab_fidelidad(catalogo, movimientos, metodo):
    st.markdown(
        f"El gemelo se congela en el día **{FIDELIDAD['dias_ajuste']}**, corre "
        f"**{FIDELIDAD['dias_prueba']} días** hacia adelante con la política que SOAZ "
        f"realmente aplicó, y se compara contra el registro que no vio. "
        f"Los umbrales se fijaron **antes** de correr la prueba.")
    replicas = st.slider("Réplicas de validación", 8, 40, 20, 4, key="rep_val")

    if not st.button("✅ Ejecutar la retro-prueba", type="primary",
                     use_container_width=True, key="btn_val"):
        return
    with st.spinner("Validando fidelidad..."):
        val = _validacion(catalogo, movimientos, metodo, replicas)
    r = val["resumen"]

    if r["veredicto"]:
        st.success(f"**PASA** — {r['n_criterios_cumplidos']} de {r['n_criterios']} criterios cumplidos.")
    else:
        st.error(f"**NO PASA** — {r['n_criterios_cumplidos']} de {r['n_criterios']} criterios. "
                 f"Se reporta igual, con el diagnóstico: un gemelo que falla y explica por qué "
                 f"es un resultado legítimo; uno que nunca se validó, no.")

    dfc = pd.DataFrame([{
        "Criterio": c["Criterio"], "Métrica": c["Métrica"],
        "Obtenido": c["Obtenido"], "Umbral": c["Umbral"], "Unidad": c["unidad"],
        "Cumple": "✅" if c["Cumple"] else "❌",
        "Por qué esta métrica": c.get("Por qué esta métrica", ""),
    } for c in r["criterios"]])
    narrativa.mostrar_tabla(dfc, "Criterios de aceptación",
                            narrativa.interpretar_fidelidad(val), expandido=True)

    # --- Trayectoria real vs banda predicha ---
    opciones = {f["SKU"]: f for f in val["por_sku"]}
    sku = st.selectbox("SKU a inspeccionar", list(opciones.keys()), key="sku_val")
    f = opciones[sku]
    x = list(range(1, len(f["_real"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=f["_banda_alta"] + f["_banda_baja"][::-1],
                             fill="toself", fillcolor=COLOR["banda"],
                             line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                             name="Banda 95% del gemelo"))
    fig.add_trace(go.Scatter(x=x, y=f["_media_sim"], mode="lines", name="Media del gemelo",
                             line=dict(color=COLOR["stock"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=f["_real"], mode="lines", name="Stock real registrado",
                             line=dict(color=COLOR["riesgo"], width=2.5)))
    fig.update_layout(template=PLANTILLA, height=400,
                      xaxis_title=f"Día del período de prueba (desde el {r['dia_corte']})",
                      yaxis_title="Unidades en stock",
                      legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=45))
    narrativa.mostrar_figura(fig, f"Lo que el gemelo predijo vs. lo que pasó — {sku}",
                             narrativa.interpretar_trayectoria_validacion(f),
                             clave=f"fig_val_{sku}")

    narrativa.mostrar_tabla(
        pd.DataFrame([{k: v for k, v in f.items() if not k.startswith("_")}
                      for f in val["por_sku"]]),
        "Fidelidad por SKU", narrativa.interpretar_fidelidad(val))

    st.markdown("---")
    st.markdown("##### ¿La corrección por censura mejora la fidelidad medida?")
    if st.button("Comparar estimadores en la retro-prueba", key="btn_val_met"):
        with st.spinner("Corriendo la retro-prueba con cada estimador..."):
            filas = comparar_fidelidad_por_metodo(catalogo, movimientos, n_replicas=10)
        interp = {
            "que_muestra": ("La misma retro-prueba corrida con cada estimador de demanda. "
                            "Si la corrección por censura no mejorara la fidelidad medida, "
                            "usarla sería una preferencia metodológica y no una decisión "
                            "justificada."),
            "por_que": ("El estimador afecta al gemelo por dos vías: la demanda que genera "
                        "y la regla de compra que reproduce. En la retro-prueba la regla "
                        "usa la demanda percibida por el negocio, así que la diferencia "
                        "que se ve aquí viene sobre todo de la primera vía."),
            "limitacion": ("Una sola partición del histórico. Un corte en otra fecha, con "
                           "otra estacionalidad, puede ordenar distinto a los estimadores."),
        }
        narrativa.mostrar_tabla(pd.DataFrame(filas), "Fidelidad según el estimador", interp)


# ==========================================================================
# 5. SHAPLEY POR SKU
# ==========================================================================
def _tab_shapley(estado):
    opciones = {e["sku"]: e for e in estado}
    sku = st.selectbox("🔬 SKU a explicar", list(opciones.keys()), key="sku_shap")
    e = opciones[sku]

    with st.spinner("Calculando Shapley exacto y midiendo el sustituto..."):
        ex = explicar_sku(e, estado)
        fid = fidelidad_sustituto(estado, n_replicas=6)

    c = st.columns(4)
    c[0].metric("Costo anual del SKU", f"{MONEDA} {ex['valor_sku']:,.0f}",
                delta=f"{ex['valor_sku'] - ex['valor_base']:+,.0f} vs. SKU promedio")
    c[1].metric("Eficiencia de Shapley", "exacta" if ex["eficiencia_ok"] else "revisar",
                help=f"Error {ex['error_eficiencia']:.2e}. Las contribuciones deben sumar "
                     f"exactamente la diferencia explicada.")
    c[2].metric("Coaliciones", f"{ex['n_coaliciones']}",
                help="2^9 = 512. Enumeración completa, sin muestreo.")
    c[3].metric("Fidelidad del sustituto", f"{fid['error_absoluto_medio']:.0f}% error",
                help=f"Correlación con la simulación: {fid['correlacion']:.2f}. "
                     f"{'Apto' if fid['apto'] else 'NO apto: la explicación es dudosa'}")

    tabla = sorted(ex["tabla"], key=lambda f: f["Contribución (S/.)"])
    fig = go.Figure(go.Bar(
        x=[f["Contribución (S/.)"] for f in tabla], y=[f["Factor"] for f in tabla],
        orientation="h",
        marker_color=[COLOR["riesgo"] if f["Contribución (S/.)"] > 0 else COLOR["ok"]
                      for f in tabla],
        hovertemplate="%{y}<br>%{x:+,.0f} " + MONEDA + "<extra></extra>"))
    fig.add_vline(x=0, line_color=COLOR["neutro"])
    fig.update_layout(template=PLANTILLA, height=430,
                      xaxis_title=f"Contribución al costo anual ({MONEDA})", margin=dict(t=40, l=10))
    narrativa.mostrar_figura(fig, f"Por qué {sku} cuesta lo que cuesta",
                             narrativa.interpretar_shapley(ex, fid), expandido=True,
                             clave=f"fig_shap_{sku}")
    narrativa.mostrar_tabla(pd.DataFrame(ex["tabla"]).drop(columns=["clave"]),
                            "Contribuciones exactas por factor",
                            narrativa.interpretar_shapley(ex, fid))

    st.markdown("---")
    interp = {
        "que_muestra": (f"Costo que predice el modelo analítico frente al que produce la "
                        f"simulación, SKU por SKU. Error absoluto medio "
                        f"{fid['error_absoluto_medio']:.1f}%, correlación "
                        f"{fid['correlacion']:.3f}."),
        "por_que": ("El modelo analítico es un sustituto de la simulación, no la "
                    "simulación. Explicar el sustituto y presentarlo como si se "
                    "explicara el motor sería un fraude silencioso; por eso se mide. "
                    "El término de merma se añadió justamente después de esta medición: "
                    "sin él, el sustituto erraba más del 90% en los perecibles, donde la "
                    "merma es el costo dominante."),
        "limitacion": ("La fórmula supone estado estacionario y demanda normal. En SKU con "
                       "poca rotación o vida útil muy corta el supuesto se debilita, y ahí "
                       "la explicación de Shapley debe leerse como orientativa."),
    }
    dff = pd.DataFrame(fid["tabla"]).head(12)
    narrativa.mostrar_tabla(dff, "Fidelidad del modelo que se explica", interp)


# ==========================================================================
# 6. QUÉ MUEVE EL COSTO
# ==========================================================================
def _tab_global(estado, dias):
    tn = tornado_sensibilidad(estado)
    filas = list(reversed(tn["filas"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f["efecto_bajo"] for f in filas], y=[f["Factor"] for f in filas],
                         orientation="h", marker_color=COLOR["ok"], name="Percentil 10",
                         hovertemplate="%{y}<br>p10: %{x:+,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=[f["efecto_alto"] for f in filas], y=[f["Factor"] for f in filas],
                         orientation="h", marker_color=COLOR["riesgo"], name="Percentil 90",
                         hovertemplate="%{y}<br>p90: %{x:+,.0f}<extra></extra>"))
    fig.add_vline(x=0, line_color=COLOR["neutro"])
    fig.update_layout(template=PLANTILLA, barmode="overlay", height=460,
                      xaxis_title=f"Cambio del costo anual respecto del SKU promedio ({MONEDA})",
                      legend=dict(orientation="h", y=1.10, x=0), margin=dict(t=45, l=10))
    narrativa.mostrar_figura(fig, "Qué factores mueven más el costo",
                             narrativa.interpretar_tornado(tn), expandido=True, clave="fig_torn")
    narrativa.mostrar_tabla(pd.DataFrame(tn["filas"]).drop(
        columns=["clave", "efecto_bajo", "efecto_alto"]),
        "Sensibilidad por factor", narrativa.interpretar_tornado(tn))

    st.markdown("---")
    st.markdown("##### De qué está hecho un escenario")
    con_palancas = [c for c, v in ESCENARIOS.items() if v["palancas"]]
    codigo = st.selectbox("Escenario a descomponer", con_palancas,
                          format_func=lambda c: f"{c} — {ESCENARIOS[c]['nombre']}",
                          index=1 if len(con_palancas) > 1 else 0, key="sel_desc")
    if st.button("🧩 Descomponer", type="primary", use_container_width=True, key="btn_desc"):
        with st.spinner("Aislando palancas..."):
            d = descomponer_escenario(estado, codigo, dias=dias, n_replicas=6)
        if not d:
            st.warning("Ese escenario no tiene palancas que descomponer.")
            return
        c = st.columns(3)
        c[0].metric("Efecto del escenario completo", f"{MONEDA} {d['efecto_conjunto']:+,.0f}")
        c[1].metric("Suma de las palancas", f"{MONEDA} {d['suma_individual']:+,.0f}")
        c[2].metric("Interacción", f"{MONEDA} {d['interaccion']:+,.0f}", help=d["tipo_interaccion"])

        fig2 = go.Figure(go.Bar(x=[f["Palanca"] for f in d["filas"]],
                                y=[f["Efecto sobre el costo (S/.)"] for f in d["filas"]],
                                marker_color=COLOR["riesgo"],
                                hovertemplate="%{x}<br>%{y:+,.0f}<extra></extra>"))
        fig2.add_hline(y=d["efecto_conjunto"], line_dash="dash", line_color=COLOR["stock"],
                       annotation_text=f"Escenario completo: {d['efecto_conjunto']:+,.0f}")
        fig2.update_layout(template=PLANTILLA, height=400,
                           yaxis_title=f"Efecto sobre el costo ({MONEDA})", margin=dict(t=40))
        narrativa.mostrar_figura(fig2, f"Palancas de {d['nombre']}",
                                 narrativa.interpretar_descomposicion(d), expandido=True,
                                 clave=f"fig_desc_{codigo}")
        narrativa.mostrar_tabla(pd.DataFrame(d["filas"]).drop(columns=["clave"]),
                                "Aporte de cada palanca",
                                narrativa.interpretar_descomposicion(d))


# ==========================================================================
# 7. RECOMENDACIONES — el lazo de retorno
# ==========================================================================
def _tab_recomendaciones(estado):
    st.markdown(
        "Lo que separa un gemelo de una sombra digital: el resultado no se queda en la "
        "pantalla, vuelve al proceso como una orden concreta que el encargado aprueba o "
        "rechaza. **El humano se queda en el lazo**, y su tasa de aceptación es lo que "
        "hay que medir para saber si el gemelo sirve.")
    with st.spinner("Calculando recomendaciones..."):
        filas = recomendaciones_reposicion(estado)

    urgentes = [f for f in filas if f["Urgencia"] == "Inmediata"]
    c = st.columns(3)
    c[0].metric("Órdenes inmediatas", len(urgentes))
    c[1].metric("Órdenes esta semana", sum(1 for f in filas if f["Urgencia"] == "Esta semana"))
    c[2].metric("Valor total sugerido",
                f"{MONEDA} {sum(f['Valor de la orden (S/.)'] for f in filas):,.0f}")

    df = pd.DataFrame(filas)
    narrativa.mostrar_tabla(df, "Bandeja de reposición sugerida",
                            narrativa.interpretar_recomendaciones(filas), expandido=True)

    if urgentes:
        st.markdown("##### Por qué cada orden urgente")
        for f in urgentes[:8]:
            with st.expander(f"{f['SKU']} — {f['Cantidad sugerida (u)']} u "
                             f"({MONEDA} {f['Valor de la orden (S/.)']:,.0f})"):
                st.markdown(f["Por qué"])
                st.caption(f"Proveedor: {f['Proveedor']} · Familia: {f['Familia']} · "
                           f"Stock de seguridad: {f['Stock de seguridad (u)']} u")


# ==========================================================================
# 8. COPILOTO DE ESCENARIOS
# ==========================================================================
def _tab_copiloto(estado, dias, replicas):
    st.markdown(
        "Escribe una pregunta del negocio. El copiloto la traduce a palancas del motor, "
        "**te las muestra para que las apruebes**, y recién entonces simula. Las cifras "
        "las produce el motor de siempre: el modelo de lenguaje traduce y redacta, "
        "nunca calcula.")

    # Import diferido a propósito: las otras siete pestañas no deben pagar el
    # costo de importar langchain, ni fallar si no está instalado.
    from modulo_copiloto import proveedor

    listo, motivo = proveedor.disponible()
    if not listo:
        st.info(f"🧭 **Copiloto inactivo.** {motivo}")
        st.caption("Ninguna de las otras siete pestañas depende de este módulo.")
        return

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from modulo_copiloto import guardia, langflow_cliente
    from modulo_copiloto.esquemas import EscenarioPropuesto, catalogo_palancas
    from modulo_copiloto.grafo import construir_grafo, estado_inicial, texto_final

    traductor = langflow_cliente.traductor_si_configurado()
    if traductor is not None:
        st.caption("La traducción la hace el flujo **Traductor de escenarios SOAZ** "
                   "en Langflow. La simulación y el cerco numérico siguen en este motor; "
                   "la redacción del párrafo final también.")

    ss = st.session_state
    if "copiloto_memoria" not in ss:
        ss.copiloto_memoria = InMemorySaver()
        ss.copiloto_modelo = proveedor.modelo()
        ss.copiloto_n = 0
        ss.copiloto_estado = None
        ss.copiloto_config = None

    # El grafo se recompila en cada rerun —cuesta milisegundos— pero el
    # checkpointer y el modelo viven en la sesión. Eso es lo que permite que el
    # interrupt() sobreviva a que Streamlit reejecute el script de arriba abajo
    # cada vez que se toca un botón.
    app = construir_grafo(estado, dias, replicas,
                          modelo_chat=ss.copiloto_modelo,
                          checkpointer=ss.copiloto_memoria,
                          traductor=traductor)

    EJEMPLOS = [
        "¿Qué pasa si la demanda sube 30% en campaña navideña?",
        "¿Y si el proveedor se demora el triple y se vuelve impredecible?",
        "¿Cuánto cuesta prometer 99% de nivel de servicio?",
        "¿Y si solo puedo financiar el 70% de lo que la reposición pide?",
    ]
    st.caption("Estos son ejemplos para arrancar, no un menú: las ocho palancas del "
               "motor se combinan libremente. Si tu pregunta no se puede simular, el "
               "copiloto lo dice en vez de inventar un escenario.")
    columnas = st.columns(len(EJEMPLOS))
    for i, (columna, ejemplo) in enumerate(zip(columnas, EJEMPLOS)):
        if columna.button(ejemplo, key=f"copiloto_ej_{i}", use_container_width=True):
            ss.copiloto_pregunta = ejemplo
            st.rerun()

    st.text_area("Tu pregunta", height=80, key="copiloto_pregunta",
                 placeholder="¿Qué pasa si...?")
    pregunta = (ss.get("copiloto_pregunta") or "").strip()

    if st.button("🧭 Traducir a escenario", type="primary", use_container_width=True,
                 disabled=not pregunta, key="copiloto_btn_traducir"):
        ss.copiloto_n += 1
        ss.copiloto_config = {"configurable": {"thread_id": f"copiloto-{ss.copiloto_n}"}}
        with st.spinner("Traduciendo la pregunta a palancas del motor..."):
            ss.copiloto_estado = app.invoke(estado_inicial(pregunta),
                                            config=ss.copiloto_config)
        st.rerun()

    est = ss.copiloto_estado
    if est is None:
        return

    st.markdown("---")

    # --- El grafo se detuvo a pedir permiso -------------------------------
    interrupciones = est.get("__interrupt__")
    if interrupciones:
        carga = interrupciones[0].value
        st.markdown(f"#### Propuesta: {carga['nombre']}")
        st.caption(f"Pregunta interpretada: «{carga['pregunta']}»")
        # Esta tabla NO pasa por narrativa.mostrar_tabla: no es una tabla de
        # resultados sino un formulario de confirmación. La regla de la interfaz
        # protege las salidas del motor, y aquí todavía no hay ninguna.
        st.dataframe(pd.DataFrame(carga["filas"]), use_container_width=True,
                     hide_index=True)
        st.markdown(f"**Mecanismo declarado.** {carga['mecanismo']}")
        st.warning("Revisa las palancas antes de simular: el escenario va a responder lo "
                   "que dicen estas filas, no lo que dice la frase que escribiste.")
        izquierda, derecha = st.columns(2)
        if izquierda.button("▶️ Simular este escenario", type="primary",
                            use_container_width=True, key="copiloto_btn_simular"):
            with st.spinner(f"Corriendo Base y escenario × {replicas} réplicas..."):
                ss.copiloto_estado = app.invoke(Command(resume=True),
                                                config=ss.copiloto_config)
            st.rerun()
        if derecha.button("✖️ Descartar", use_container_width=True,
                          key="copiloto_btn_descartar"):
            ss.copiloto_estado = app.invoke(Command(resume=False),
                                            config=ss.copiloto_config)
            st.rerun()
        return

    # --- Caminos que no llegan a resultado --------------------------------
    if est.get("fallo"):
        st.error(f"**No se pudo completar.** {est['fallo']}")
        return

    if est.get("no_traducible"):
        # Rechazo legítimo, no un error. El copiloto prefiere decir «esto no se
        # puede simular» antes que fabricar un escenario que nadie pidió: un
        # escenario inventado hace dudar, con razón, también de las cifras.
        st.info(f"**Esta pregunta no se puede convertir en una simulación.**\n\n"
                f"{est['no_traducible']}")
        st.markdown("#### Lo que sí puedes preguntar")
        st.caption("El motor tiene ocho palancas. No hay preguntas predefinidas: "
                   "cualquier frase que mueva una o varias de estas sirve, y se pueden "
                   "combinar entre sí.")
        st.dataframe(pd.DataFrame(catalogo_palancas()), use_container_width=True,
                     hide_index=True)
        return

    if est.get("propuesta") is None:
        st.error(f"No se pudo construir un escenario válido después de "
                 f"{est.get('intentos_traduccion', 0)} intentos. El validador del motor "
                 f"rechazó cada propuesta:")
        for error in est.get("errores") or []:
            st.code(error)
        st.markdown("#### Las palancas disponibles")
        st.dataframe(pd.DataFrame(catalogo_palancas()), use_container_width=True,
                     hide_index=True)
        return

    if est.get("confirmado") is False:
        st.info("Escenario descartado. No se simuló nada.")
        return

    resultados = est.get("resultados")
    if not resultados:
        return

    # --- Resultado --------------------------------------------------------
    propuesta = EscenarioPropuesto(**est["propuesta"])
    interp = narrativa.interpretar_escenario_adhoc(resultados, propuesta.a_cfg(),
                                                   est.get("cerco"))
    base, prop = resultados["base"], resultados["propuesto"]
    distinguible = resultados["distinguible"]

    fig = go.Figure()
    for r, etiqueta in ((base, "Base — política actual"), (prop, prop["nombre"][:30])):
        es_base = r is base
        fig.add_trace(go.Bar(
            x=[etiqueta], y=[r["costo_total"]],
            error_y=dict(type="data", symmetric=False,
                         array=[r["costo_total_ic"][1] - r["costo_total"]],
                         arrayminus=[r["costo_total"] - r["costo_total_ic"][0]],
                         color=COLOR["neutro"], thickness=1.5, width=6),
            marker_color=(COLOR["neutro"] if es_base
                          else COLOR["riesgo"] if distinguible and r["costo_total"] > base["costo_total"]
                          else COLOR["ok"] if distinguible else COLOR["reorden"]),
            showlegend=False,
            hovertemplate=(f"{etiqueta}<br>%{{y:,.0f}}<br>"
                           f"{'Distinguible del Base' if distinguible else 'Dentro del ruido'}"
                           f"<extra></extra>")))
    fig.add_hline(y=base["costo_total"], line_dash="dot", line_color=COLOR["neutro"],
                  annotation_text="Política actual")
    fig.update_layout(template=PLANTILLA, height=420,
                      yaxis_title=f"Costo total en {dias} días ({MONEDA})",
                      margin=dict(t=40))
    narrativa.mostrar_figura(fig, "El escenario contra el Base, con su incertidumbre",
                             interp, expandido=True, clave="fig_copiloto")
    narrativa.mostrar_tabla(pd.DataFrame(resultados["tabla"]),
                            "Comparación contra la política actual", interp)

    # --- La envoltura del modelo, si el cerco la dejó pasar ---------------
    st.markdown("---")
    final = texto_final(est)
    cerco = final["cerco"]
    st.markdown("#### Lectura de negocio")
    if final["envoltura"]:
        st.markdown(final["envoltura"])
        st.caption(f"✅ {guardia.resumen(cerco)}")
    else:
        st.warning(
            "El texto generado fue **descartado por el cerco numérico**. Lo que vale es "
            "la interpretación determinista de arriba, que sale de los números de la "
            "corrida.")
        if final["envoltura_descartada"]:
            with st.expander("Ver el texto descartado y por qué"):
                st.caption(guardia.resumen(cerco))
                st.markdown(f"> {final['envoltura_descartada']}")

    with st.expander("🔒 Cómo se verificó este texto"):
        st.markdown(
            f"El modelo de lenguaje **no produce números**. Redacta sobre resultados que "
            f"ya existen, y después se comprueba una por una que cada cifra de su texto "
            f"aparezca entre las que se le entregaron. Si una sola no aparece, el texto "
            f"completo se descarta y queda solo la interpretación determinista.\n\n"
            f"- Cifras en el texto: **{cerco.get('total', 0)}**\n"
            f"- Trazadas a un resultado del motor: **{cerco.get('verificados', 0)}**\n"
            f"- Intentos de redacción: **{est.get('intentos_redaccion', 0)}** "
            f"(el cerco puede pedir una reescritura antes de descartar)\n"
            f"- Intentos de traducción: **{est.get('intentos_traduccion', 0)}** "
            f"(el validador del motor puede pedir correcciones)")
        if cerco.get("intrusos"):
            st.markdown("**Cifras rechazadas:** " +
                        ", ".join(f"`{i['literal']}`" for i in cerco["intrusos"]))
