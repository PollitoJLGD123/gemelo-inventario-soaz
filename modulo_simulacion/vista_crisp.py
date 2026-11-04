# modulo_simulacion/vista_crisp.py
"""Pantalla CRISP-DM. Las cifras salen de crisp_dm.py; aquí solo se muestran."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import narrativa
from .crisp_dm import (
    ALPHA, cargar_protocolo, eda_tablas, texto_friedman, texto_seleccion,
)

COLOR = {
    "stock": "#1d4ed8", "reorden": "#b45309", "ok": "#15803d", "neutro": "#64748b",
}
PLANTILLA = "plotly_white"

DIAS = {
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles",
    "Thursday": "jueves", "Friday": "viernes", "Saturday": "sábado",
    "Sunday": "domingo",
}


def mostrar(panel):
    st.markdown(
        "Metodología **CRISP-DM** sobre las ventas reales de la tienda 44. "
        "El orden de la pantalla es el de la evaluación que pidieron: exploración, "
        "entrenamiento, hiperparámetros, validación cruzada, selección del modelo, "
        "pruebas estadísticas y reporte.")
    fases = pd.DataFrame([
        {"Fase CRISP-DM": "Entendimiento del negocio",
         "Dónde está": "Decidir la reposición con un pronóstico de demanda medido, no a ojo."},
        {"Fase CRISP-DM": "Entendimiento de los datos",
         "Dónde está": "1. EDA"},
        {"Fase CRISP-DM": "Preparación",
         "Dónde está": "Tienda 44, ítems de mayor venta, día sin fila = 0. Sin precio ni lead time."},
        {"Fase CRISP-DM": "Modelado",
         "Dónde está": "2. Entrenamiento y 3. Hiperparámetros"},
        {"Fase CRISP-DM": "Evaluación",
         "Dónde está": "4. Validación cruzada, 5. Selección y 6. Pruebas"},
        {"Fase CRISP-DM": "Despliegue",
         "Dónde está": "El modelo elegido queda declarado. El gemelo sigue simulando con el motor."},
    ])
    narrativa.mostrar_tabla(
        fases, "Las seis fases y en qué paso de esta pantalla viven",
        {"que_muestra": "El mapa de CRISP-DM contra los siete apartados de abajo.",
         "por_que": "La metodología no es un modelo: es el orden en que se justifica "
                    "el pronóstico que después usa el gemelo.",
         "limitacion": "Desplegar aquí significa dejar el ganador escrito y usable "
                       "en el reporte. No sustituye la aprobación humana de una compra."})

    paquete = cargar_protocolo()
    pasos = st.tabs([
        "1. EDA", "2. Entrenamiento", "3. Hiperparámetros",
        "4. Validación cruzada", "5. Selección", "6. Pruebas", "7. Reporte",
    ])
    with pasos[0]:
        _eda(panel)
    if paquete is None:
        for paso in pasos[1:]:
            with paso:
                st.info("El protocolo todavía no está calculado. "
                        "Se genera con `python -m modulo_simulacion.crisp_dm`.")
        return
    datos, metricas, cortes = paquete
    with pasos[1]:
        _entrenamiento(datos, metricas)
    with pasos[2]:
        _hiperparametros(datos)
    with pasos[3]:
        _validacion(datos, metricas)
    with pasos[4]:
        _seleccion(datos)
    with pasos[5]:
        _pruebas(datos)
    with pasos[6]:
        _reporte(panel, datos, metricas, cortes)


def _eda(panel):
    por_item, diario, perfil = eda_tablas(panel)
    ceros = float((panel["unit_sales"].clip(lower=0) == 0).mean() * 100)
    st.caption(f"Días-ítem con venta 0: {ceros:.1f}%. "
               "Favorita no registra ese día; el panel lo rellena.")

    fig = go.Figure(go.Scatter(
        x=diario["date"], y=diario["u"], mode="lines",
        line=dict(color=COLOR["stock"], width=1.3)))
    fig.update_layout(template=PLANTILLA, height=380,
                      yaxis_title="Unidades (suma del alcance)", margin=dict(t=30))
    narrativa.mostrar_figura(
        fig, "Serie agregada del alcance",
        {"que_muestra": "Suma diaria de las unidades de los ítems que entran a CRISP-DM.",
         "por_que": "Es la demanda observada de la tienda 44, no una simulación.",
         "limitacion": "Un cero puede ser falta de stock o falta de clientes. "
                       "El dataset no lo distingue."},
        clave="crisp_eda_serie")

    fig_s = go.Figure(go.Bar(
        x=[DIAS.get(d, d) for d in perfil.index], y=perfil.to_numpy(),
        marker_color=COLOR["reorden"]))
    fig_s.update_layout(template=PLANTILLA, height=360,
                        yaxis_title="Unidades medias", margin=dict(t=30))
    pico = DIAS.get(perfil.idxmax(), str(perfil.idxmax()))
    narrativa.mostrar_figura(
        fig_s, "Perfil semanal",
        {"que_muestra": f"Media de unidades por día de la semana. El pico está el {pico}.",
         "por_que": "La estacionalidad semanal es la razón del orden de Fourier de Prophet "
                    "y del rezago 7 del ingenuo y de LightGBM.",
         "limitacion": "Es un promedio de todo el histórico. No separa feriados ni el "
                       "terremoto de abril de 2016."},
        clave="crisp_eda_semana")

    narrativa.mostrar_tabla(
        por_item.sort_values("media", ascending=False)[
            ["sku", "familia", "media", "pct_cero", "promo"]
        ].round(2).rename(columns={
            "media": "Media diaria", "pct_cero": "% días en cero", "promo": "% días en promoción",
        }),
        "Cada ítem del alcance",
        {"que_muestra": "Media, porcentaje de ceros y tasa de promoción, por ítem.",
         "por_que": "Salen de train.csv y de onpromotion. El costo no está porque "
                    "el dataset no trae precios.",
         "limitacion": "La promoción no es un experimento: se enciende cuando ya "
                       "se espera vender más."})


def _entrenamiento(datos, metricas):
    n = datos["n_series"]
    origenes = datos["origenes"]
    filas = pd.DataFrame([
        {"Modelo": "Ingenuo semanal", "Qué se ajusta": "Nada. Copia la última semana.",
         "Ajustes en la validación": n * origenes},
        {"Modelo": "Prophet",
         "Qué se ajusta": f"Estacionalidad semanal de orden {datos['elegidos']['fourier']} y anual.",
         "Ajustes en la validación": n * origenes},
        {"Modelo": "LightGBM",
         "Qué se ajusta": (f"80 árboles, {datos['elegidos']['num_leaves']} hojas, "
                           f"tasa {datos['elegidos']['learning_rate']}."),
         "Ajustes en la validación": n * origenes},
        {"Modelo": "LSTM, N-BEATS, DeepAR, TFT, SARIMA",
         "Qué se ajusta": "No se reentrenan en este corte.",
         "Ajustes en la validación": 0},
    ])
    narrativa.mostrar_tabla(
        filas, "Qué se entrenó y qué no",
        {"que_muestra": (f"{n} series × {origenes} orígenes. "
                         f"Horizonte {datos['horizonte']} días. "
                         f"Cortes registrados: {metricas['fecha_corte'].nunique()}."),
         "por_que": "Prophet y LightGBM son locales o se ajustan por serie. "
                    "Los archivos de models/ vienen de otra corrida y de otro dataset.",
         "limitacion": "Dejar fuera a los modelos profundos es una decisión de cómputo, "
                       "no una prueba de que sean peores en Favorita."})


def _hiperparametros(datos):
    grilla = pd.DataFrame(datos["grilla"])
    fig = go.Figure(go.Bar(
        x=grilla["config"], y=grilla["mase_mediana"],
        marker_color=[COLOR["ok"] if _es_elegida(fila, datos) else COLOR["neutro"]
                      for _, fila in grilla.iterrows()]))
    fig.update_layout(template=PLANTILLA, height=420,
                      yaxis_title="MASE mediano en la grilla", margin=dict(t=30))
    elegidos = datos["elegidos"]
    narrativa.mostrar_figura(
        fig, "Grilla de hiperparámetros",
        {"que_muestra": (f"Verde: lo elegido. Prophet con Fourier {elegidos['fourier']}. "
                         f"LightGBM con {elegidos['num_leaves']} hojas y tasa "
                         f"{elegidos['learning_rate']}."),
         "por_que": (f"Se midió en {datos['n_grilla']} series "
                     f"({', '.join(datos['skus_grilla'][:4])}…). "
                     "El criterio es la mediana del MASE en los tres orígenes."),
         "limitacion": "La grilla es corta a propósito. No explora changepoints de "
                       "Prophet ni profundidad mayor de LightGBM."},
        clave="crisp_grilla")
    narrativa.mostrar_tabla(
        grilla[["modelo", "config", "mase_mediana", "n_cortes"]].round(3),
        "Resultado de cada configuración",
        {"que_muestra": "Un renglón por configuración probada.",
         "por_que": "n_cortes es series de la grilla por orígenes.",
         "limitacion": "Elegir aquí y volver a medir en las mismas series infla un poco "
                       "al ganador. Por eso la selección final usa todas las series, "
                       "con estos hiperparámetros ya congelados."})


def _es_elegida(fila, datos):
    e = datos["elegidos"]
    if fila["modelo"] == "Prophet":
        return int(fila["fourier"]) == int(e["fourier"])
    return (int(fila["num_leaves"]) == int(e["num_leaves"])
            and float(fila["learning_rate"]) == float(e["learning_rate"]))


def _validacion(datos, metricas):
    resumen = (metricas.groupby(["modelo", "fold"])["mase"].median()
               .reset_index())
    fig = go.Figure()
    paleta = {"Ingenuo semanal": COLOR["neutro"], "Prophet": COLOR["stock"],
              "LightGBM": COLOR["ok"]}
    for modelo, grupo in resumen.groupby("modelo"):
        fig.add_trace(go.Bar(
            x=grupo["fold"].astype(str), y=grupo["mase"],
            name=modelo, marker_color=paleta.get(modelo, COLOR["reorden"])))
    fig.update_layout(template=PLANTILLA, height=420, barmode="group",
                      xaxis_title="Origen (1 = más antiguo, 3 = más reciente)",
                      yaxis_title="MASE mediano", margin=dict(t=30))
    narrativa.mostrar_figura(
        fig, "Validación cruzada por origen",
        {"que_muestra": (f"Tres ventanas de {datos['horizonte']} días, separadas "
                         f"{datos['horizonte']} días. El entrenamiento solo ve lo anterior al corte."),
         "por_que": "Un solo holdout final puede caer en una semana rara. "
                    "Tres orígenes preguntan si el orden de los modelos se sostiene.",
         "limitacion": "No es un k-fold aleatorio. Barajar días de una serie temporal "
                       "dejaría que el modelo viera el futuro."},
        clave="crisp_cv")


def _seleccion(datos):
    ranking = pd.DataFrame(datos["ranking"])
    colores = [COLOR["ok"] if m == datos["ganador"] else COLOR["neutro"]
               for m in ranking["modelo"]]
    fig = go.Figure(go.Bar(
        x=ranking["modelo"], y=ranking["mase_mediana"], marker_color=colores))
    fig.update_layout(template=PLANTILLA, height=400,
                      yaxis_title="MASE mediano entre series", margin=dict(t=30))
    narrativa.mostrar_figura(
        fig, "Selección del modelo",
        texto_seleccion(datos),
        clave="crisp_seleccion")
    narrativa.mostrar_tabla(
        ranking.round(3),
        "Ranking",
        {"que_muestra": "Mediana del MASE, del WAPE y del MAE.",
         "por_que": texto_seleccion(datos)["por_que"],
         "limitacion": texto_seleccion(datos)["limitacion"]})


def _pruebas(datos):
    friedman = datos["friedman"]
    ranking = datos["ranking"]
    st.metric("Friedman, p-valor", f"{friedman['p_valor']:.4f}",
              help=f"Estadístico {friedman['estadistico']:.2f}. Alpha {ALPHA}.")
    narrativa.mostrar_tabla(
        pd.DataFrame([{
            "Estadístico": round(friedman["estadistico"], 3),
            "p-valor": round(friedman["p_valor"], 4),
            "Series": friedman["n_series"],
            "Rechaza igualdad al 5%": "Sí" if friedman["rechaza"] else "No",
        }]),
        "Friedman sobre el MASE",
        texto_friedman(friedman, ranking))

    wil = pd.DataFrame(datos["wilcoxon"])
    narrativa.mostrar_tabla(
        wil.assign(
            p_valor=lambda d: d["p_valor"].round(4),
            estadistico=lambda d: d["estadistico"].round(3),
            mase_a=lambda d: d["mase_a"].round(3),
            mase_b=lambda d: d["mase_b"].round(3),
            rechaza=lambda d: np.where(d["rechaza"], "Sí", "No"),
        )[["modelo_a", "modelo_b", "mase_a", "mase_b", "estadistico", "p_valor",
           "rechaza", "menor_mase"]],
        "Wilcoxon de a pares",
        {"que_muestra": "Rangos con signo del MASE medio por serie, un par de modelos.",
         "por_que": "Si p < 0.05, la diferencia de esa pareja no se explica por el "
                    "azar de las series. menor_mase es quien tiene la mediana más baja, "
                    "haya o no rechazo.",
         "limitacion": "Tres parejas sin corrección por comparaciones múltiples. "
                       "El alpha sigue siendo 0.05 en cada fila."})

    dm = pd.DataFrame(datos["diebold"])
    narrativa.mostrar_tabla(
        dm.assign(
            estadistico=lambda d: d["estadistico"].round(3),
            p_valor=lambda d: d["p_valor"].round(4),
            media_diferencial=lambda d: d["media_diferencial"].round(2),
            rechaza=lambda d: np.where(d["rechaza"], "Sí", "No"),
        )[["modelo_a", "modelo_b", "estadistico", "p_valor", "rechaza",
           "menor_perdida", "series_con_rechazo", "n_series"]],
        "Diebold-Mariano en el origen más reciente",
        {"que_muestra": "En cada serie, diferencia de error cuadrático de los 28 días "
                        "del último origen, con Newey-West de 7 días. La fila resume "
                        "esas medias con una t de Student entre series.",
         "por_que": "Un diferencial negativo quiere decir que el modelo A pierde menos "
                    "que el B. series_con_rechazo cuenta en cuántas series el contraste "
                    "individual sale significativo al 5%.",
         "limitacion": "Dentro de cada serie hay 28 errores y el horizonte también es 28, "
                       "así que no se aplica la corrección de Harvey de horizonte completo. "
                       "La t entre series supone que los ítems son independientes."})


def _reporte(panel, datos, metricas, cortes):
    documento = _html(panel, datos, metricas, cortes)
    st.download_button(
        "Descargar reporte HTML", documento,
        file_name="crisp_dm_favorita_tienda44.html", mime="text/html",
        use_container_width=True)
    st.caption("El archivo repite las tablas de esta pantalla y, bajo cada una, "
               "el mismo pie de interpretabilidad.")


def _html(panel, datos, metricas, cortes):
    por_item, _diario, perfil = eda_tablas(panel)
    bloques = [
        _bloque_tabla(
            "1. EDA — ítems",
            por_item.sort_values("media", ascending=False)[
                ["sku", "familia", "media", "pct_cero", "promo"]].round(2),
            {"que_muestra": "Media diaria, porcentaje de ceros y promoción.",
             "por_que": "Columnas del dataset público, tienda 44.",
             "limitacion": "El cero rellenado no es una ruptura comprobada."}),
        _bloque_tabla(
            "1. EDA — perfil semanal",
            pd.DataFrame({"dia": [DIAS.get(d, d) for d in perfil.index],
                          "media": perfil.to_numpy().round(2)}),
            {"que_muestra": "Unidades medias por día de la semana.",
             "por_que": "Justifica el ciclo de 7 días de los tres modelos.",
             "limitacion": "No separa feriados."}),
        _bloque_tabla(
            "3. Hiperparámetros",
            pd.DataFrame(datos["grilla"])[["modelo", "config", "mase_mediana"]].round(3),
            {"que_muestra": f"Grilla en {datos['n_grilla']} series. "
                            f"Prophet Fourier {datos['elegidos']['fourier']}. "
                            f"LightGBM hojas {datos['elegidos']['num_leaves']}, "
                            f"tasa {datos['elegidos']['learning_rate']}.",
             "por_que": "Menor MASE mediano dentro de cada familia.",
             "limitacion": "Grilla corta, medida en las series de mayor venta."}),
        _bloque_tabla(
            "4 y 5. Validación cruzada y selección",
            pd.DataFrame(datos["ranking"]).round(3),
            texto_seleccion(datos)),
        _bloque_tabla(
            "6. Friedman",
            pd.DataFrame([datos["friedman"]]),
            texto_friedman(datos["friedman"], datos["ranking"])),
        _bloque_tabla(
            "6. Wilcoxon",
            pd.DataFrame(datos["wilcoxon"]).round(4),
            {"que_muestra": "Pares de MASE por serie.",
             "por_que": "Prueba de rangos, alpha 0.05, sin corrección múltiple.",
             "limitacion": "Tres comparaciones comparten las mismas series."}),
        _bloque_tabla(
            "6. Diebold-Mariano",
            pd.DataFrame(datos["diebold"]).round(4),
            {"que_muestra": "Diebold-Mariano por serie en el origen más reciente, "
                            "resumido con una t entre series.",
             "por_que": "Newey-West con rezago 7 dentro de cada serie.",
             "limitacion": "No usa la corrección de Harvey de horizonte 28, "
                           "y trata a los ítems como independientes."}),
    ]
    del metricas, cortes
    cuerpo = "\n".join(bloques)
    ganador = html.escape(str(datos["ganador"]))
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>CRISP-DM — Favorita tienda 44</title>
<style>
body {{ font-family: Georgia, serif; max-width: 920px; margin: 2rem auto; color: #1c1917; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; }}
th, td {{ border: 1px solid #d6d3d1; padding: 0.35rem 0.5rem; text-align: left; font-size: 0.92rem; }}
th {{ background: #f5f5f4; }}
.pie {{ background: #fafaf9; border-left: 3px solid #0f766e; padding: 0.6rem 0.9rem; margin: 0 0 1.8rem; }}
h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-bottom: 0.3rem; }}
</style></head><body>
<h1>CRISP-DM — pronóstico de demanda</h1>
<p>Corporación Favorita, tienda 44, Quito. Series de clase A del corte cacheado.
Horizonte {datos['horizonte']} días, {datos['origenes']} orígenes.
Modelo seleccionado: <b>{ganador}</b>. {html.escape(datos['criterio'])}</p>
{cuerpo}
</body></html>"""


def _bloque_tabla(titulo, df, interp):
    tabla = df.to_html(index=False, border=0)
    return f"""<h2>{html.escape(titulo)}</h2>
{tabla}
<div class="pie">
<p><b>Qué muestra.</b> {html.escape(interp['que_muestra'])}</p>
<p><b>Por qué sale así.</b> {html.escape(interp['por_que'])}</p>
<p><b>Qué no se puede concluir.</b> {html.escape(interp['limitacion'])}</p>
</div>"""
