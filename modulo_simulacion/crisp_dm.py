# modulo_simulacion/crisp_dm.py
"""
Protocolo CRISP-DM del pronóstico, sobre el corte real de Favorita.

No reentrena LSTM, N-BEATS, DeepAR ni TFT: esos archivos se ajustaron en otro
dataset y no caben en una sesión de la aplicación. Aquí se comparan tres
pronósticos que sí se estiman sobre estas series:

  ingenuo semanal  — repite la última semana
  Prophet          — la especificación local, con el orden de Fourier elegido
  LightGBM         — rezagos y día de la semana, con la grilla elegida

La validación es cruzada en el tiempo: tres orígenes, horizonte de 28 días,
sin mezclar el futuro en el ajuste. Los hiperparámetros se eligen en las ocho
series de mayor venta y después se congelan para el resto.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .favorita import RAIZ
from .prophet_demanda import escala_mase

CACHE = RAIZ / "data" / "favorita_grocery" / "cache"
RUTA_JSON = CACHE / "crisp_dm.json"
RUTA_METRICAS = CACHE / "crisp_dm_metricas.parquet"
RUTA_CORTES = CACHE / "crisp_dm_cortes.parquet"

HORIZONTE = 28
ORIGENES = 3
PASO = 28
MIN_ENTRENAMIENTO = 180
N_GRILLA = 8
ESTACION = 7
ALPHA = 0.05
LAGS = (1, 7, 14, 28)
FOURIER = (3, 5, 8)
GRILLA_LGB = (
    {"num_leaves": 15, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.1},
)


def pronostico_ingenuo(y, horizonte=HORIZONTE, estacion=ESTACION):
    """Repite el último ciclo estacional. No estima ningún parámetro."""
    historia = [float(v) for v in np.asarray(y, dtype=float)]
    if len(historia) < estacion:
        media = float(np.mean(historia)) if historia else 0.0
        return np.full(horizonte, media)
    salida = []
    for _ in range(horizonte):
        salida.append(historia[-estacion])
        historia.append(historia[-estacion])
    return np.asarray(salida, dtype=float)


def mase(y_true, y_pred, y_train, estacion=ESTACION):
    escala = escala_mase(y_train, estacion)
    if not np.isfinite(escala) or escala <= 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / escala)


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(100 * np.sum(np.abs(y_true - np.asarray(y_pred))) / denom)


def cortes_validacion(n, horizonte=HORIZONTE, origenes=ORIGENES, paso=PASO,
                      minimo=MIN_ENTRENAMIENTO):
    """Índices donde acaba el entrenamiento, del más antiguo al más reciente."""
    puntos = []
    for i in range(origenes):
        fin = n - horizonte - i * paso
        if fin >= minimo:
            puntos.append(fin)
    return list(reversed(puntos))


def preparar_series(panel):
    """Una serie diaria no negativa por ítem, ordenada en el tiempo."""
    series = []
    for item, grupo in panel.groupby("item_nbr"):
        grupo = grupo.sort_values("date")
        y = grupo["unit_sales"].clip(lower=0).to_numpy(dtype=float)
        if len(y) < MIN_ENTRENAMIENTO + HORIZONTE * ORIGENES:
            continue
        series.append({
            "item_nbr": int(item),
            "sku": f"IT-{int(item)}",
            "familia": str(grupo["family"].iloc[0]),
            "fechas": pd.to_datetime(grupo["date"]).to_numpy(),
            "y": y,
            "media": float(np.mean(y)),
            "pct_cero": float(np.mean(y == 0) * 100),
            "tasa_promo": float(grupo["onpromotion"].astype(bool).mean() * 100),
        })
    series.sort(key=lambda s: -s["media"])
    return series


def _pronostico_prophet(fechas, y, horizonte, fourier):
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    tren = pd.DataFrame({"ds": pd.to_datetime(fechas), "y": np.asarray(y, dtype=float)})
    modelo = Prophet(
        weekly_seasonality=int(fourier),
        yearly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    modelo.fit(tren)
    futuro = modelo.make_future_dataframe(periods=horizonte, freq="D")
    return np.clip(modelo.predict(futuro)["yhat"].to_numpy()[-horizonte:], 0, None)


def _pronostico_lightgbm(y, fechas, horizonte, params):
    import lightgbm as lgb

    y = np.asarray(y, dtype=float)
    dias = pd.to_datetime(fechas).dayofweek.to_numpy()
    max_lag = max(LAGS)
    filas, destino = [], []
    for t in range(max_lag, len(y)):
        filas.append([y[t - lag] for lag in LAGS] + [int(dias[t])])
        destino.append(y[t])
    booster = lgb.train(
        {
            "objective": "regression",
            "num_leaves": int(params["num_leaves"]),
            "learning_rate": float(params["learning_rate"]),
            "verbosity": -1,
            "num_threads": 1,
            "force_col_wise": True,
        },
        lgb.Dataset(np.asarray(filas), label=np.asarray(destino)),
        num_boost_round=80,
    )
    historia = list(y)
    futuros = pd.date_range(
        pd.Timestamp(fechas[-1]) + pd.Timedelta(days=1), periods=horizonte, freq="D")
    salida = []
    for i, dia in enumerate(futuros.dayofweek):
        t = len(historia)
        fila = np.asarray([[historia[t - lag] for lag in LAGS] + [int(dia)]])
        predicho = max(0.0, float(booster.predict(fila)[0]))
        salida.append(predicho)
        historia.append(predicho)
    return np.asarray(salida, dtype=float)


def _evaluar_serie(serie, fin, modelo, config):
    y = serie["y"]
    fechas = serie["fechas"]
    tren, real = y[:fin], y[fin:fin + HORIZONTE]
    if modelo == "Ingenuo semanal":
        pronostico = pronostico_ingenuo(tren)
    elif modelo == "Prophet":
        pronostico = _pronostico_prophet(fechas[:fin], tren, HORIZONTE, config["fourier"])
    elif modelo == "LightGBM":
        pronostico = _pronostico_lightgbm(tren, fechas[:fin], HORIZONTE, config)
    else:
        raise ValueError(modelo)
    return {
        "item_nbr": serie["item_nbr"],
        "sku": serie["sku"],
        "familia": serie["familia"],
        "modelo": modelo,
        "mase": mase(real, pronostico, tren),
        "wape": wape(real, pronostico),
        "mae": float(np.mean(np.abs(real - pronostico))),
        "rmse": float(np.sqrt(np.mean((real - pronostico) ** 2))),
        "real": real,
        "pronostico": pronostico,
        "fecha_corte": pd.Timestamp(fechas[fin]).date().isoformat(),
    }


def _mediana(valores):
    valores = [v for v in valores if np.isfinite(v)]
    if not valores:
        return float("nan")
    return float(np.median(valores))


def _buscar_grilla(series):
    """Elige el Fourier de Prophet y las hojas de LightGBM en las series de la grilla."""
    filas = []
    for fourier in FOURIER:
        masas = []
        for serie in series:
            for fin in cortes_validacion(len(serie["y"])):
                corte = _evaluar_serie(serie, fin, "Prophet", {"fourier": fourier})
                masas.append(corte["mase"])
        filas.append({
            "modelo": "Prophet",
            "config": f"Fourier semanal {fourier}",
            "fourier": fourier,
            "num_leaves": None,
            "learning_rate": None,
            "mase_mediana": _mediana(masas),
            "n_cortes": len(masas),
        })
        print(f"  grilla Prophet fourier={fourier}: MASE {_mediana(masas):.3f}")
    for params in GRILLA_LGB:
        masas = []
        for serie in series:
            for fin in cortes_validacion(len(serie["y"])):
                corte = _evaluar_serie(serie, fin, "LightGBM", params)
                masas.append(corte["mase"])
        filas.append({
            "modelo": "LightGBM",
            "config": f"hojas {params['num_leaves']}, tasa {params['learning_rate']}",
            "fourier": None,
            "num_leaves": params["num_leaves"],
            "learning_rate": params["learning_rate"],
            "mase_mediana": _mediana(masas),
            "n_cortes": len(masas),
        })
        print(f"  grilla LightGBM {params}: MASE {_mediana(masas):.3f}")
    tabla = pd.DataFrame(filas)
    prophet = tabla[tabla["modelo"] == "Prophet"].sort_values("mase_mediana").iloc[0]
    luz = tabla[tabla["modelo"] == "LightGBM"].sort_values("mase_mediana").iloc[0]
    return tabla, {
        "fourier": int(prophet["fourier"]),
        "num_leaves": int(luz["num_leaves"]),
        "learning_rate": float(luz["learning_rate"]),
    }


def _validar(series, elegidos):
    metricas, cortes = [], []
    config_p = {"fourier": elegidos["fourier"]}
    config_l = {"num_leaves": elegidos["num_leaves"],
                "learning_rate": elegidos["learning_rate"]}
    modelos = (
        ("Ingenuo semanal", None),
        ("Prophet", config_p),
        ("LightGBM", config_l),
    )
    for i, serie in enumerate(series, start=1):
        puntos = cortes_validacion(len(serie["y"]))
        for modelo, config in modelos:
            for fold, fin in enumerate(puntos, start=1):
                corte = _evaluar_serie(serie, fin, modelo, config or {})
                metricas.append({
                    "item_nbr": corte["item_nbr"], "sku": corte["sku"],
                    "familia": corte["familia"], "modelo": modelo, "fold": fold,
                    "fecha_corte": corte["fecha_corte"],
                    "mase": corte["mase"], "wape": corte["wape"],
                    "mae": corte["mae"], "rmse": corte["rmse"],
                })
                for paso, (real, pred) in enumerate(zip(corte["real"], corte["pronostico"]), start=1):
                    cortes.append({
                        "item_nbr": corte["item_nbr"], "modelo": modelo, "fold": fold,
                        "paso": paso, "real": float(real), "pronostico": float(pred),
                    })
        print(f"  validación {i}/{len(series)} {serie['sku']}")
    return pd.DataFrame(metricas), pd.DataFrame(cortes)


def diebold_mariano(perdida_a, perdida_b, rezago=ESTACION):
    """
    Diferencial de pérdida apilado. Estadístico negativo: A pierde menos que B.
    El rezago de Newey-West es la semana, no el horizonte completo: con 28
    errores y rezago 28 la corrección de Harvey se anula.
    """
    d = np.asarray(perdida_a, dtype=float) - np.asarray(perdida_b, dtype=float)
    n = len(d)
    media = float(np.mean(d))
    centrado = d - media
    gamma0 = float(np.dot(centrado, centrado) / n)
    varianza = gamma0
    for k in range(1, rezago + 1):
        gamma = float(np.dot(centrado[k:], centrado[:-k]) / n)
        varianza += 2 * (1 - k / (rezago + 1)) * gamma
    varianza = max(varianza, 1e-12) / n
    estadistico = media / np.sqrt(varianza)
    p_valor = float(2 * (1 - stats.norm.cdf(abs(estadistico))))
    return {"estadistico": float(estadistico), "p_valor": p_valor, "media_diferencial": media}


def _dm_entre_series(bloque, modelo_a, modelo_b):
    """Un Diebold-Mariano por serie y una t de Student sobre esas medias."""
    a = bloque[bloque["modelo"] == modelo_a][
        ["item_nbr", "paso", "real", "pronostico"]].rename(columns={"pronostico": "pa"})
    b = bloque[bloque["modelo"] == modelo_b][
        ["item_nbr", "paso", "pronostico"]].rename(columns={"pronostico": "pb"})
    unido = a.merge(b, on=["item_nbr", "paso"])
    medias = []
    rechazos = 0
    for _, grupo in unido.groupby("item_nbr"):
        grupo = grupo.sort_values("paso")
        perdida_a = (grupo["real"] - grupo["pa"]) ** 2
        perdida_b = (grupo["real"] - grupo["pb"]) ** 2
        prueba = diebold_mariano(perdida_a, perdida_b)
        medias.append(prueba["media_diferencial"])
        if prueba["p_valor"] < ALPHA:
            rechazos += 1
    medias = np.asarray(medias, dtype=float)
    media = float(np.mean(medias)) if len(medias) else float("nan")
    if len(medias) < 2 or float(np.std(medias, ddof=1)) == 0:
        estadistico, p_valor = 0.0, 1.0
    else:
        estadistico, p_valor = stats.ttest_1samp(medias, 0.0)
    return {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "media_diferencial": media,
        "n_series": int(len(medias)),
        "series_con_rechazo": int(rechazos),
    }


def pruebas_estadisticas(metricas, cortes):
    """Friedman y Wilcoxon sobre el MASE medio por serie; Diebold-Mariano en el último pliegue."""
    ancho = (metricas.groupby(["sku", "modelo"])["mase"].mean().unstack("modelo"))
    ancho = ancho.dropna()
    columnas = [c for c in ("Ingenuo semanal", "Prophet", "LightGBM") if c in ancho.columns]
    ancho = ancho[columnas]
    estadistico, p_valor = stats.friedmanchisquare(*[ancho[c].to_numpy() for c in columnas])
    friedman = {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "n_series": int(len(ancho)),
        "alpha": ALPHA,
        "rechaza": bool(p_valor < ALPHA),
    }
    wilcoxon = []
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            try:
                stat, p = stats.wilcoxon(ancho[a], ancho[b], zero_method="wilcox")
            except ValueError:
                stat, p = float("nan"), 1.0
            mejor = a if float(ancho[a].median()) <= float(ancho[b].median()) else b
            wilcoxon.append({
                "modelo_a": a, "modelo_b": b,
                "estadistico": float(stat), "p_valor": float(p),
                "rechaza": bool(p < ALPHA),
                "menor_mase": mejor,
                "mase_a": float(ancho[a].median()),
                "mase_b": float(ancho[b].median()),
            })
    ultimo = int(cortes["fold"].max())
    bloque = cortes[cortes["fold"] == ultimo]
    diebold = []
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            prueba = _dm_entre_series(bloque, a, b)
            prueba.update({
                "modelo_a": a, "modelo_b": b,
                "rechaza": bool(prueba["p_valor"] < ALPHA),
                "menor_perdida": a if prueba["media_diferencial"] < 0 else b,
            })
            diebold.append(prueba)
    return friedman, wilcoxon, diebold, ancho


def resumen_modelos(metricas):
    filas = []
    for modelo, grupo in metricas.groupby("modelo"):
        por_serie = grupo.groupby("sku")[["mase", "wape"]].mean()
        filas.append({
            "modelo": modelo,
            "mase_mediana": float(np.nanmedian(por_serie["mase"])),
            "wape_mediana": float(np.nanmedian(por_serie["wape"])),
            "mae_mediana": float(np.nanmedian(grupo["mae"])),
        })
    tabla = pd.DataFrame(filas).sort_values("mase_mediana")
    return tabla


def ejecutar_protocolo(panel, limite=None, guardar=True):
    """Corre grilla, validación cruzada, selección y pruebas. `limite` recorta series."""
    series = preparar_series(panel)
    if limite:
        series = series[:limite]
    grilla_series = series[:min(N_GRILLA, len(series))]
    print(f"Series: {len(series)}. Grilla sobre {len(grilla_series)}.")
    tabla_grilla, elegidos = _buscar_grilla(grilla_series)
    print(f"Elegidos: {elegidos}")
    metricas, cortes = _validar(series, elegidos)
    ranking = resumen_modelos(metricas)
    ganador = str(ranking.iloc[0]["modelo"])
    friedman, wilcoxon, diebold, _ancho = pruebas_estadisticas(metricas, cortes)
    paquete = {
        "n_series": len(series),
        "n_grilla": len(grilla_series),
        "horizonte": HORIZONTE,
        "origenes": ORIGENES,
        "elegidos": elegidos,
        "ganador": ganador,
        "criterio": "Menor mediana del MASE por serie, promediando los tres orígenes.",
        "grilla": tabla_grilla.to_dict(orient="records"),
        "ranking": ranking.to_dict(orient="records"),
        "friedman": friedman,
        "wilcoxon": wilcoxon,
        "diebold": diebold,
        "skus_grilla": [s["sku"] for s in grilla_series],
    }
    if guardar:
        CACHE.mkdir(parents=True, exist_ok=True)
        RUTA_JSON.write_text(json.dumps(paquete, ensure_ascii=False, indent=2), encoding="utf-8")
        metricas.to_parquet(RUTA_METRICAS, index=False)
        cortes.to_parquet(RUTA_CORTES, index=False)
        print(f"Guardado. Ganador: {ganador}")
    return paquete, metricas, cortes


def cargar_protocolo():
    """None si el protocolo todavía no se corrió."""
    if not RUTA_JSON.exists():
        return None
    paquete = json.loads(RUTA_JSON.read_text(encoding="utf-8"))
    metricas = pd.read_parquet(RUTA_METRICAS)
    cortes = pd.read_parquet(RUTA_CORTES)
    return paquete, metricas, cortes


def eda_tablas(panel):
    """Resúmenes de exploración que no dependen del entrenamiento."""
    u = panel["unit_sales"].clip(lower=0)
    por_item = panel.assign(u=u, cero=u.eq(0)).groupby("item_nbr").agg(
        familia=("family", "first"),
        media=("u", "mean"),
        pct_cero=("cero", "mean"),
        promo=("onpromotion", "mean"),
    ).reset_index()
    por_item["sku"] = por_item["item_nbr"].map(lambda n: f"IT-{int(n)}")
    por_item["pct_cero"] = por_item["pct_cero"] * 100
    por_item["promo"] = por_item["promo"].astype(float) * 100
    diario = panel.assign(u=u).groupby("date", as_index=False)["u"].sum()
    semanal = panel.assign(u=u, dia=pd.to_datetime(panel["date"]).dt.day_name())
    perfil = semanal.groupby("dia")["u"].mean()
    orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    perfil = perfil.reindex([d for d in orden if d in perfil.index])
    return por_item, diario, perfil


def texto_friedman(friedman, ranking):
    orden = ", ".join(
        f"{f['modelo']} ({f['mase_mediana']:.2f})" for f in ranking)
    if friedman["rechaza"]:
        decision = (f"p = {friedman['p_valor']:.4f} < {ALPHA}. "
                    "Se rechaza que los tres modelos tengan el mismo MASE.")
    else:
        decision = (f"p = {friedman['p_valor']:.4f}. "
                    "No se rechaza que las diferencias de MASE sean ruido de muestreo.")
    return {
        "que_muestra": (f"Prueba de Friedman sobre el MASE medio de "
                        f"{friedman['n_series']} series. Orden por mediana: {orden}."),
        "por_que": (f"Estadístico {friedman['estadistico']:.2f}. {decision} "
                    "Es una prueba de rangos: no supone normalidad."),
        "limitacion": "Rechazar igualdad no dice cuál modelo gana. Eso lo dicen "
                      "Wilcoxon y Diebold-Mariano, de a pares.",
    }


def texto_seleccion(paquete):
    ganador = paquete["ganador"]
    fila = next(f for f in paquete["ranking"] if f["modelo"] == ganador)
    return {
        "que_muestra": (f"Mediana del MASE en la validación cruzada. "
                        f"El menor es {ganador} ({fila['mase_mediana']:.2f}). "
                        f"WAPE mediano {fila['wape_mediana']:.1f}%."),
        "por_que": paquete["criterio"] + " MASE < 1 mejora al ingenuo de hace una semana "
                   "en la escala de esa serie; MASE > 1 no lo mejora.",
        "limitacion": "El ganador lo es en este corte de la tienda 44 y con estos "
                      "tres candidatos. No se reentrenaron los modelos profundos "
                      "guardados en models/.",
    }


def _entrada():
    from .favorita import cargar_alcance, catalogo_desde_panel
    from .catalogo import catalogo_en_alcance, clasificar_abc

    panel = cargar_alcance()
    alcance = catalogo_en_alcance(clasificar_abc(catalogo_desde_panel(panel)))
    items = {int(s["item_nbr"]) for s in alcance}
    ejecutar_protocolo(panel[panel["item_nbr"].isin(items)])


if __name__ == "__main__":
    _entrada()
