# modulo_simulacion/prophet_demanda.py
"""
Evaluación del pronóstico con Prophet sobre las series reales de Favorita.

Prophet es un modelo local: el archivo models/prophet.pkl se entrenó sobre
una serie de referencia y no sabe de los ítems de la tienda 44. El README de
proyección de demanda lo dice así: en producción se reentrena sobre el
historial de la serie que se va a pronosticar, con la misma especificación
(estacionalidad semanal de orden 5 y estacionalidad anual, sin diaria).

Eso es lo que hace este módulo. El horizonte de evaluación es 28 días, el
mismo del protocolo de esa tesis. Las métricas son MAE, RMSE, WAPE y MASE.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .favorita import RAIZ

HOLDOUT = 28
ESTACIONALIDAD = 7
MODELO = RAIZ / "models" / "prophet.pkl"
CACHE = RAIZ / "data" / "favorita_grocery" / "cache" / "prophet_evaluacion.parquet"


def disponible():
    """(¿se puede pronosticar?, motivo si no)."""
    try:
        import prophet  # noqa: F401
    except ImportError:
        return False, ("Falta el paquete `prophet`. Se instala con "
                       "`pip install prophet==1.3.0` en el entorno del gemelo.")
    return True, "Prophet listo."


def evaluar_panel(panel, holdout=HOLDOUT):
    """
    Un pronóstico por ítem contra los últimos `holdout` días, que no entran
    al ajuste. Devuelve el detalle diario y las métricas por ítem.
    """
    esperados = {int(x) for x in panel["item_nbr"].unique()}
    if CACHE.exists():
        detalle = pd.read_parquet(CACHE)
        detalle["date"] = pd.to_datetime(detalle["date"])
        guardados = {int(x) for x in detalle["item_nbr"].unique()}
        if esperados <= guardados:
            detalle = detalle[detalle["item_nbr"].isin(esperados)].copy()
            return detalle, metricas_desde_detalle(detalle)

    listo, motivo = disponible()
    if not listo:
        raise RuntimeError(motivo)

    detalles = []
    for item, grupo in panel.groupby("item_nbr"):
        grupo = grupo.sort_values("date")
        if len(grupo) <= holdout + ESTACIONALIDAD + 30:
            continue
        y = grupo["unit_sales"].clip(lower=0).to_numpy(dtype=float)
        fechas = grupo["date"].to_numpy()
        pronostico = _pronosticar(fechas[:-holdout], y[:-holdout], holdout)
        real = y[-holdout:]
        tren = y[:-holdout]
        fila = pd.DataFrame({
            "date": fechas[-holdout:],
            "item_nbr": int(item),
            "familia": str(grupo["family"].iloc[0]),
            "real": real,
            "pronostico": np.clip(pronostico, 0, None),
            "media_historica": float(np.mean(tren)),
            "escala_mase": escala_mase(tren),
        })
        detalles.append(fila)
        print(f"  Prophet IT-{int(item)}: {len(detalles)} series")

    if not detalles:
        raise RuntimeError("Ninguna serie tuvo historial suficiente para Prophet.")
    detalle = pd.concat(detalles, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_parquet(CACHE, index=False)
    return detalle, metricas_desde_detalle(detalle)


def metricas_desde_detalle(detalle):
    filas = []
    for item, g in detalle.groupby("item_nbr"):
        y = g["real"].to_numpy(dtype=float)
        p = g["pronostico"].to_numpy(dtype=float)
        # La escala del MASE sale del histórico, no del holdout. Aquí solo
        # tenemos el holdout guardado; la media histórica viaja en la columna
        # y el MASE se recalcula si el detalle trae `escala`. Si no, se usa
        # la desviación del propio holdout como no-opción: mejor guardar escala.
        escala = g["escala_mase"].iloc[0] if "escala_mase" in g.columns else np.nan
        filas.append({
            "item_nbr": int(item),
            "sku": f"IT-{int(item)}",
            "familia": g["familia"].iloc[0],
            "MAE": _mae(y, p),
            "RMSE": _rmse(y, p),
            "WAPE": _wape(y, p),
            "MASE": float(np.mean(np.abs(y - p)) / escala) if escala and escala > 0 else np.nan,
            "media_historica": float(g["media_historica"].iloc[0]),
            "media_pronostico": float(np.mean(p)),
        })
    return pd.DataFrame(filas).sort_values("WAPE")


def _pronosticar(fechas, y, horizonte):
    from prophet import Prophet

    tren = pd.DataFrame({"ds": pd.to_datetime(fechas), "y": np.asarray(y, dtype=float)})
    # Orden 5 en la semana, anual activa, diaria apagada: la misma
    # especificación que src/prophet_model.py del proyecto de proyección.
    modelo = Prophet(
        weekly_seasonality=5,
        yearly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    modelo.fit(tren)
    futuro = modelo.make_future_dataframe(periods=horizonte, freq="D")
    salida = modelo.predict(futuro)
    return salida["yhat"].to_numpy()[-horizonte:]


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _wape(y, p):
    denom = float(np.sum(np.abs(y)))
    # Sin demanda real en el holdout el porcentaje no está definido.
    # Devolver 0 haría parecer un pronóstico perfecto.
    if denom == 0:
        return float("nan")
    return float(100 * np.sum(np.abs(y - p)) / denom)


def escala_mase(y, estacionalidad=ESTACIONALIDAD):
    y = np.asarray(y, dtype=float)
    if len(y) <= estacionalidad:
        return np.nan
    return float(np.mean(np.abs(y[estacionalidad:] - y[:-estacionalidad])))
