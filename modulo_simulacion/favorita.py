# modulo_simulacion/favorita.py
"""
Ventas reales de Corporación Favorita para el gemelo.

El archivo train.csv pesa varios GB y no trae filas de venta cero. Este módulo
se queda con la tienda de más transacciones (la 44, Quito), los ítems de mayor
venta, y un panel diario completo: el día que Favorita no registró se guarda
como 0. Eso no es una ruptura comprobada —el dataset no dice si había stock—
y así se declara en la interfaz.

El costo unitario y el tiempo de entrega no están en el dataset. Entran como
supuesto explícito para que el motor pueda correr; las unidades vendidas no
se inventan.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .parametros import PROVEEDORES

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "data" / "favorita_grocery"
CACHE = CARPETA / "cache" / "alcance_diario.parquet"

# La 44 es la de mayor transacción en transactions.csv (Quito, tipo A).
TIENDA = 44
N_ITEMS = 80
# Favorita no publica precios. El costo solo ordena el ABC y valora el stock;
# las unidades salen del dataset.
COSTO_PERECIBLE = 6.5
COSTO_SECO = 3.5
VIDA_PERECIBLE = 14
VIDA_SECO = 400


def cargar_alcance():
    """Panel diario cacheado. Lo construye la primera vez, leyendo train.csv."""
    if not CACHE.exists():
        construir_alcance()
    panel = pd.read_parquet(CACHE)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def construir_alcance(tienda=TIENDA, n_items=N_ITEMS):
    """
    Una pasada por train.csv: solo la tienda elegida, luego los n ítems
    de mayor venta. El resultado cabe en un parquet pequeño.
    """
    train = CARPETA / "train.csv"
    if not train.exists():
        raise FileNotFoundError(f"No está {train}")

    partes = []
    lector = pd.read_csv(
        train,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32",
               "onpromotion": "string"},
        parse_dates=["date"],
        chunksize=1_000_000,
    )
    for i, bloque in enumerate(lector, start=1):
        sub = bloque[bloque["store_nbr"] == tienda]
        if len(sub):
            partes.append(sub)
        if i % 20 == 0:
            print(f"  train.csv: {i} bloques, filas de la tienda {sum(len(p) for p in partes):,}")
    if not partes:
        raise RuntimeError(f"La tienda {tienda} no aparece en train.csv")

    ventas = pd.concat(partes, ignore_index=True)
    # Las unidades negativas son devoluciones. Para la demanda se truncan en 0
    # más abajo; aquí se conservan para poder contarlas.
    ranking = (ventas.assign(u=ventas["unit_sales"].clip(lower=0))
               .groupby("item_nbr")["u"].sum()
               .sort_values(ascending=False))
    elegidos = ranking.head(n_items).index
    ventas = ventas[ventas["item_nbr"].isin(elegidos)].copy()

    items = pd.read_csv(CARPETA / "items.csv", dtype={"item_nbr": "int32"})
    tienda_meta = pd.read_csv(CARPETA / "stores.csv", dtype={"store_nbr": "int16"})
    tienda_meta = tienda_meta[tienda_meta["store_nbr"] == tienda].iloc[0]

    ventas = ventas.merge(items, on="item_nbr", how="left")
    ventas["city"] = tienda_meta["city"]
    ventas["state"] = tienda_meta["state"]
    ventas["store_type"] = tienda_meta["type"]
    ventas["cluster"] = int(tienda_meta["cluster"])
    ventas["onpromotion"] = (
        ventas["onpromotion"].fillna("").str.strip().str.lower().isin(("true", "1", "t")))

    fechas = pd.date_range(ventas["date"].min(), ventas["date"].max(), freq="D")
    base = pd.MultiIndex.from_product(
        [fechas, elegidos], names=["date", "item_nbr"]).to_frame(index=False)
    panel = base.merge(
        ventas.drop(columns=["store_nbr"]),
        on=["date", "item_nbr"], how="left")
    panel["unit_sales"] = panel["unit_sales"].fillna(0.0)
    panel["onpromotion"] = panel["onpromotion"].fillna(False).astype(bool)
    panel["store_nbr"] = tienda
    meta = items.set_index("item_nbr")
    panel["family"] = panel["family"].fillna(panel["item_nbr"].map(meta["family"]))
    panel["class"] = panel["class"].fillna(panel["item_nbr"].map(meta["class"]))
    panel["perishable"] = panel["perishable"].fillna(panel["item_nbr"].map(meta["perishable"]))
    panel["perishable"] = panel["perishable"].fillna(0).astype("int8")
    panel["city"] = tienda_meta["city"]
    panel["state"] = tienda_meta["state"]
    panel["store_type"] = tienda_meta["type"]
    panel["cluster"] = int(tienda_meta["cluster"])

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE, index=False)
    print(f"Alcance guardado: {CACHE} ({len(panel):,} filas, {panel['item_nbr'].nunique()} ítems)")
    return panel


def catalogo_desde_panel(panel):
    """
    Un registro de catálogo por ítem, con la media observada como demanda.

    Los campos que el dataset sí trae (familia, clase, perecible, tienda) se
    copian. Costo, vida útil y proveedor se declaran porque no existen ahí.
    """
    proveedor = "Centro de distribución"
    prov = PROVEEDORES["Distribuidor regional"]
    catalogo = []
    for item, g in panel.groupby("item_nbr"):
        unidades = g["unit_sales"].clip(lower=0)
        perecible = int(g["perishable"].iloc[0]) == 1
        familia = str(g["family"].iloc[0])
        media = float(unidades.mean())
        catalogo.append({
            "sku": f"IT-{int(item)}",
            "item_nbr": int(item),
            "descripcion": f"{familia} {int(item)}",
            "familia": familia,
            "clase_item": int(g["class"].iloc[0]) if pd.notna(g["class"].iloc[0]) else 0,
            "perecible": perecible,
            "proveedor": proveedor,
            "tienda": int(g["store_nbr"].iloc[0]),
            "ciudad": str(g["city"].iloc[0]),
            "costo_unitario": COSTO_PERECIBLE if perecible else COSTO_SECO,
            "vida_util_dias": VIDA_PERECIBLE if perecible else VIDA_SECO,
            "tasa_promocion": float(g["onpromotion"].mean()),
            "unidades_observadas": float(unidades.sum()),
            "devoluciones": float((-g["unit_sales"].clip(upper=0)).sum()),
            "_demanda_media": max(media, 0.05),
            "_demanda_desv": float(unidades.std(ddof=1) or 0.0),
            "_estacionalidad": 0.0,
            "_fase_estacional": 0.0,
            "_lead_time_medio": prov["lead_time_medio_dias"],
            "_lead_time_desv": prov["lead_time_desv_dias"],
            "_prob_incumplimiento": prov["prob_incumplimiento"],
            "_fraccion_si_incumple": prov["fraccion_si_incumple"],
            "lote_minimo": prov["lote_minimo"],
        })
    return catalogo


def demanda_observada(panel, fechas):
    """
    Dict (sku, día) → unidades, con día = 1..n sobre `fechas`.

    Los días sin fila ya vienen en 0 en el panel. Las devoluciones no cuentan
    como demanda.
    """
    fechas = list(pd.to_datetime(fechas))
    indice = {d: i + 1 for i, d in enumerate(fechas)}
    sub = panel[panel["date"].isin(indice)].copy()
    sub["dia"] = sub["date"].map(indice)
    sub["sku"] = sub["item_nbr"].map(lambda n: f"IT-{int(n)}")
    sub["demanda"] = sub["unit_sales"].clip(lower=0)
    return {(sku, int(dia)): float(u) for sku, dia, u in
            zip(sub["sku"], sub["dia"], sub["demanda"])}


def resumen_fuente(panel):
    """Texto corto para la barra lateral y la pestaña de datos."""
    return {
        "tienda": int(panel["store_nbr"].iloc[0]),
        "ciudad": str(panel["city"].iloc[0]),
        "provincia": str(panel["state"].iloc[0]),
        "tipo": str(panel["store_type"].iloc[0]),
        "cluster": int(panel["cluster"].iloc[0]),
        "desde": panel["date"].min().date().isoformat(),
        "hasta": panel["date"].max().date().isoformat(),
        "items": int(panel["item_nbr"].nunique()),
        "dias": int(panel["date"].nunique()),
    }
