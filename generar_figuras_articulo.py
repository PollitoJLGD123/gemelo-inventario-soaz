#!/usr/bin/env python3
# generar_figuras_articulo.py
"""
Regenera las 4 figuras del artículo (PARIMANGO_GUTIERREZ_ARTICULO) con el
motor del gemelo, en lugar de dibujarlas a mano.

Uso:
    python generar_figuras_articulo.py --fuente favorita   # ventas reales, tienda 44
    python generar_figuras_articulo.py --fuente sintetico  # mundo de verificar_motor.py
    python generar_figuras_articulo.py --fuente favorita --solo 1 3

Salida: outputs/figures/articulo/<fuente>/fig{1..4}.png + datos_figuras.json
Cada número del gráfico sale de la corrida; nada se escribe a mano.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modulo_simulacion.catalogo import generar_catalogo, clasificar_abc, catalogo_en_alcance
from modulo_simulacion.datos_historicos import generar_historico
from modulo_simulacion.estimacion import construir_estado_gemelo, comparar_estimadores
from modulo_simulacion.politicas import construir_politica
from modulo_simulacion.motor_des import simular
from modulo_simulacion.validacion import retro_prueba
from modulo_simulacion.explicabilidad import explicar_sku, fidelidad_sustituto
from modulo_simulacion.parametros import EJECUCION, NIVELES_SERVICIO, FIDELIDAD

N_REPLICAS = 50
AZUL, ROJO, CELESTE, BANDA = "#0B4F9C", "#D64545", "#9ECAE1", "#C9D6E8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Traducción de factores de Shapley y familias para un artículo en inglés
FACTOR_EN = {
    "demanda_media": "Mean demand", "demanda_desv": r"Demand variability ($\sigma_d$)",
    "lead_time_medio": "Mean lead time ($L$)",
    "lead_time_desv": r"Lead-time variability ($\sigma_L$)",
    "costo_unitario": "Unit cost", "costo_pedido": "Ordering cost",
    "nivel_servicio_z": "Service level ($z$)", "cantidad_pedido": "Lot size ($Q$)",
    "vida_util": "Shelf life",
}
FAMILIA_EN = {
    "Abarrote seco": "Groceries", "Bebida": "Beverages",
    "Lácteo y perecible": "Dairy and\nperishables", "Limpieza y hogar": "Cleaning",
    "Cuidado personal": "Personal\ncare",
}


# --------------------------------------------------------------------------
def mundo(fuente):
    """Mismo mundo que la app (favorita) o que verificar_motor.py (sintetico)."""
    if fuente == "sintetico":
        rng = np.random.default_rng(2026)
        catalogo = catalogo_en_alcance(clasificar_abc(generar_catalogo(rng)))
        movimientos, verdad = generar_historico(catalogo, 540, 2026)
    else:
        from modulo_simulacion.favorita import (
            cargar_alcance, catalogo_desde_panel, demanda_observada)
        panel = cargar_alcance()
        catalogo = catalogo_en_alcance(clasificar_abc(catalogo_desde_panel(panel)))
        items = {int(s["item_nbr"]) for s in catalogo}
        fechas = sorted(pd.to_datetime(panel["date"].unique()))
        dias = EJECUCION["dias_historico"]
        if len(fechas) > dias:
            fechas = fechas[-int(dias):]
        ventana = panel[panel["item_nbr"].isin(items) & panel["date"].isin(fechas)]
        movimientos, verdad = generar_historico(
            catalogo, semilla=EJECUCION["semilla_base"],
            demanda_observada=demanda_observada(ventana, fechas))
    estado = construir_estado_gemelo(catalogo, movimientos)
    return catalogo, movimientos, verdad, estado


def _etiqueta_barra(ax, x, y, texto, arriba=True, desp=0.6):
    ax.text(x, y + (desp if arriba else -desp), texto, ha="center",
            va="bottom" if arriba else "top", fontsize=10)


# --------------------------------------------------------------------------
def figura_1(catalogo, movimientos, verdad, estado, carpeta):
    comp = comparar_estimadores(catalogo, movimientos, verdad)
    r = comp["resumen"]
    perdida = float(verdad["demanda_perdida"].sum() / max(verdad["demanda_real"].sum(), 1) * 100)

    # (b) subestimación del SS simplificado, misma fórmula que verificar_motor.py
    razones = {}
    for e in estado:
        p = construir_politica(e)
        if p["ss_simplificado"] > 0:
            razones.setdefault(e["familia"], []).append(p["stock_seguridad"] / p["ss_simplificado"])
    sub = {f: (1 - 1 / float(np.mean(v))) * 100 for f, v in razones.items()}
    todas = [x for v in razones.values() for x in v]
    sub_total = (1 - 1 / float(np.mean(todas))) * 100

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.1), gridspec_kw={"width_ratios": [1, 1.25]})
    metodos = [("ingenuo", "Naive\naverage"), ("exclusion", "Excluding\nstock-out days"),
               ("em", "EM censored\nnormal")]
    x = np.arange(len(metodos))
    w = 0.36
    for i, (m, _) in enumerate(metodos):
        mb, wb = r[m]["sesgo_medio"], r[m]["peor_sesgo"]
        a.bar(i - w / 2, mb, w, color=AZUL, label="Mean bias" if i == 0 else None)
        a.bar(i + w / 2, wb, w, color=ROJO, label="Worst SKU" if i == 0 else None)
        _etiqueta_barra(a, i - w / 2, mb, f"{mb:+.1f}".replace("-", "−"), mb >= 0)
        _etiqueta_barra(a, i + w / 2, wb, f"{wb:+.1f}".replace("-", "−"), wb >= 0)
    a.axhline(0, color="black", lw=1)
    a.set_xticks(x, [n for _, n in metodos])
    a.set_ylabel("Bias relative to latent demand (%)")
    a.set_title("(a) Demand estimators", loc="left", fontweight="bold")
    a.legend(loc="lower right", frameon=False)
    a.grid(axis="x", visible=False)
    lo = min(min(r[m]["peor_sesgo"] for m, _ in metodos), 0)
    a.set_ylim(lo * 1.18 - 1, max(3, max(r[m]["sesgo_medio"] for m, _ in metodos) + 3))

    fams = [f for f in FAMILIA_EN if f in sub] + [f for f in sub if f not in FAMILIA_EN]
    vals = [sub[f] for f in fams] + [sub_total]
    nombres = [FAMILIA_EN.get(f, f.title()[:18]) for f in fams] + ["All class-A\nSKUs"]
    colores = [CELESTE] * len(fams) + [AZUL]
    b.bar(range(len(vals)), vals, 0.72, color=colores)
    for i, v in enumerate(vals):
        _etiqueta_barra(b, i, v, f"{v:.0f} %", True, 1.0)
    b.set_xticks(range(len(vals)), nombres, fontsize=9 if len(vals) <= 7 else 7,
                 rotation=0 if len(vals) <= 7 else 35, ha="center" if len(vals) <= 7 else "right")
    b.set_ylabel("Underestimation of safety stock (%)")
    b.set_ylim(0, max(vals) * 1.18 + 2)
    b.set_title("(b) Simplified safety-stock formula", loc="left", fontweight="bold")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(carpeta / "fig1_estimadores_y_ss.png")
    plt.close(fig)
    return {"demanda_no_registrada_pct": perdida, "estimadores": r,
            "subestimacion_ss_por_familia_pct": sub, "subestimacion_ss_total_pct": sub_total,
            "n_sku_por_familia": {f: len(v) for f, v in razones.items()}}


# --------------------------------------------------------------------------
def figura_2(catalogo, movimientos, carpeta):
    val = retro_prueba(catalogo, movimientos, n_replicas=N_REPLICAS)
    filas = val["por_sku"]
    # SKU representativo: el de error de nivel medio más cercano a la mediana
    med = float(np.median([f["Error nivel medio (%)"] for f in filas]))
    f = min(filas, key=lambda f: abs(f["Error nivel medio (%)"] - med))
    x = np.arange(1, len(f["_real"]) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.fill_between(x, f["_banda_baja"], f["_banda_alta"], color=BANDA, alpha=0.8,
                    label=f"95 % prediction band (twin, N = {N_REPLICAS} replications)")
    ax.plot(x, f["_media_sim"], color=AZUL, lw=2, label="Twin mean")
    ax.plot(x, f["_real"], color=ROJO, lw=1.5, label="Recorded stock (unseen segment)")
    ax.set_xlabel(f"Day of the validation horizon (after cut-off D$_0$ = {val['resumen']['dia_corte']})")
    ax.set_ylabel("Stock on hand (units)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(carpeta / "fig2_retroprueba.png")
    plt.close(fig)
    resumen = {k: v for k, v in val["resumen"].items() if k != "criterios"}
    criterios = [{k: v for k, v in c.items() if not str(k).startswith("_")}
                 for c in val["resumen"].get("criterios", [])]
    return {"sku_mostrado": f["SKU"],
            "metricas_sku": {k: v for k, v in f.items() if not k.startswith("_")},
            "resumen": resumen, "criterios": criterios}


# --------------------------------------------------------------------------
def figura_3(estado, carpeta):
    niveles = sorted(NIVELES_SERVICIO)   # el motor solo tabula estos z
    dias = EJECUCION["dias_simulacion"]
    puntos = []
    for nivel in niveles:
        corr = [simular(estado, {"nivel_servicio": nivel}, dias=dias,
                        semilla=EJECUCION["semilla_base"] + i * 1009)
                for i in range(N_REPLICAS)]
        c = np.array([k["costo_total"] for k in corr]) / 1000
        fr = np.array([k["fill_rate"] for k in corr])
        h = 1.96 * c.std(ddof=1) / np.sqrt(len(c))
        puntos.append({"nivel": nivel, "costo_miles": float(c.mean()), "ic95": [float(c.mean() - h), float(c.mean() + h)],
                       "fill_rate": float(fr.mean())})

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    xs = [p["nivel"] * 100 for p in puntos]
    ys = [p["costo_miles"] for p in puntos]
    err = [[p["costo_miles"] - p["ic95"][0] for p in puntos], [p["ic95"][1] - p["costo_miles"] for p in puntos]]
    ax.errorbar(xs, ys, yerr=err, color=AZUL, lw=2.2, marker="o", ms=7, capsize=4)
    rango = max(ys) - min(ys)
    for xv, yv in zip(xs, ys):
        ax.text(xv, yv + max(rango * 0.08, 0.05), f"{yv:.1f}", ha="center", va="bottom")
    ax.set_xlabel("Target service level (%)")
    ax.set_ylabel(f"Expected total cost\n(thousand PEN per {dias} days)")
    ax.set_ylim(min(ys) - max(rango * 0.35, 0.3), max(ys) + max(rango * 0.35, 0.3))
    ax.set_xlim(88.5, 100.5)
    fig.tight_layout()
    fig.savefig(carpeta / "fig3_frontera_servicio_costo.png")
    plt.close(fig)
    return {"puntos": puntos, "politica": "(s,Q) por defecto del motor", "dias": dias}


# --------------------------------------------------------------------------
def figura_4(estado, carpeta):
    perecibles = [e for e in estado if e.get("vida_util_dias", 999) <= 30] or estado
    # SKU perecible representativo: el de mayor costo según el sustituto
    explicaciones = [(e, explicar_sku(e, estado)) for e in perecibles]
    e, ex = max(explicaciones, key=lambda t: t[1]["valor_sku"])
    fid = fidelidad_sustituto(estado, n_replicas=8)

    contrib = sorted(((k, v / 1000) for k, v in ex["contribuciones"].items()), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    nombres = [FACTOR_EN.get(k, k) for k, _ in contrib]
    vals = [v for _, v in contrib]
    ax.barh(nombres, vals, color=[ROJO if v > 0 else AZUL for v in vals], height=0.7)
    ax.axvline(0, color="black", lw=1)
    span = max(abs(v) for v in vals)
    for i, v in enumerate(vals):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), i,
                f"{v:+.2f}".replace("-", "−"), va="center", ha="left" if v >= 0 else "right")
    ax.set_xlim(min(min(vals), 0) - span * 0.25, max(max(vals), 0) + span * 0.2)
    ax.set_xlabel("Shapley contribution to annual cost (thousand PEN)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(carpeta / "fig4_shapley.png")
    plt.close(fig)
    return {"sku": ex["sku"], "familia": e["familia"], "costo_sku": ex["valor_sku"],
            "costo_referencia": ex["valor_base"], "error_eficiencia": ex["error_eficiencia"],
            "n_coaliciones": ex["n_coaliciones"],
            "contribuciones": ex["contribuciones"],
            "fidelidad_sustituto": {"error_absoluto_medio": fid["error_absoluto_medio"],
                                    "correlacion": fid["correlacion"], "apto": fid["apto"]}}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", choices=["favorita", "sintetico"], default="favorita")
    ap.add_argument("--solo", nargs="*", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()

    carpeta = Path(__file__).resolve().parent / "outputs" / "figures" / "articulo" / args.fuente
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / "datos_figuras.json"
    datos = json.loads(ruta_json.read_text(encoding="utf-8")) if ruta_json.exists() else {}

    t = time.time()
    catalogo, movimientos, verdad, estado = mundo(args.fuente)
    datos["fuente"] = args.fuente
    datos["n_sku_alcance"] = len(estado)
    print(f"[{args.fuente}] mundo listo: {len(estado)} SKU ({time.time() - t:.0f}s)", flush=True)

    tareas = {1: lambda: figura_1(catalogo, movimientos, verdad, estado, carpeta),
              2: lambda: figura_2(catalogo, movimientos, carpeta),
              3: lambda: figura_3(estado, carpeta),
              4: lambda: figura_4(estado, carpeta)}
    for n in args.solo:
        t = time.time()
        datos[f"fig{n}"] = tareas[n]()
        ruta_json.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=float),
                             encoding="utf-8")
        print(f"  figura {n} lista ({time.time() - t:.0f}s)", flush=True)
    print(f"Salida: {carpeta}")


if __name__ == "__main__":
    main()
