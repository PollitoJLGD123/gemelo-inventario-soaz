# modulo_simulacion/catalogo.py
"""
Catálogo de SKU de la Comercializadora SOAZ y clasificación ABC.

El catálogo lleva DOS conjuntos de parámetros por SKU y la distinción es
deliberada:

  parámetros VERDADEROS  — gobiernan cómo se comporta el negocio real. En el
                           proyecto real no existen: son la naturaleza. Aquí se
                           generan para poder VALIDAR el gemelo contra algo.
  parámetros ESTIMADOS   — los que el gemelo deduce del histórico de
                           movimientos. Son los únicos que el gemelo puede usar.

Mantenerlos separados es lo que permite responder la pregunta que ningún
proyecto de curso responde: ¿qué tan lejos está el gemelo de la realidad que
dice replicar? Si el motor pudiera leer los parámetros verdaderos, la
validación de fidelidad sería una tautología.
"""
import numpy as np

from .parametros import FAMILIAS, PROVEEDORES, CATALOGO, NEGOCIO


def _uniforme(rng, rango):
    return float(rng.uniform(rango[0], rango[1]))


def generar_catalogo(rng, n_sku=None):
    """
    Genera el catálogo con sus parámetros VERDADEROS (ocultos para el gemelo).

    En el proyecto real esta función se reemplaza por la lectura de la tabla
    de productos del sistema de inventarios. La firma se mantiene igual para
    que el cambio no toque nada más.
    """
    n_sku = n_sku or CATALOGO["n_sku_total"]
    familias = list(FAMILIAS.keys())
    proveedores = list(PROVEEDORES.keys())
    catalogo = []

    for i in range(n_sku):
        familia = str(rng.choice(familias))
        cfg = FAMILIAS[familia]
        proveedor = str(rng.choice(proveedores))
        prov = PROVEEDORES[proveedor]

        # La rotación real de un catálogo es muy desigual: unos pocos SKU
        # concentran el movimiento. Sin este sesgo la curva ABC sale plana y
        # la clase A se lleva medio catálogo, que no es lo que se observa en
        # una comercializadora.
        sesgo = float(rng.lognormal(mean=0.0, sigma=0.85))
        demanda_media = _uniforme(rng, cfg["demanda_diaria_media"]) * sesgo
        cv = _uniforme(rng, cfg["coef_variacion"])
        costo = _uniforme(rng, cfg["costo_unitario"])

        catalogo.append({
            "sku": f"SKU-{i + 1:03d}",
            "descripcion": f"{familia} {i + 1:03d}",
            "familia": familia,
            "proveedor": proveedor,
            "costo_unitario": round(costo, 2),
            "vida_util_dias": cfg["vida_util_dias"],
            # --- verdad oculta: el gemelo NO puede leer esto ---
            "_demanda_media": demanda_media,
            "_demanda_desv": demanda_media * cv,
            "_estacionalidad": cfg["estacionalidad"],
            "_fase_estacional": float(rng.uniform(0, 2 * np.pi)),
            "_lead_time_medio": prov["lead_time_medio_dias"],
            "_lead_time_desv": prov["lead_time_desv_dias"],
            "_prob_incumplimiento": prov["prob_incumplimiento"],
            "_fraccion_si_incumple": prov["fraccion_si_incumple"],
            "lote_minimo": prov["lote_minimo"],
        })
    return catalogo


def clasificar_abc(catalogo, umbral_a=None):
    """
    Clasificación ABC por valor de movimiento anual (demanda × costo unitario).

    Es la regla que delimita el alcance del proyecto: se modela la clase A.
    Se declara el criterio porque «los SKU más importantes» no es un criterio.
    """
    umbral_a = umbral_a if umbral_a is not None else CATALOGO["umbral_clase_a"]
    dias_anio = NEGOCIO["dias_habiles_semana"] * 52

    for s in catalogo:
        s["valor_anual"] = s["_demanda_media"] * dias_anio * s["costo_unitario"]

    ordenado = sorted(catalogo, key=lambda s: -s["valor_anual"])
    total = sum(s["valor_anual"] for s in ordenado) or 1.0

    acumulado = 0.0
    for s in ordenado:
        acumulado += s["valor_anual"]
        participacion = acumulado / total
        s["participacion_acumulada"] = participacion
        if participacion <= umbral_a:
            s["clase"] = "A"
        elif participacion <= 0.95:
            s["clase"] = "B"
        else:
            s["clase"] = "C"
    # El primero siempre es clase A aunque por sí solo supere el umbral
    ordenado[0]["clase"] = "A"
    return ordenado


def catalogo_en_alcance(catalogo):
    """Devuelve los SKU dentro del alcance declarado del proyecto."""
    if not CATALOGO["solo_clase_a"]:
        return catalogo
    return [s for s in catalogo if s["clase"] == "A"]


def resumen_abc(catalogo):
    """Resumen de la clasificación, para la tabla de alcance."""
    total_valor = sum(s["valor_anual"] for s in catalogo) or 1.0
    filas = []
    for clase in ("A", "B", "C"):
        items = [s for s in catalogo if s["clase"] == clase]
        if not items:
            continue
        valor = sum(s["valor_anual"] for s in items)
        filas.append({
            "Clase": clase,
            "SKU": len(items),
            "% del catálogo": round(len(items) / len(catalogo) * 100, 1),
            "Valor anual (S/.)": round(valor, 0),
            "% del movimiento": round(valor / total_valor * 100, 1),
            "En alcance": "Sí" if clase == "A" or not CATALOGO["solo_clase_a"] else "No",
        })
    return filas
