# modulo_simulacion/escenarios.py
"""
Ejecución de escenarios con réplicas e intervalos de confianza.

TODA CIFRA VIENE CON INCERTIDUMBRE
-----------------------------------
El motor es estocástico: la demanda y el tiempo de entrega se sortean. Una
sola corrida es una muestra de tamaño 1, y presentar su resultado como si
fuera un hecho es el error que más rápido detecta un jurado que sabe de
simulación. Cada escenario se corre N veces con semillas distintas y se
reporta media e intervalo de confianza.

De ahí sale la columna que decide la comparación entre políticas: no la
diferencia de costo, sino si esa diferencia SOBREVIVE al ruido del modelo.
Sin esa columna, cualquier política parece funcionar.
"""
import numpy as np
from scipy import stats

from .motor_des import simular, gasto_reposicion_mensual
from .parametros import ESCENARIOS, EJECUCION, NEGOCIO


def _ic(muestras, confianza=None):
    """Media, intervalo de confianza por t de Student y desviación."""
    confianza = confianza or EJECUCION["nivel_confianza"]
    a = np.asarray(muestras, dtype=float)
    media = float(a.mean())
    if a.size < 2:
        return media, media, media, 0.0
    margen = float(stats.sem(a) * stats.t.ppf((1 + confianza) / 2.0, a.size - 1))
    return media, media - margen, media + margen, float(a.std(ddof=1))


def ejecutar_escenario(estado, codigo="Base", dias=None, n_replicas=None,
                       semilla_base=None, cfg=None):
    """
    Corre un escenario con N réplicas y devuelve agregados con incertidumbre.

    `cfg` permite correr un escenario que NO está declarado en ESCENARIOS: se
    le pasa directamente el diccionario {nombre, pregunta, palancas, mecanismo}
    y se saltea la búsqueda por código. Existe porque hay escenarios que nacen
    de una pregunta del negocio en el momento, no de la lista fija.

    Sin ese parámetro, un código desconocido caía en ESCENARIOS["Base"] EN
    SILENCIO: se preguntaba por un escenario y se respondía con otro, sin que
    nada en la salida lo delatara. Un escenario ad-hoc siempre habría dado el
    caso base disfrazado de respuesta.
    """
    dias = dias or EJECUCION["dias_simulacion"]
    n_replicas = n_replicas or EJECUCION["n_replicas"]
    semilla_base = semilla_base if semilla_base is not None else EJECUCION["semilla_base"]

    if cfg is not None:
        # Un escenario ad-hoc no tiene código en la lista fija; se etiqueta para
        # que la salida nunca se confunda con la de un escenario declarado.
        codigo = cfg.get("codigo", "Ad-hoc")
    else:
        cfg = ESCENARIOS.get(codigo, ESCENARIOS["Base"])
    palancas = cfg["palancas"]

    metricas = {k: [] for k in ("fill_rate", "costo_total", "costo_ordenar",
                                "costo_mantener", "costo_ruptura", "costo_merma",
                                "capital_inmovilizado", "ordenes", "unidades_perdidas",
                                "unidades_merma", "sku_con_ruptura")}
    por_sku_acumulado = {}
    ultima = None

    for r in range(n_replicas):
        res = simular(estado, palancas, dias=dias, semilla=semilla_base + r * 1009)
        for k in metricas:
            metricas[k].append(res[k])
        for s in res["por_sku"]:
            d = por_sku_acumulado.setdefault(s["sku"], {"fill": [], "costo": [],
                                                        "ruptura": [], "merma": [],
                                                        "capital": [], "ordenes": []})
            d["fill"].append(s["fill_rate"])
            d["costo"].append(s["costo_total"])
            d["ruptura"].append(s["dias_ruptura"])
            d["merma"].append(s["merma"])
            d["capital"].append(s["capital_inmovilizado"])
            d["ordenes"].append(s["ordenes"])
        ultima = res

    salida = {
        "codigo": codigo,
        "nombre": cfg["nombre"],
        "pregunta": cfg["pregunta"],
        "mecanismo": cfg["mecanismo"],
        "dias": dias,
        "n_replicas": n_replicas,
        "n_sku": ultima["n_sku"],
        "politica": ultima["politica"],
        "_replica_muestra": ultima,
    }
    for k, v in metricas.items():
        media, bajo, alto, sd = _ic(v)
        salida[k] = media
        salida[f"{k}_ic"] = (bajo, alto)
        salida[f"{k}_sd"] = sd

    # Tabla por SKU con la media de las réplicas
    filas = []
    mapa = {s["sku"]: s for s in ultima["por_sku"]}
    for sku, d in por_sku_acumulado.items():
        ref = mapa[sku]
        filas.append({
            "SKU": sku,
            "Familia": ref["familia"],
            "Fill rate (%)": round(float(np.mean(d["fill"])), 2),
            "Días con ruptura": round(float(np.mean(d["ruptura"])), 1),
            "Órdenes": round(float(np.mean(d["ordenes"])), 1),
            "Merma (u)": round(float(np.mean(d["merma"])), 1),
            "Capital inmovilizado (S/.)": round(float(np.mean(d["capital"])), 0),
            "Costo total (S/.)": round(float(np.mean(d["costo"])), 0),
        })
    salida["por_sku"] = sorted(filas, key=lambda f: -f["Costo total (S/.)"])
    return salida


def distinguible_del_base(resultado, base):
    """
    ¿La diferencia de costo sobrevive al ruido del modelo?

    Los intervalos de confianza tienen que separarse: si se solapan, la
    diferencia observada es compatible con el azar de las réplicas y no se
    puede afirmar nada.

    Vive aquí, y no repetida en cada pantalla, para que no lleguen a existir
    dos definiciones de «distinguible» que digan cosas distintas.
    """
    if resultado is None or base is None:
        return False
    return (resultado["costo_total_ic"][0] > base["costo_total_ic"][1] or
            resultado["costo_total_ic"][1] < base["costo_total_ic"][0])


def fila_comparativa(resultado, base, meses):
    """
    Una fila de la tabla que compara un escenario contra el Base.

    Vive aquí porque la usan dos pantallas — la de escenarios declarados y la
    del copiloto — y las columnas tienen que ser las mismas en las dos. Si
    mañana se agrega una métrica, aparece en ambas o en ninguna.
    """
    es_base = resultado is base
    delta_costo = resultado["costo_total"] - base["costo_total"]
    delta_fill = resultado["fill_rate"] - base["fill_rate"]
    return {
        "Código": resultado["codigo"],
        "Escenario": resultado["nombre"],
        "Fill rate (%)": round(resultado["fill_rate"], 2),
        "Δ fill vs Base (pp)": round(delta_fill, 2),
        "Costo total (S/.)": round(resultado["costo_total"], 0),
        "IC 95% costo": (f"[{resultado['costo_total_ic'][0]:,.0f}, "
                         f"{resultado['costo_total_ic'][1]:,.0f}]"),
        "Δ costo vs Base (S/.)": round(delta_costo, 0),
        "¿Distinguible del Base?": (
            "—" if es_base else ("Sí" if distinguible_del_base(resultado, base) else "No")),
        "Capital inmovilizado (S/.)": round(resultado["capital_inmovilizado"], 0),
        "Órdenes": round(resultado["ordenes"], 1),
        "Costo mensual (S/.)": round(resultado["costo_total"] / max(meses, 1e-9), 0),
    }


def comparar_escenarios(estado, codigos=None, dias=None, n_replicas=None):
    """
    Compara escenarios contra el Base.

    La columna «¿Distinguible del Base?» responde si el intervalo del escenario
    se separa del intervalo del Base. Es la que convierte una tabla de números
    en una tabla de conclusiones.
    """
    codigos = codigos or list(ESCENARIOS.keys())
    dias = dias or EJECUCION["dias_simulacion"]
    n_replicas = n_replicas or EJECUCION["n_replicas"]

    resultados = {c: ejecutar_escenario(estado, c, dias, n_replicas) for c in codigos}
    base = resultados.get("Base")
    meses = dias / NEGOCIO["dias_por_mes"]

    filas = [fila_comparativa(r, base, meses) for r in resultados.values()]
    return {"tabla": filas, "resultados": resultados, "meses": meses}


def comparar_politicas(estado, dias=None, n_replicas=None,
                       niveles=(0.90, 0.95, 0.975, 0.99)):
    """
    Curva de intercambio entre nivel de servicio y costo.

    Responde la pregunta que el negocio hace de verdad: "¿cuánto me cuesta
    prometer más?" — y muestra que la respuesta no siempre es "más caro",
    porque a niveles bajos el costo de la ruptura supera al de mantener.
    """
    dias = dias or EJECUCION["dias_simulacion"]
    n_replicas = n_replicas or max(8, (n_replicas or EJECUCION["n_replicas"]) // 2)
    filas = []
    for nivel in niveles:
        for tipo in ("(s,Q)", "(R,S)"):
            corridas = [simular(estado, {"nivel_servicio": nivel, "tipo_politica": tipo},
                                dias=dias, semilla=EJECUCION["semilla_base"] + i * 1009)
                        for i in range(n_replicas)]
            fill, f_lo, f_hi, _ = _ic([c["fill_rate"] for c in corridas])
            costo, c_lo, c_hi, _ = _ic([c["costo_total"] for c in corridas])
            capital, _, _, _ = _ic([c["capital_inmovilizado"] for c in corridas])
            ordenes, _, _, _ = _ic([c["ordenes"] for c in corridas])
            filas.append({
                "Política": tipo,
                "Nivel de servicio objetivo": f"{nivel * 100:.1f}%",
                "nivel": nivel,
                "Fill rate logrado (%)": round(fill, 2),
                "Costo total (S/.)": round(costo, 0),
                "costo_lo": c_lo, "costo_hi": c_hi,
                "Capital inmovilizado (S/.)": round(capital, 0),
                "Órdenes": round(ordenes, 1),
            })
    return filas


def recomendaciones_reposicion(estado, escenario="Base", dias=None, semilla=None):
    """
    Lazo de retorno del gemelo hacia el sistema real.

    Esto es lo que separa un gemelo de una sombra digital: el resultado no se
    queda en la pantalla, vuelve al proceso como una orden concreta que el
    encargado aprueba o rechaza. Cada fila lleva la razón que la sustenta,
    porque una recomendación sin razón no se aprueba, se ignora.
    """
    from .politicas import construir_politica

    dias = dias or 30
    semilla = semilla if semilla is not None else EJECUCION["semilla_base"]
    res = simular(estado, ESCENARIOS.get(escenario, ESCENARIOS["Base"])["palancas"],
                  dias=dias, semilla=semilla)
    por_sku = {s["sku"]: s for s in res["por_sku"]}

    filas = []
    for e in estado:
        pol = construir_politica(e)
        r = por_sku[e["sku"]]
        posicion_actual = e["stock_inicial"]
        cobertura_dias = posicion_actual / max(e["demanda_media"], 1e-9)

        if posicion_actual <= pol["punto_reorden"]:
            urgencia = ("Inmediata" if cobertura_dias < e["lead_time_medio"]
                        else "Esta semana")
            cantidad = int(np.ceil(pol["cantidad_pedido"]))
            razon = (f"La posición ({posicion_actual:.0f} u) está bajo el punto de "
                     f"reorden ({pol['punto_reorden']:.0f} u). Cubre "
                     f"{cobertura_dias:.1f} días y el proveedor tarda "
                     f"{e['lead_time_medio']:.1f} ± {e['lead_time_desv']:.1f} días.")
        else:
            urgencia = "No requiere"
            cantidad = 0
            razon = (f"La posición ({posicion_actual:.0f} u) supera el punto de reorden "
                     f"({pol['punto_reorden']:.0f} u): cubre {cobertura_dias:.1f} días.")

        filas.append({
            "SKU": e["sku"],
            "Familia": e["familia"],
            "Proveedor": e["proveedor"],
            "Stock actual (u)": int(posicion_actual),
            "Punto de reorden (u)": int(round(pol["punto_reorden"])),
            "Stock de seguridad (u)": int(round(pol["stock_seguridad"])),
            "Cantidad sugerida (u)": cantidad,
            "Valor de la orden (S/.)": round(cantidad * e["costo_unitario"], 2),
            "Urgencia": urgencia,
            "Cobertura actual (días)": round(cobertura_dias, 1),
            "Fill rate proyectado (%)": round(r["fill_rate"], 1),
            "Por qué": razon,
        })
    orden_urgencia = {"Inmediata": 0, "Esta semana": 1, "No requiere": 2}
    return sorted(filas, key=lambda f: (orden_urgencia[f["Urgencia"]],
                                        -f["Valor de la orden (S/.)"]))
