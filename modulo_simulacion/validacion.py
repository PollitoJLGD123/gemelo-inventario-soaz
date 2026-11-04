# modulo_simulacion/validacion.py
"""
Validación de fidelidad del gemelo — la retro-prueba.

POR QUÉ ESTE MÓDULO ES EL DIFERENCIAL DEL PROYECTO
---------------------------------------------------
Un gemelo cuyo parecido con el original nunca se midió es una animación. La
mayoría de los trabajos de curso construyen el simulador, muestran que
produce curvas bonitas y concluyen. Aquí se hace la pregunta incómoda: ¿el
gemelo reproduce el pasado que SÍ ocurrió?

EL PROTOCOLO, Y POR QUÉ CADA PASO ES COMO ES
---------------------------------------------
1. Se corta el histórico en D0. El gemelo estima sus parámetros usando
   ÚNICAMENTE los movimientos anteriores a D0. No puede ver el futuro.

2. Se le carga el stock real que había en D0 y se lo hace correr hacia
   adelante 90 días con la política que SOAZ REALMENTE APLICÓ en ese
   período —no con la política optimizada que el gemelo recomienda—. Este
   punto es el que más se equivoca: validar con la política mejorada no
   prueba fidelidad, prueba otra cosa.

3. Se compara la trayectoria simulada contra la registrada.

4. Los umbrales de aceptación se fijaron ANTES, en parametros.FIDELIDAD.
   Fijarlos después de ver el resultado convierte una validación en una
   justificación.

DOS MÉTRICAS, PORQUE UNA SOLA ENGAÑA
-------------------------------------
  MAPE de la media   — cuánto se aparta la trayectoria media del gemelo de la real.
  Cobertura de banda — en qué porcentaje de días la trayectoria real cae dentro
                       de la banda del 95% del gemelo.

La segunda es la correcta para un modelo estocástico: el gemelo no tiene por
qué acertar el valor exacto de un día, tiene que acertar el RANGO. Un modelo
con MAPE bajo y cobertura baja está sobreajustado; uno con cobertura alta y
bandas enormes no dice nada. Hay que mirar las dos.

SI EL GEMELO NO PASA
---------------------
Se reporta igual, con el diagnóstico. Un gemelo que falla la validación y
explica por qué es un resultado científico legítimo; uno que nunca se validó,
no lo es.
"""
import numpy as np
import pandas as pd

from .estimacion import construir_estado_gemelo
from .motor_des import simular
from .parametros import FIDELIDAD


def _stock_real_diario(movimientos, dia_inicio, dia_fin):
    """Stock de cierre de cada día por SKU, según el registro del sistema."""
    m = movimientos[(movimientos["dia"] >= dia_inicio) & (movimientos["dia"] <= dia_fin)]
    m = m.sort_values(["sku", "dia"])
    cierre = m.groupby(["sku", "dia"])["stock_resultante"].last().reset_index()
    return cierre.pivot(index="dia", columns="sku", values="stock_resultante")


def _ordenes_reales(movimientos, dia_inicio, dia_fin):
    m = movimientos[(movimientos["tipo"] == "ORDEN") &
                    (movimientos["dia"] >= dia_inicio) & (movimientos["dia"] <= dia_fin)]
    return m.groupby("sku").size().to_dict()


def _dias_ruptura_reales(movimientos, dia_inicio, dia_fin):
    ventas = movimientos[(movimientos["tipo"] == "VENTA") &
                         (movimientos["dia"] >= dia_inicio) & (movimientos["dia"] <= dia_fin)]
    return ventas[ventas["stock_resultante"] <= 0].groupby("sku").size().to_dict()


def pisos_irreducibles(estado, dias, n_pares=8, semilla_base=91000):
    """
    Piso de error que NINGÚN modelo puede bajar: el gemelo contra SÍ MISMO.

    Se corren pares de réplicas idénticas en todo salvo la semilla y se mide,
    entre ellas, exactamente las mismas métricas con que se compara el gemelo
    contra el registro real. El resultado es el ruido puro del proceso.

    POR QUÉ ESTO CAMBIA LA VALIDACIÓN ENTERA
    -----------------------------------------
    Los umbrales de fidelidad de este proyecto se fijaron primero a ojo: 20% de
    MAPE, 2 días de diferencia en rupturas. Al medir los pisos resultó que el
    piso del MAPE es ~57% y el de los días de ruptura ~4.6: los dos umbrales
    estaban POR DEBAJO del ruido irreducible, o sea que eran imposibles de
    cumplir para cualquier modelo, incluido uno perfecto.

    Un umbral que ni el modelo verdadero puede cumplir no es exigente: está
    mal formulado. Por eso las métricas que comparan trayectorias estocásticas
    se juzgan RELATIVAS a su piso, y no en valor absoluto.
    """
    # Piso de la cobertura: se construye una banda con un grupo de réplicas y
    # se mide qué porcentaje de días cubre a una réplica DEJADA FUERA. Aunque
    # la banda sea nominalmente del 95%, no cubre el 95% de los días: los
    # niveles de stock de días consecutivos están fuertemente autocorrelacionados,
    # así que el número efectivo de observaciones independientes es mucho menor
    # que el de días. Comparar contra el 95% teórico sería otro umbral imposible.
    grupo = [simular(estado, {"tipo_politica": "(R,s,Q)"}, dias=dias,
                     semilla=semilla_base + 20000 + i * 41) for i in range(15)]
    fuera = simular(estado, {"tipo_politica": "(R,s,Q)"}, dias=dias,
                    semilla=semilla_base + 40000)
    coberturas = []
    for k, s_fuera in enumerate(fuera["por_sku"]):
        matriz = np.array([[p["stock"] for p in rep["por_sku"][k]["serie"]] for rep in grupo])
        objetivo = np.array([p["stock"] for p in s_fuera["serie"]], dtype=float)
        n = min(matriz.shape[1], len(objetivo))
        lo = np.percentile(matriz[:, :n], 2.5, axis=0)
        hi = np.percentile(matriz[:, :n], 97.5, axis=0)
        coberturas.append(float(np.mean((objetivo[:n] >= lo) & (objetivo[:n] <= hi)) * 100))

    mapes, rupturas, ordenes, niveles = [], [], [], []
    for i in range(n_pares):
        a = simular(estado, {"tipo_politica": "(R,s,Q)"}, dias=dias,
                    semilla=semilla_base + i * 17)
        b = simular(estado, {"tipo_politica": "(R,s,Q)"}, dias=dias,
                    semilla=semilla_base + 5000 + i * 17)
        for sa, sb in zip(a["por_sku"], b["por_sku"]):
            x = np.array([p["stock"] for p in sa["serie"]], dtype=float)
            y = np.array([p["stock"] for p in sb["serie"]], dtype=float)
            n = min(len(x), len(y))
            escala = max(float(np.mean(y[:n])), 1.0)
            mapes.append(float(np.mean(np.abs(x[:n] - y[:n])) / escala * 100))
            niveles.append(float(abs(np.mean(x[:n]) - np.mean(y[:n])) / escala * 100))
            rupturas.append(abs(sa["dias_ruptura"] - sb["dias_ruptura"]))
            ordenes.append(abs(sa["ordenes"] - sb["ordenes"]) / max(sb["ordenes"], 1) * 100)
    return {
        "mape": float(np.mean(mapes)),
        "nivel": float(np.mean(niveles)),
        "dias_ruptura": float(np.mean(rupturas)),
        "error_ordenes": float(np.mean(ordenes)),
        "cobertura": float(np.mean(coberturas)),
        "n_pares": n_pares,
    }


def retro_prueba(catalogo, movimientos, dias_ajuste=None, dias_prueba=None,
                 n_replicas=20, metodo_demanda="em", semilla_base=5000):
    """
    Ejecuta la retro-prueba completa y devuelve el veredicto.

    El gemelo estima con los datos hasta `dias_ajuste`, corre `dias_prueba`
    días hacia adelante con la práctica real, y se compara con el registro.
    """
    dias_ajuste = dias_ajuste or FIDELIDAD["dias_ajuste"]
    dias_prueba = dias_prueba or FIDELIDAD["dias_prueba"]
    d0, d1 = dias_ajuste, dias_ajuste + dias_prueba

    # --- Paso 1: el gemelo solo ve el pasado ---
    estado = construir_estado_gemelo(catalogo, movimientos,
                                     metodo_demanda=metodo_demanda,
                                     dia_corte=d0)

    # --- Paso 2: correr hacia adelante con la política REAL de SOAZ ---
    replicas = [simular(estado, {"tipo_politica": "(R,s,Q)"}, dias=dias_prueba,
                        semilla=semilla_base + i * 733)
                for i in range(n_replicas)]

    # Trayectorias simuladas: [replica][sku][dia]
    trayectorias = {}
    for rep in replicas:
        for s in rep["por_sku"]:
            serie = [p["stock"] for p in s["serie"]]
            trayectorias.setdefault(s["sku"], []).append(serie)

    real = _stock_real_diario(movimientos, d0 + 1, d1)
    ordenes_reales = _ordenes_reales(movimientos, d0 + 1, d1)
    rupturas_reales = _dias_ruptura_reales(movimientos, d0 + 1, d1)

    ordenes_sim = {}
    rupturas_sim = {}
    for rep in replicas:
        for s in rep["por_sku"]:
            ordenes_sim.setdefault(s["sku"], []).append(s["ordenes"])
            rupturas_sim.setdefault(s["sku"], []).append(s["dias_ruptura"])

    filas = []
    for e in estado:
        sku = e["sku"]
        if sku not in trayectorias or sku not in real.columns:
            continue
        matriz = np.array(trayectorias[sku])                 # (replicas, dias)
        n_dias = min(matriz.shape[1], len(real.index))
        matriz = matriz[:, :n_dias]
        serie_real = real[sku].to_numpy(dtype=float)[:n_dias]

        media_sim = matriz.mean(axis=0)
        banda_baja = np.percentile(matriz, 2.5, axis=0)
        banda_alta = np.percentile(matriz, 97.5, axis=0)

        # MAPE relativo al nivel medio de stock del SKU: el MAPE clásico
        # explota cuando el stock real toca cero, que es justo lo que pasa
        # en un almacén. Se normaliza por el promedio, no por cada punto.
        escala = max(float(np.mean(serie_real)), 1.0)
        mape = float(np.mean(np.abs(media_sim - serie_real)) / escala * 100)
        cobertura = float(np.mean((serie_real >= banda_baja) & (serie_real <= banda_alta)) * 100)
        # Error del NIVEL: estadístico estable, inmune al desfase del diente
        error_nivel = float(abs(np.mean(media_sim) - np.mean(serie_real)) / escala * 100)

        ord_real = ordenes_reales.get(sku, 0)
        ord_sim = float(np.mean(ordenes_sim[sku]))
        err_ordenes = abs(ord_sim - ord_real) / max(ord_real, 1) * 100

        rup_real = rupturas_reales.get(sku, 0)
        rup_sim = float(np.mean(rupturas_sim[sku]))

        filas.append({
            "SKU": sku,
            "Clase": e["clase"],
            "Stock medio real (u)": round(float(np.mean(serie_real)), 1),
            "Stock medio simulado (u)": round(float(np.mean(media_sim)), 1),
            "Error nivel medio (%)": round(error_nivel, 1),
            "MAPE stock (%)": round(mape, 1),
            "Cobertura de banda (%)": round(cobertura, 1),
            "Días ruptura real": int(rup_real),
            "Días ruptura simulado": round(rup_sim, 1),
            "Δ días ruptura": round(abs(rup_sim - rup_real), 1),
            "Órdenes reales": int(ord_real),
            "Órdenes simuladas": round(ord_sim, 1),
            "Error órdenes (%)": round(err_ordenes, 1),
            "_media_sim": media_sim.tolist(),
            "_banda_baja": banda_baja.tolist(),
            "_banda_alta": banda_alta.tolist(),
            "_real": serie_real.tolist(),
        })

    clase_a = [f for f in filas if f["Clase"] == "A"] or filas
    pisos = pisos_irreducibles(estado, dias_prueba)
    piso = pisos["mape"]
    mape_medio = float(np.mean([f["MAPE stock (%)"] for f in clase_a]))
    delta_rup = float(np.mean([f["Δ días ruptura"] for f in clase_a]))
    err_ord = float(np.mean([f["Error órdenes (%)"] for f in clase_a]))

    resumen = {
        "dia_corte": d0,
        "dias_prueba": dias_prueba,
        "n_replicas": n_replicas,
        "n_sku": len(filas),
        "metodo_demanda": metodo_demanda,
        "error_nivel": float(np.mean([f["Error nivel medio (%)"] for f in clase_a])),
        "mape_stock": mape_medio,
        "mape_irreducible": piso,
        "mape_relativo": mape_medio / max(piso, 1e-9),
        "pisos": pisos,
        "ruptura_relativa": delta_rup / max(pisos["dias_ruptura"], 1e-9),
        "cobertura_relativa": (float(np.mean([f["Cobertura de banda (%)"] for f in clase_a]))
                               / max(pisos["cobertura"], 1e-9)),
        "ordenes_relativo": err_ord / max(pisos["error_ordenes"], 1e-9),
        "cobertura_banda": float(np.mean([f["Cobertura de banda (%)"] for f in clase_a])),
        "delta_dias_ruptura": float(np.mean([f["Δ días ruptura"] for f in clase_a])),
        "error_ordenes": float(np.mean([f["Error órdenes (%)"] for f in clase_a])),
    }

    criterios = [
        {
            "Criterio": "Nivel medio de stock",
            "Métrica": "Error del stock medio, clase A",
            "Obtenido": round(resumen["error_nivel"], 1),
            "Umbral": FIDELIDAD["umbral_error_nivel"],
            "Cumple": resumen["error_nivel"] <= FIDELIDAD["umbral_error_nivel"],
            "unidad": "%",
            "Por qué esta métrica": "El nivel medio es estable frente al desfase de las "
                                    "llegadas; mide si el gemelo mantiene tanto inventario "
                                    "como el negocio real.",
        },
        {
            "Criterio": "Cobertura, relativa al techo",
            "Métrica": f"Cobertura del gemelo / cobertura alcanzable ({pisos['cobertura']:.0f}%)",
            "Obtenido": round(resumen["cobertura_relativa"], 2),
            "Umbral": FIDELIDAD["umbral_cobertura_relativa"],
            "Cumple": resumen["cobertura_relativa"] >= FIDELIDAD["umbral_cobertura_relativa"],
            "unidad": "×",
            "Por qué esta métrica": (
                f"Un modelo estocástico no debe acertar el valor de un día, debe acertar "
                f"el RANGO. Pero una banda del 95% no cubre el 95% de los días: los "
                f"niveles de stock de días consecutivos están autocorrelacionados, así "
                f"que el techo alcanzable es {pisos['cobertura']:.0f}%, no 95%."),
            "mayor_es_mejor": True,
            "absoluto": f"{resumen['cobertura_banda']:.1f}%",
        },
        {
            "Criterio": "MAPE relativo al piso",
            "Métrica": f"MAPE del gemelo / MAPE irreducible ({piso:.0f}%)",
            "Obtenido": round(resumen["mape_relativo"], 2),
            "Umbral": FIDELIDAD["umbral_mape_relativo"],
            "Cumple": resumen["mape_relativo"] <= FIDELIDAD["umbral_mape_relativo"],
            "unidad": "×",
            "Por qué esta métrica": "El MAPE absoluto no se puede interpretar en un proceso "
                                    "de diente de sierra. Contra el piso del propio motor, sí.",
        },
        {
            "Criterio": "Días de ruptura, relativo al piso",
            "Métrica": f"Δ días del gemelo / Δ días irreducible ({pisos['dias_ruptura']:.1f} d)",
            "Obtenido": round(resumen["ruptura_relativa"], 2),
            "Umbral": FIDELIDAD["umbral_ruptura_relativa"],
            "Cumple": resumen["ruptura_relativa"] <= FIDELIDAD["umbral_ruptura_relativa"],
            "unidad": "×",
            "Por qué esta métrica": f"El conteo de rupturas en 90 días es ruidoso: dos "
                                    f"réplicas del propio gemelo difieren en "
                                    f"{pisos['dias_ruptura']:.1f} días en promedio. "
                                    f"Exigir menos que eso es exigir lo imposible.",
            "absoluto": f"{resumen['delta_dias_ruptura']:.1f} días",
        },
        {
            "Criterio": "Órdenes emitidas",
            "Métrica": "Error relativo medio",
            "Obtenido": round(resumen["error_ordenes"], 1),
            "Umbral": FIDELIDAD["umbral_error_pedidos"],
            "Cumple": resumen["error_ordenes"] <= FIDELIDAD["umbral_error_pedidos"],
            "unidad": "%",
            "Por qué esta métrica": "Verifica que el gemelo reproduce la REGLA DE DECISIÓN "
                                    "del negocio, no solo la física del almacén.",
        },
    ]
    resumen["criterios"] = criterios
    resumen["veredicto"] = all(c["Cumple"] for c in criterios)
    resumen["n_criterios_cumplidos"] = sum(1 for c in criterios if c["Cumple"])
    resumen["n_criterios"] = len(criterios)
    return {"resumen": resumen, "por_sku": filas}


def comparar_fidelidad_por_metodo(catalogo, movimientos, metodos=("ingenuo", "em"),
                                  n_replicas=12):
    """
    Corre la retro-prueba con distintos estimadores de demanda.

    Muestra que la corrección por censura no es un adorno metodológico: mejora
    la fidelidad medida del gemelo. Sin esta comparación, usar EM es una
    preferencia; con ella, es una decisión justificada.
    """
    filas = []
    for m in metodos:
        r = retro_prueba(catalogo, movimientos, metodo_demanda=m, n_replicas=n_replicas)
        s = r["resumen"]
        filas.append({
            "Estimador de demanda": m,
            "Error nivel (%)": round(s["error_nivel"], 1),
            "Cobertura de banda (%)": round(s["cobertura_banda"], 1),
            "MAPE relativo (×)": round(s["mape_relativo"], 2),
            "Δ días ruptura": round(s["delta_dias_ruptura"], 1),
            "Error órdenes (%)": round(s["error_ordenes"], 1),
            "Criterios cumplidos": f"{s['n_criterios_cumplidos']}/{s['n_criterios']}",
        })
    return filas
