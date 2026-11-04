# modulo_simulacion/estimacion.py
"""
Estimación de los parámetros del gemelo a partir del histórico de movimientos.

Este módulo es la frontera del gemelo: todo lo que entra al motor pasa por
aquí, y aquí solo se puede leer lo que el sistema transaccional registra.
Los parámetros verdaderos del catálogo son inaccesibles por diseño.

EL PROBLEMA CENTRAL: DEMANDA CENSURADA
---------------------------------------
Las ventas registradas no son la demanda. Los días en que el producto se
agotó, el cliente que no fue atendido no dejó registro. Ajustar la
distribución sobre las ventas crudas **subestima sistemáticamente la
demanda**, el gemelo calcula un stock de seguridad demasiado bajo y termina
recomendando comprar de menos — es decir, reproduce el problema que vino a
resolver. Es el error que más silenciosamente arruina un proyecto de
inventarios, porque el modelo se ve perfectamente sano.

Se implementan tres estimadores para poder medir el efecto:

  ingenuo   — media y desviación de todas las ventas. Es lo que hace el 90%
              de los trabajos, y es el sesgado.
  exclusion — descarta los días con ruptura. Reduce el sesgo pero también
              descarta justo los días de demanda alta, así que sigue sesgado.
  em        — trata los días con ruptura como observaciones censuradas por la
              derecha (se sabe que D >= venta) y estima por
              Esperanza-Maximización sobre una normal truncada.

La comparación de los tres, contra la verdad conocida, es una figura que
justifica sola una sección del informe.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .parametros import NIVELES_SERVICIO


# --------------------------------------------------------------------------
# Detección de censura sin mirar la verdad
# --------------------------------------------------------------------------
def marcar_censura(movimientos):
    """
    Marca qué días de venta estuvieron censurados, usando SOLO el registro.

    Criterio observable: el stock quedó en cero después de la venta. En ese
    día pudo haber demanda que no se atendió. Es una regla imperfecta —no
    detecta la ruptura que ocurre a media tarde y se repone al día siguiente—
    y esa imperfección se declara como limitación.
    """
    ventas = movimientos[movimientos["tipo"] == "VENTA"].copy()
    ventas["censurado"] = ventas["stock_resultante"] <= 0
    return ventas


# --------------------------------------------------------------------------
# Estimadores de demanda
# --------------------------------------------------------------------------
def _estimar_ingenuo(x):
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _estimar_exclusion(x, censurado):
    limpio = x[~censurado]
    if len(limpio) < 5:
        return _estimar_ingenuo(x)
    return float(np.mean(limpio)), float(np.std(limpio, ddof=1))


def _estimar_em(x, censurado, max_iter=80, tol=1e-4):
    """
    Estimación por Esperanza-Maximización con censura por la derecha.

    Para un día censurado con venta s, se sabe que la demanda real fue D >= s.
    El paso E reemplaza esa observación por su esperanza condicional bajo la
    normal actual:

        E[D | D >= s] = mu + sigma * phi(z) / (1 - Phi(z)),   z = (s - mu)/sigma

    El paso M vuelve a estimar mu y sigma con las observaciones completadas.
    Se itera hasta que mu deje de moverse.
    """
    x = np.asarray(x, dtype=float)
    censurado = np.asarray(censurado, dtype=bool)
    if censurado.sum() == 0 or (~censurado).sum() < 5:
        return _estimar_ingenuo(x)

    mu, sigma = _estimar_exclusion(x, censurado)
    sigma = max(sigma, 1e-6)

    for _ in range(max_iter):
        # --- Paso E: completar las observaciones censuradas ---
        completo = x.copy()                     # valor esperado de cada dia
        varianza_extra = np.zeros_like(x)       # incertidumbre de lo imputado

        z = (x[censurado] - mu) / sigma
        cola = np.clip(1.0 - stats.norm.cdf(z), 1e-10, 1.0)
        razon = stats.norm.pdf(z) / cola        # razon inversa de Mills
        completo[censurado] = mu + sigma * razon
        varianza_extra[censurado] = np.clip(
            sigma ** 2 * (1.0 + z * razon - razon ** 2), 0.0, None)

        # --- Paso M: reestimar con las observaciones completadas ---
        mu_nuevo = float(np.mean(completo))
        sigma_nuevo = float(np.sqrt(max(
            np.mean((completo - mu_nuevo) ** 2 + varianza_extra), 1e-12)))

        convergio = abs(mu_nuevo - mu) < tol
        mu, sigma = mu_nuevo, max(sigma_nuevo, 1e-6)
        if convergio:
            break

    return mu, sigma


def estimar_demanda(movimientos, metodo="em"):
    """
    Estima media y desviación de la demanda diaria por SKU.

    metodo: "ingenuo" | "exclusion" | "em"
    """
    ventas = marcar_censura(movimientos)
    estimaciones = {}
    for sku, grupo in ventas.groupby("sku"):
        x = grupo["cantidad"].to_numpy(dtype=float)
        c = grupo["censurado"].to_numpy(dtype=bool)
        if metodo == "ingenuo":
            mu, sd = _estimar_ingenuo(x)
        elif metodo == "exclusion":
            mu, sd = _estimar_exclusion(x, c)
        else:
            mu, sd = _estimar_em(x, c)
        estimaciones[sku] = {
            "demanda_media": mu,
            "demanda_desv": max(sd, 1e-6),
            "dias_observados": len(x),
            "dias_censurados": int(c.sum()),
            "pct_censurado": float(c.mean() * 100),
            "metodo": metodo,
        }
    return estimaciones


# --------------------------------------------------------------------------
# Estimación del tiempo de entrega
# --------------------------------------------------------------------------
def estimar_lead_time(movimientos):
    """
    Estima media y desviación del tiempo de entrega por SKU.

    A diferencia de la demanda, el tiempo de entrega SÍ se observa completo:
    hay fecha de orden y fecha de llegada. No hay censura que corregir, pero
    sí hay pocos datos —una orden cada pocas semanas— y eso se refleja en la
    incertidumbre del estimador.
    """
    ordenes = movimientos[movimientos["tipo"] == "ORDEN"]
    if "dia_llegada" not in ordenes.columns:
        return {}
    ordenes = ordenes.dropna(subset=["dia_llegada"]).copy()
    ordenes["lead_time"] = ordenes["dia_llegada"] - ordenes["dia"]

    estimaciones = {}
    for sku, grupo in ordenes.groupby("sku"):
        lt = grupo["lead_time"].to_numpy(dtype=float)
        estimaciones[sku] = {
            "lead_time_medio": float(np.mean(lt)) if len(lt) else 4.0,
            "lead_time_desv": float(np.std(lt, ddof=1)) if len(lt) > 1 else 1.0,
            "n_ordenes": len(lt),
        }
    return estimaciones


# --------------------------------------------------------------------------
# Estacionalidad
# --------------------------------------------------------------------------
CICLO_DIAS = 312.0        # un año comercial de 6 días por semana
VENTANA_PERCEPCION = 30   # días que el dueño "recuerda" al estimar su demanda


def estimar_estacionalidad(dias, cantidades, ciclo=CICLO_DIAS):
    """
    Ajusta una estacionalidad sinusoidal anual por mínimos cuadrados.

    Se supone que la demanda sigue  media * (1 + a*sin(w t) + b*cos(w t)).
    Los coeficientes salen de una regresión lineal sobre la serie centrada,
    así que no hay optimización iterativa ni riesgo de no converger.

    Que la forma sea sinusoidal ES un supuesto del modelo, y hay que
    declararlo: captura una campaña anual suave (fiestas, temporada), no un
    pico de una semana. Si SOAZ tiene un pico agudo en una fecha concreta, esta
    forma lo suaviza y el gemelo lo subestimará.
    """
    d = np.asarray(dias, dtype=float)
    y = np.asarray(cantidades, dtype=float)
    media = float(np.mean(y))
    if media <= 1e-9 or len(y) < 60:
        return {"amplitud": 0.0, "fase": 0.0, "r2": 0.0}

    w = 2.0 * np.pi / ciclo
    X = np.column_stack([np.sin(w * d), np.cos(w * d), np.ones_like(d)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    amplitud = float(np.sqrt(a ** 2 + b ** 2) / max(c, 1e-9))
    fase = float(np.arctan2(b, a))
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - media) ** 2)) or 1.0
    return {"amplitud": min(amplitud, 0.6), "fase": fase, "r2": 1.0 - ss_res / ss_tot}


# --------------------------------------------------------------------------
# Cumplimiento del proveedor
# --------------------------------------------------------------------------
def estimar_cumplimiento(movimientos):
    """
    Estima con qué frecuencia el proveedor entrega menos de lo pedido.

    Se emparejan las órdenes con sus llegadas por día previsto. Es una
    conciliación imperfecta —dos órdenes que llegan el mismo día se confunden—
    y por eso se reporta a nivel agregado por SKU y no por orden.
    """
    ordenes = movimientos[movimientos["tipo"] == "ORDEN"]
    compras = movimientos[movimientos["tipo"] == "COMPRA"]
    salida = {}
    for sku in movimientos["sku"].unique():
        o = ordenes[ordenes["sku"] == sku]["cantidad"].sum()
        c = compras[compras["sku"] == sku]["cantidad"].sum()
        if o <= 0:
            salida[sku] = {"fraccion_recibida": 1.0, "prob_incumplimiento": 0.0,
                           "fraccion_si_incumple": 1.0}
            continue
        fraccion = float(min(1.0, c / o))
        # El faltante agregado se reparte entre "cada cuánto falla" y "cuánto
        # falta cuando falla". Sin más datos se fija lo segundo y se despeja
        # lo primero; el supuesto se declara.
        fraccion_si_falla = 0.6
        prob = float(np.clip((1.0 - fraccion) / (1.0 - fraccion_si_falla), 0.0, 1.0))
        salida[sku] = {"fraccion_recibida": fraccion,
                       "prob_incumplimiento": prob,
                       "fraccion_si_incumple": fraccion_si_falla}
    return salida


# --------------------------------------------------------------------------
# Ensamblado del estado estimado del gemelo
# --------------------------------------------------------------------------
def construir_estado_gemelo(catalogo, movimientos, metodo_demanda="em",
                            nivel_servicio=0.95, dia_corte=None):
    """
    Construye el estado del gemelo: un registro por SKU con SOLO parámetros
    observables o estimados. Es lo que consume el motor de simulación.
    """
    if dia_corte is not None:
        movimientos = movimientos[movimientos["dia"] <= dia_corte]

    demanda = estimar_demanda(movimientos, metodo_demanda)
    # La demanda PERCIBIDA por el negocio es el promedio crudo de ventas: es
    # con ese número —sesgado a la baja por la censura— con el que SOAZ ha
    # venido comprando. El gemelo necesita ambos: el corregido para planificar
    # el futuro, y el percibido para reproducir el pasado en la retro-prueba.
    # El dueño no promedia 18 meses: mira las últimas semanas. Estimar su
    # percepción con una ventana móvil corta —y no con todo el histórico— es
    # lo que permite que el gemelo reproduzca SUS decisiones, no las que
    # habría tomado un analista con la serie completa.
    dia_tope = int(movimientos["dia"].max())
    ventana = movimientos[movimientos["dia"] > dia_tope - VENTANA_PERCEPCION]
    percibida = estimar_demanda(ventana if len(ventana) else movimientos, "ingenuo")
    lead = estimar_lead_time(movimientos)
    cumplimiento = estimar_cumplimiento(movimientos)
    ventas = movimientos[movimientos["tipo"] == "VENTA"]
    stock_final = ventas.sort_values("dia").groupby("sku")["stock_resultante"].last()
    dia_final = int(movimientos["dia"].max())

    estado = []
    for s in catalogo:
        sku = s["sku"]
        d = demanda.get(sku, {"demanda_media": 1.0, "demanda_desv": 1.0,
                              "dias_censurados": 0, "pct_censurado": 0.0})
        l = lead.get(sku, {"lead_time_medio": 4.0, "lead_time_desv": 1.5, "n_ordenes": 0})
        cum = cumplimiento.get(sku, {"prob_incumplimiento": 0.0, "fraccion_si_incumple": 1.0})
        v_sku = ventas[ventas["sku"] == sku]
        est_sku = estimar_estacionalidad(v_sku["dia"].to_numpy(),
                                         v_sku["cantidad"].to_numpy())
        estado.append({
            "sku": sku,
            "descripcion": s["descripcion"],
            "familia": s["familia"],
            "proveedor": s["proveedor"],
            "clase": s.get("clase", "A"),
            "costo_unitario": s["costo_unitario"],
            "vida_util_dias": s["vida_util_dias"],
            "lote_minimo": s["lote_minimo"],
            # --- parámetros ESTIMADOS del histórico ---
            "demanda_media": d["demanda_media"],
            "demanda_desv": d["demanda_desv"],
            "lead_time_medio": l["lead_time_medio"],
            "lead_time_desv": l["lead_time_desv"],
            "n_ordenes_observadas": l["n_ordenes"],
            "dias_censurados": d.get("dias_censurados", 0),
            "pct_censurado": d.get("pct_censurado", 0.0),
            "nivel_servicio": nivel_servicio,
            "z": NIVELES_SERVICIO.get(round(nivel_servicio, 3), 1.6449),
            "stock_inicial": int(stock_final.get(sku, 0)),
            "demanda_percibida": percibida.get(sku, {}).get("demanda_media",
                                                            d["demanda_media"]),
            "estacionalidad": est_sku.get("amplitud", 0.0),
            "fase_estacional": est_sku.get("fase", 0.0),
            "r2_estacionalidad": est_sku.get("r2", 0.0),
            "prob_incumplimiento": cum.get("prob_incumplimiento", 0.0),
            "fraccion_si_incumple": cum.get("fraccion_si_incumple", 1.0),
            "dia_inicial": dia_final,
        })
    return estado


# --------------------------------------------------------------------------
# Diagnóstico del sesgo por censura
# --------------------------------------------------------------------------
def comparar_estimadores(catalogo, movimientos, verdad):
    """
    Compara los tres estimadores contra la demanda verdadera.

    Esta función SOLO existe para el informe: usa `verdad`, que en el proyecto
    real no está disponible. Es la evidencia de que la corrección importa, y
    la justificación de por qué el motor usa EM y no el promedio de ventas.
    """
    reales = verdad.groupby("sku")["demanda_real"].agg(["mean", "std"])
    metodos = ["ingenuo", "exclusion", "em"]
    estimaciones = {m: estimar_demanda(movimientos, m) for m in metodos}

    filas = []
    for s in catalogo:
        sku = s["sku"]
        if sku not in reales.index:
            continue
        real_mu = float(reales.loc[sku, "mean"])
        fila = {
            "SKU": sku,
            "Clase": s.get("clase", "A"),
            "Demanda real": round(real_mu, 2),
            "% días censurados": round(estimaciones["em"][sku]["pct_censurado"], 1),
        }
        for m in metodos:
            est = estimaciones[m][sku]["demanda_media"]
            fila[f"Est. {m}"] = round(est, 2)
            fila[f"Sesgo {m} (%)"] = round((est - real_mu) / max(real_mu, 1e-9) * 100, 2)
        filas.append(fila)

    resumen = {}
    for m in metodos:
        sesgos = [f[f"Sesgo {m} (%)"] for f in filas]
        resumen[m] = {
            "sesgo_medio": float(np.mean(sesgos)),
            "sesgo_absoluto_medio": float(np.mean(np.abs(sesgos))),
            "peor_sesgo": float(min(sesgos)),
        }
    return {"tabla": filas, "resumen": resumen}
