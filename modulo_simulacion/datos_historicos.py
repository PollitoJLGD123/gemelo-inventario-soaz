# modulo_simulacion/datos_historicos.py
"""
Generación del histórico de movimientos de SOAZ.

QUÉ ES ESTE MÓDULO Y QUÉ NO ES
-------------------------------
En el proyecto final este módulo se reemplaza por una lectura de la tabla
`movimiento_inventario` del sistema web de la tesis. Mientras ese sistema no
esté operativo —el riesgo #1 declarado en la propuesta— el gemelo necesita
una fuente de movimientos, y esta la provee: simula el pasado del negocio
bajo la práctica ACTUAL de compra y emite exactamente los registros que un
sistema transaccional habría capturado.

La distinción crítica, y el motivo por el que este módulo existe en vez de
generar demanda con un `rng.normal` y ya:

    DEMANDA  ≠  VENTA

La demanda es cuánto quería comprar el cliente. La venta es cuánto se le
pudo despachar. Cuando hay ruptura de stock el cliente se va y **no deja
rastro en el sistema**. El histórico que produce este módulo está censurado
igual que el real, y eso obliga al gemelo a lidiar con el problema en vez de
esquivarlo.

Para poder medir el sesgo, el módulo devuelve también la demanda verdadera,
en una estructura aparte que el motor NO consume. Solo la usa la validación.
"""
import numpy as np
import pandas as pd

from .parametros import NEGOCIO


# --------------------------------------------------------------------------
# Práctica actual de compra en SOAZ
# --------------------------------------------------------------------------
PRACTICA_ACTUAL = {
    "dias_cobertura_reorden": 10,   # SUPUESTO — regla del pulgar del negocio
    "dias_cobertura_pedido": 20,    # cuánto se compra cuando se compra
    "dia_revision_semana": 1,       # revisa los lunes, cuando baja a Trujillo
    "descripcion": ("Revisión semanal a ojo: se pide cuando el stock alcanza para "
                    "unos 10 días y se compra para unos 20. No hay stock de "
                    "seguridad explícito ni se considera la variabilidad del "
                    "proveedor."),
}


def _demanda_del_dia(sku, dia, rng, factor=1.0):
    """
    Demanda verdadera del día: media estacional + ruido, truncada en cero.

    La estacionalidad es sinusoidal anual. No es un supuesto inocente —
    determina si el gemelo puede anticipar la campaña de fiestas— y por eso
    la amplitud está declarada por familia en parametros.py.
    """
    estacion = 1.0 + sku["_estacionalidad"] * np.sin(
        2 * np.pi * dia / 312.0 + sku["_fase_estacional"])
    media = sku["_demanda_media"] * estacion * factor
    valor = rng.normal(media, sku["_demanda_desv"])
    return max(0, int(round(valor)))


def _lead_time(sku, rng):
    """Tiempo de entrega realizado: gamma truncada, nunca menor a 1 día."""
    mu, sd = sku["_lead_time_medio"], sku["_lead_time_desv"]
    if sd <= 0:
        return max(1, int(round(mu)))
    forma = (mu / sd) ** 2
    escala = sd ** 2 / mu
    return max(1, int(round(rng.gamma(forma, escala))))


def generar_historico(catalogo, dias=None, semilla=2026, factor_demanda=1.0,
                      demanda_observada=None):
    """
    Simula el pasado del negocio y devuelve lo que el sistema habría registrado.

    `demanda_observada`, si viene, es un dict (sku, día) → unidades. Así el
    mismo lazo de compra corre sobre ventas reales (Favorita) en vez de sobre
    una demanda sorteada. Sin ese dict, el histórico sigue siendo sintético:
    las pruebas del motor no dependen del dataset.

    Devuelve:
      movimientos : DataFrame con las columnas que tendría el sistema real
      verdad      : DataFrame con la demanda real por día (NO lo consume el motor)
    """
    if demanda_observada:
        dias = max(dia for _, dia in demanda_observada)
    dias = dias or 540
    rng = np.random.default_rng(semilla)
    movimientos, verdad = [], []

    for sku in catalogo:
        # Estado inicial: 20 días de cobertura, como haría el negocio
        stock = int(sku["_demanda_media"] * PRACTICA_ACTUAL["dias_cobertura_pedido"])
        en_transito = []           # [(dia_llegada, cantidad)]
        lotes = [(0, stock)]       # [(dia_ingreso, cantidad)] para la merma
        # La práctica actual usa un promedio "a ojo": arranca con el verdadero
        # y luego se ajusta con lo que el dueño observa (o sea, con ventas).
        media_percibida = sku["_demanda_media"]
        ventas_recientes = []

        for dia in range(1, dias + 1):
            # 1. Llegadas del día
            llegadas = [(d, q) for d, q in en_transito if d == dia]
            for _, q in llegadas:
                stock += q
                lotes.append((dia, q))
                movimientos.append({
                    "dia": dia, "sku": sku["sku"], "tipo": "COMPRA",
                    "cantidad": q, "stock_resultante": stock,
                })
            en_transito = [(d, q) for d, q in en_transito if d > dia]

            # 2. Merma por vencimiento
            vida = sku["vida_util_dias"]
            vencidos = sum(q for d, q in lotes if dia - d >= vida)
            if vencidos > 0:
                lotes = [(d, q) for d, q in lotes if dia - d < vida]
                stock = max(0, stock - vencidos)
                movimientos.append({
                    "dia": dia, "sku": sku["sku"], "tipo": "MERMA",
                    "cantidad": vencidos, "stock_resultante": stock,
                })

            # 3. Demanda del día — la verdad. Si hay serie real, manda esa.
            if demanda_observada is not None:
                demanda = int(round(max(0.0, float(
                    demanda_observada.get((sku["sku"], dia), 0.0)))))
            else:
                demanda = _demanda_del_dia(sku, dia, rng, factor_demanda)
            # 4. Venta — lo único que el sistema ve
            venta = min(demanda, stock)
            perdida = demanda - venta
            stock -= venta
            # descontar de los lotes más antiguos (PEPS)
            restante = venta
            nuevos_lotes = []
            for d, q in lotes:
                if restante <= 0:
                    nuevos_lotes.append((d, q))
                elif q <= restante:
                    restante -= q
                else:
                    nuevos_lotes.append((d, q - restante))
                    restante = 0
            lotes = nuevos_lotes

            movimientos.append({
                "dia": dia, "sku": sku["sku"], "tipo": "VENTA",
                "cantidad": venta, "stock_resultante": stock,
            })
            verdad.append({
                "dia": dia, "sku": sku["sku"],
                "demanda_real": demanda, "venta_registrada": venta,
                "demanda_perdida": perdida, "stock_inicio_dia": stock + venta,
                "hubo_ruptura": stock == 0 or perdida > 0,
            })

            ventas_recientes.append(venta)
            if len(ventas_recientes) > 30:
                ventas_recientes.pop(0)
            if dia % 30 == 0 and ventas_recientes:
                # El dueño ajusta su promedio con lo que vendió — o sea, con
                # el dato censurado. Así se propaga el sesgo en la vida real.
                media_percibida = float(np.mean(ventas_recientes))

            # 5. Revisión semanal y decisión de compra
            if dia % 7 == PRACTICA_ACTUAL["dia_revision_semana"] % 7:
                posicion = stock + sum(q for _, q in en_transito)
                punto_reorden = media_percibida * PRACTICA_ACTUAL["dias_cobertura_reorden"]
                if posicion <= punto_reorden:
                    cantidad = int(max(
                        sku["lote_minimo"],
                        round(media_percibida * PRACTICA_ACTUAL["dias_cobertura_pedido"])))
                    lt = _lead_time(sku, rng)
                    if rng.random() < sku["_prob_incumplimiento"]:
                        cantidad = int(cantidad * sku["_fraccion_si_incumple"])
                    en_transito.append((dia + lt, cantidad))
                    movimientos.append({
                        "dia": dia, "sku": sku["sku"], "tipo": "ORDEN",
                        "cantidad": cantidad, "stock_resultante": stock,
                        "dia_llegada": dia + lt,
                    })

    return pd.DataFrame(movimientos), pd.DataFrame(verdad)


def resumen_historico(movimientos, verdad=None):
    """Resumen del histórico, para la pestaña de datos."""
    ventas = movimientos[movimientos["tipo"] == "VENTA"]
    ordenes = movimientos[movimientos["tipo"] == "ORDEN"]
    mermas = movimientos[movimientos["tipo"] == "MERMA"]
    dias = int(movimientos["dia"].max())

    resumen = {
        "Días de histórico": dias,
        "Meses aproximados": round(dias / NEGOCIO["dias_por_mes"], 1),
        "SKU con movimiento": int(movimientos["sku"].nunique()),
        "Registros totales": len(movimientos),
        "Unidades vendidas": int(ventas["cantidad"].sum()),
        "Órdenes emitidas": len(ordenes),
        "Unidades de merma": int(mermas["cantidad"].sum()) if len(mermas) else 0,
    }
    if verdad is not None and len(verdad):
        dias_ruptura = int(verdad["hubo_ruptura"].sum())
        perdida = int(verdad["demanda_perdida"].sum())
        demanda_total = int(verdad["demanda_real"].sum())
        resumen["Días-SKU con ruptura"] = dias_ruptura
        resumen["Unidades de demanda perdida"] = perdida
        resumen["Demanda no observada (%)"] = round(perdida / max(demanda_total, 1) * 100, 2)
    return resumen
