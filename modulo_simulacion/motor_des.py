# modulo_simulacion/motor_des.py
"""
Motor de simulación de eventos discretos del gemelo — SimPy.

POR QUÉ EVENTOS DISCRETOS Y NO UN BUCLE POR DÍA
------------------------------------------------
Un bucle diario obliga a que todo ocurra en la misma grilla. El inventario no
funciona así: la demanda llega a lo largo del día, la orden se emite en el
instante en que la posición cruza el punto de reorden, y el camión llega
cuando llega —un tiempo continuo que se sortea de una gamma. SimPy deja que
el reloj salte de evento a evento y que cada proceso tenga su propio ritmo,
que es lo que permite representar la revisión continua sin fingir que ocurre
una vez al día.

CUATRO PROCESOS POR SKU
------------------------
    demanda   → llega, se despacha lo que hay, se registra lo que faltó
    revisión  → continua en (s,Q), cada R días en (R,S); emite la orden
    llegada   → tras el tiempo de entrega; puede llegar incompleta
    día       → vencimiento de lotes y registro del estado

LA VARIABLE QUE DISPARA LA COMPRA ES LA POSICIÓN, NO EL STOCK
--------------------------------------------------------------
    posicion = stock_a_mano + en_transito - pendientes_por_atender

Confundirla con el stock a la mano produce sobrecompra: el sistema vuelve a
pedir lo que ya viene en camino. Es el error más común en una simulación de
inventarios y el más difícil de ver en los resultados, porque el servicio
mejora y solo el costo delata el problema.
"""
import numpy as np
import simpy

from .parametros import COSTOS, NEGOCIO

CICLO_DIAS = 312.0   # año comercial de 6 días por semana
from .politicas import construir_politica, construir_politica_practica_actual


class GemeloSKU:
    """Réplica virtual de un SKU dentro del almacén."""

    def __init__(self, env, estado, politica, rng, escenario, presupuesto=None):
        self.env = env
        self.e = estado
        self.p = politica
        self.rng = rng
        self.esc = escenario
        self.presupuesto = presupuesto
        # Qué media de demanda usa el motor para GENERAR la demanda. Al
        # reproducir el pasado se usa la corregida (es la demanda que
        # realmente hubo); la percibida solo entra en la REGLA DE COMPRA.
        self.media_planificacion = estado["demanda_media"]

        # La práctica actual de SOAZ no usa un número fijo: el dueño reestima
        # su promedio cada mes con lo que vendió. Congelar esa percepción hacía
        # que el gemelo no siguiera la campaña estacional y quebrara stock donde
        # el negocio real sí se había ajustado. Reproducir la fidelidad de un
        # gemelo exige modelar cómo APRENDE el que decide, no solo lo que cree hoy.
        self.percibida = estado.get("demanda_percibida", estado["demanda_media"])
        self.ventas_recientes = []
        if politica["tipo"] == "(R,s,Q)":
            env.process(self.proceso_aprendizaje())

        # --- Estado sincronizado desde el sistema real ---
        self.stock = float(estado["stock_inicial"])
        self.en_transito = 0.0
        self.pendientes = 0.0                 # demanda no atendida acumulada
        self.lotes = [[0.0, self.stock]]      # [[dia_ingreso, cantidad]]

        # --- Acumuladores ---
        self.demanda_total = 0.0
        self.atendida = 0.0
        self.perdida = 0.0
        self.merma = 0.0
        self.ordenes = 0
        self.ordenes_rechazadas = 0
        self.dias_ruptura = 0
        self.unidades_dia = 0.0               # integral del stock, para costo de mantener
        self.serie = []                       # traza diaria

        env.process(self.proceso_demanda())
        env.process(self.proceso_dia())
        if politica["tipo"] in ("(R,S)", "(R,s,Q)"):
            env.process(self.proceso_revision_periodica())

    # ----------------------------------------------------------------------
    @property
    def posicion(self):
        return self.stock + self.en_transito - self.pendientes

    def _demanda_del_dia(self, dia):
        """
        Demanda diaria con estacionalidad estimada y la palanca del escenario.

        El día se cuenta desde `dia_inicial`, que es el último día del
        histórico: si el gemelo arranca en noviembre, tiene que arrancar en la
        parte de la curva estacional que corresponde a noviembre. Arrancar
        siempre en la fase cero es un error silencioso que solo se nota cuando
        la retro-prueba falla.
        """
        absoluto = self.e.get("dia_inicial", 0) + dia
        amplitud = self.e.get("estacionalidad", 0.0)
        estacion = 1.0
        if amplitud > 1e-6:
            estacion = 1.0 + amplitud * np.sin(
                2 * np.pi * absoluto / CICLO_DIAS + self.e.get("fase_estacional", 0.0))
        media = self.media_planificacion * estacion * self.esc.get("factor_demanda", 1.0)
        return max(0.0, float(self.rng.normal(media, self.e["demanda_desv"])))

    def _lead_time(self):
        mu = self.e["lead_time_medio"] * self.esc.get("factor_lead_time", 1.0)
        sd = self.e["lead_time_desv"] * self.esc.get("factor_desv_lead_time", 1.0)
        if sd <= 1e-6:
            return max(1.0, mu)
        forma = (mu / sd) ** 2
        escala = sd ** 2 / mu
        return max(1.0, float(self.rng.gamma(forma, escala)))

    # ----------------------------------------------------------------------
    def proceso_demanda(self):
        """La demanda del día llega repartida en cuatro momentos."""
        while True:
            dia = int(self.env.now)
            porciones = 4
            cantidad = self._demanda_del_dia(dia) / porciones
            for _ in range(porciones):
                yield self.env.timeout(1.0 / porciones)
                self.demanda_total += cantidad
                despachado = min(cantidad, self.stock)
                self.stock -= despachado
                self.atendida += despachado
                faltante = cantidad - despachado
                if faltante > 1e-9:
                    # Venta perdida: en una comercializadora el cliente se va,
                    # no espera. Es el supuesto correcto para este negocio y
                    # cambia el costo de la ruptura respecto de un backorder.
                    self.perdida += faltante
                self._consumir_lotes(despachado)
                if self.p["tipo"] == "(s,Q)":
                    self._revisar_continua()

    def proceso_revision_periodica(self):
        """
        Revisión cada R días.

        En (R,S) se pide hasta el nivel objetivo. En (R,s,Q) —la práctica
        actual de SOAZ— se pide una cantidad fija cuando la posición cayó por
        debajo del punto de reorden.
        """
        R = self.p["periodo_revision"] or 7
        while True:
            yield self.env.timeout(R)
            if self.p["tipo"] == "(R,S)":
                objetivo = self.p["nivel_objetivo_S"]
                if self.posicion < objetivo:
                    self._emitir_orden(objetivo - self.posicion)
            else:  # (R,s,Q)
                if self.posicion <= self.p["punto_reorden"]:
                    self._emitir_orden(self.p["cantidad_pedido"])

    def _revisar_continua(self):
        if self.posicion <= self.p["punto_reorden"]:
            self._emitir_orden(self.p["cantidad_pedido"])

    def _emitir_orden(self, cantidad):
        cantidad = max(self.e.get("lote_minimo", 1), cantidad)
        costo = cantidad * self.e["costo_unitario"]
        # Escenario de presupuesto limitado: el racionamiento es un evento del
        # modelo, no un supuesto silencioso.
        if self.presupuesto is not None and not self.presupuesto.autorizar(costo):
            self.ordenes_rechazadas += 1
            return
        self.ordenes += 1
        self.en_transito += cantidad
        self.env.process(self._proceso_llegada(cantidad))

    def _proceso_llegada(self, cantidad):
        yield self.env.timeout(self._lead_time())
        recibido = cantidad
        # El proveedor puede incumplir: llega menos de lo pedido.
        if self.rng.random() < self.e.get("prob_incumplimiento", 0.0):
            recibido = cantidad * self.e.get("fraccion_si_incumple", 1.0)
        self.en_transito -= cantidad
        self.stock += recibido
        self.lotes.append([self.env.now, recibido])

    def proceso_aprendizaje(self):
        """Cada 30 días el negocio reestima su demanda con lo que vendió."""
        while True:
            yield self.env.timeout(30.0)
            if len(self.ventas_recientes) >= 10:
                self.percibida = float(np.mean(self.ventas_recientes[-30:]))
                self.p["punto_reorden"] = self.percibida * 10.0
                self.p["cantidad_pedido"] = max(self.e.get("lote_minimo", 1),
                                                self.percibida * 20.0)

    def proceso_dia(self):
        """Cierre del día: vencimiento de lotes, costo de mantener y traza."""
        while True:
            yield self.env.timeout(1.0)
            self.ventas_recientes.append(self.atendida - sum(self.ventas_recientes))
            self._vencer_lotes()
            self.unidades_dia += self.stock
            if self.stock <= 1e-9:
                self.dias_ruptura += 1
            self.serie.append({
                "dia": int(self.env.now),
                "stock": self.stock,
                "posicion": self.posicion,
                "en_transito": self.en_transito,
                "punto_reorden": self.p["punto_reorden"],
                "stock_seguridad": self.p["stock_seguridad"],
            })

    # ----------------------------------------------------------------------
    def _consumir_lotes(self, cantidad):
        """Consumo PEPS: sale primero el lote más antiguo."""
        restante = cantidad
        nuevos = []
        for lote in self.lotes:
            if restante <= 1e-12:
                nuevos.append(lote)
            elif lote[1] <= restante:
                restante -= lote[1]
            else:
                lote[1] -= restante
                restante = 0.0
                nuevos.append(lote)
        self.lotes = nuevos

    def _vencer_lotes(self):
        vida = self.e["vida_util_dias"]
        ahora = self.env.now
        vivos, vencido = [], 0.0
        for d, q in self.lotes:
            if ahora - d >= vida:
                vencido += q
            else:
                vivos.append([d, q])
        if vencido > 1e-9:
            self.lotes = vivos
            self.stock = max(0.0, self.stock - vencido)
            self.merma += vencido

    # ----------------------------------------------------------------------
    def resultados(self, dias):
        """Indicadores del SKU al cierre de la simulación."""
        c = self.e["costo_unitario"]
        tasa_dia = COSTOS["tasa_mantenimiento_anual"] / (NEGOCIO["dias_habiles_semana"] * 52)
        costo_ordenar = self.ordenes * COSTOS["costo_pedido"]
        costo_mantener = self.unidades_dia * c * tasa_dia
        costo_ruptura = self.perdida * c * COSTOS["penalidad_ruptura_factor"]
        costo_merma = self.merma * c * COSTOS["valor_merma_factor"]
        fill_rate = (self.atendida / self.demanda_total * 100) if self.demanda_total > 0 else 100.0

        return {
            "sku": self.e["sku"],
            "familia": self.e["familia"],
            "clase": self.e.get("clase", "A"),
            "demanda_total": self.demanda_total,
            "atendida": self.atendida,
            "perdida": self.perdida,
            "fill_rate": fill_rate,
            "dias_ruptura": self.dias_ruptura,
            "pct_dias_ruptura": self.dias_ruptura / max(dias, 1) * 100,
            "ordenes": self.ordenes,
            "ordenes_rechazadas": self.ordenes_rechazadas,
            "merma": self.merma,
            "stock_medio": self.unidades_dia / max(dias, 1),
            "capital_inmovilizado": self.unidades_dia / max(dias, 1) * c,
            "costo_ordenar": costo_ordenar,
            "costo_mantener": costo_mantener,
            "costo_ruptura": costo_ruptura,
            "costo_merma": costo_merma,
            "costo_total": costo_ordenar + costo_mantener + costo_ruptura + costo_merma,
            "serie": self.serie,
        }


def gasto_reposicion_mensual(estado):
    """Valor de compra que la reposición requiere en un mes, a costo."""
    return sum(s["demanda_media"] * s["costo_unitario"] for s in estado) * NEGOCIO["dias_por_mes"]


class Presupuesto:
    """Presupuesto mensual compartido de compra. Se reinicia cada 26 días."""

    def __init__(self, env, monto_mensual):
        self.env = env
        self.monto = monto_mensual
        self.disponible = monto_mensual
        env.process(self._renovar())

    def _renovar(self):
        while True:
            yield self.env.timeout(NEGOCIO["dias_por_mes"])
            self.disponible = self.monto

    def autorizar(self, costo):
        if costo <= self.disponible:
            self.disponible -= costo
            return True
        return False


# --------------------------------------------------------------------------
def simular(estado, escenario=None, dias=90, semilla=0, politica_base=None):
    """
    Corre una réplica del gemelo sobre todos los SKU del alcance.

    `escenario` es el diccionario de palancas de parametros.ESCENARIOS.
    Devuelve los resultados por SKU y los agregados del almacén.
    """
    escenario = escenario or {}
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()

    presupuesto = None
    monto = escenario.get("presupuesto_mensual")
    if monto is None and "fraccion_presupuesto" in escenario:
        # Un presupuesto en soles absolutos no es comparable entre catálogos.
        # Se declara como fracción de lo que la reposición efectivamente pide,
        # que es la forma en que el negocio lo vive: "me alcanza para el 85%".
        monto = escenario["fraccion_presupuesto"] * gasto_reposicion_mensual(estado)
    if monto is not None:
        presupuesto = Presupuesto(env, monto)

    tipo = escenario.get("tipo_politica", (politica_base or {}).get("tipo", "(s,Q)"))
    periodo = escenario.get("periodo_revision_dias", (politica_base or {}).get("periodo_revision_dias", 7))
    nivel = escenario.get("nivel_servicio", (politica_base or {}).get("nivel_servicio", 0.95))

    gemelos = []
    for e in estado:
        if tipo == "(R,s,Q)":
            politica = construir_politica_practica_actual(e, periodo_revision=periodo)
        else:
            politica = construir_politica(e, tipo=tipo, periodo_revision=periodo,
                                          nivel_servicio=nivel)
        gemelos.append(GemeloSKU(env, e, politica,
                                 np.random.default_rng(rng.integers(2 ** 32)),
                                 escenario, presupuesto))

    env.run(until=dias)
    por_sku = [g.resultados(dias) for g in gemelos]

    demanda = sum(r["demanda_total"] for r in por_sku)
    atendida = sum(r["atendida"] for r in por_sku)
    return {
        "por_sku": por_sku,
        "dias": dias,
        "fill_rate": (atendida / demanda * 100) if demanda > 0 else 100.0,
        "costo_total": sum(r["costo_total"] for r in por_sku),
        "costo_ordenar": sum(r["costo_ordenar"] for r in por_sku),
        "costo_mantener": sum(r["costo_mantener"] for r in por_sku),
        "costo_ruptura": sum(r["costo_ruptura"] for r in por_sku),
        "costo_merma": sum(r["costo_merma"] for r in por_sku),
        "capital_inmovilizado": sum(r["capital_inmovilizado"] for r in por_sku),
        "ordenes": sum(r["ordenes"] for r in por_sku),
        "ordenes_rechazadas": sum(r["ordenes_rechazadas"] for r in por_sku),
        "sku_con_ruptura": sum(1 for r in por_sku if r["dias_ruptura"] > 0),
        "n_sku": len(por_sku),
        "unidades_perdidas": sum(r["perdida"] for r in por_sku),
        "unidades_merma": sum(r["merma"] for r in por_sku),
        "politica": {"tipo": tipo, "nivel_servicio": nivel, "periodo_revision": periodo},
    }
