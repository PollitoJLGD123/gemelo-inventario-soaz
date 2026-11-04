#!/usr/bin/env python3
# verificar_motor.py
"""
Autoverificación numérica del motor del Gemelo Digital de Inventario — SOAZ.

No comprueba que el código corra: comprueba que el MODELO se comporte como
dice comportarse. Cada prueba corresponde a una afirmación que el informe va a
hacer, y falla ruidosamente si la afirmación no se sostiene.

Uso:  python verificar_motor.py
"""
import importlib.util
import sys
import time

import numpy as np

sys.path.insert(0, ".")


def _consola_utf8():
    """
    Sin esto, `python verificar_motor.py > archivo.log` revienta en Windows.

    La salida redirigida usa cp1252, y el primer «→» de un detalle tumba el
    reporte entero DESPUÉS de que las 15 pruebas ya corrieron: el proceso
    termina en 1 y parece que algo falló, cuando lo que falló fue imprimir.
    Es el mismo arreglo que hace lanzador.py para la consola del ejecutable.
    """
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_consola_utf8()

from modulo_simulacion.catalogo import (
    generar_catalogo, clasificar_abc, catalogo_en_alcance, resumen_abc,
)
from modulo_simulacion.datos_historicos import generar_historico
from modulo_simulacion.escenarios import ejecutar_escenario, comparar_escenarios
from modulo_simulacion.estimacion import comparar_estimadores, construir_estado_gemelo
from modulo_simulacion.explicabilidad import (
    explicar_sku, shapley_exacto, vector_referencia, tornado_sensibilidad,
    fidelidad_sustituto, descomponer_escenario,
)
from modulo_simulacion.motor_des import simular
from modulo_simulacion.parametros import CATALOGO, FIDELIDAD, FACTORES_SHAPLEY
from modulo_simulacion.politicas import (
    construir_politica, stock_seguridad, stock_seguridad_simplificado, vector_factores,
)
from modulo_simulacion.validacion import retro_prueba, pisos_irreducibles

RESULTADOS = []
_MUNDO = {}


def mundo():
    """Catálogo, histórico y estado del gemelo, construidos una sola vez."""
    if not _MUNDO:
        rng = np.random.default_rng(2026)
        completo = clasificar_abc(generar_catalogo(rng))
        catalogo = catalogo_en_alcance(completo)
        movimientos, verdad = generar_historico(catalogo, 540, 2026)
        estado = construir_estado_gemelo(catalogo, movimientos)
        _MUNDO.update(completo=completo, catalogo=catalogo, movimientos=movimientos,
                      verdad=verdad, estado=estado)
    return _MUNDO


def prueba(nombre):
    def envoltorio(fn):
        def ejecutar():
            t = time.time()
            try:
                RESULTADOS.append((nombre, True, fn(), time.time() - t))
            except AssertionError as e:
                RESULTADOS.append((nombre, False, str(e), time.time() - t))
        return ejecutar
    return envoltorio


# ==========================================================================
@prueba("El alcance ABC concentra el movimiento declarado")
def t_abc():
    m = mundo()
    filas = resumen_abc(m["completo"])
    a = next(f for f in filas if f["Clase"] == "A")
    objetivo = CATALOGO["umbral_clase_a"] * 100
    assert abs(a["% del movimiento"] - objetivo) < 8, (
        f"la clase A concentra {a['% del movimiento']}%, lejos del {objetivo}% declarado")
    assert a["% del catálogo"] < 55, (
        f"la clase A es el {a['% del catálogo']}% del catálogo: la curva ABC está plana")
    return (f"{a['SKU']} SKU ({a['% del catálogo']}% del catálogo) concentran "
            f"{a['% del movimiento']}% del movimiento")


@prueba("El histórico está censurado, como el real")
def t_censura_existe():
    m = mundo()
    v = m["verdad"]
    perdida = float(v["demanda_perdida"].sum() / max(v["demanda_real"].sum(), 1) * 100)
    assert perdida > 2.0, (
        f"solo {perdida:.2f}% de demanda no observada: el histórico no reproduce el "
        f"problema de censura que el gemelo dice resolver")
    return f"{perdida:.2f}% de la demanda nunca llegó al registro (días con ruptura)"


@prueba("La corrección por censura reduce el sesgo de la demanda")
def t_censura_corregida():
    m = mundo()
    c = comparar_estimadores(m["catalogo"], m["movimientos"], m["verdad"])["resumen"]
    ing, em = abs(c["ingenuo"]["sesgo_medio"]), abs(c["em"]["sesgo_medio"])
    assert ing > 3.0, f"el estimador ingenuo apenas se sesga ({ing:.2f}%): no hay qué corregir"
    assert em < ing / 2, f"EM ({em:.2f}%) no mejora suficiente sobre el ingenuo ({ing:.2f}%)"
    return (f"promedio de ventas: {c['ingenuo']['sesgo_medio']:+.2f}% de sesgo "
            f"(peor SKU {c['ingenuo']['peor_sesgo']:+.1f}%) → EM: "
            f"{c['em']['sesgo_medio']:+.2f}%")


@prueba("El stock de seguridad completo supera al simplificado")
def t_ss_lead_time():
    m = mundo()
    razones = []
    for e in m["estado"]:
        p = construir_politica(e)
        if p["ss_simplificado"] > 0:
            razones.append(p["stock_seguridad"] / p["ss_simplificado"])
    media = float(np.mean(razones))
    assert media > 1.1, (
        f"la fórmula completa apenas supera a la simplificada ({media:.2f}×): "
        f"la variabilidad del proveedor no está entrando")
    return (f"omitir sigma_L subestima el stock de seguridad en "
            f"{(1 - 1 / media) * 100:.0f}% en promedio ({media:.2f}× de diferencia)")


@prueba("Axioma de eficiencia de Shapley")
def t_eficiencia():
    m = mundo()
    errores = [explicar_sku(e, m["estado"])["error_eficiencia"] for e in m["estado"][:12]]
    peor = max(errores)
    assert peor < 1e-6, f"error de eficiencia demasiado grande: {peor:.2e}"
    return f"12 SKU, peor error {peor:.2e} sobre {1 << len(FACTORES_SHAPLEY)} coaliciones"


@prueba("Axioma nulo (dummy) de Shapley")
def t_dummy():
    m = mundo()
    base = vector_referencia(m["estado"])
    v = dict(base)
    v["demanda_media"] = base["demanda_media"] * 2.0
    v["lead_time_desv"] = base["lead_time_desv"] * 3.0
    sh = shapley_exacto(v, base)
    nulos = {k: sh["contribuciones"][k] for k in FACTORES_SHAPLEY
             if k not in ("demanda_media", "lead_time_desv")}
    peor = max(abs(x) for x in nulos.values())
    assert peor < 1e-6, f"un factor idéntico a la referencia recibió contribución {peor:.2e}"
    return (f"{len(nulos)} factores en su valor de referencia reciben 0 "
            f"(peor |phi| = {peor:.1e})")


@prueba("El sustituto analítico es fiel a la simulación")
def t_sustituto():
    m = mundo()
    f = fidelidad_sustituto(m["estado"], dias=312, n_replicas=6)
    assert f["correlacion"] > 0.85, (
        f"correlación de solo {f['correlacion']:.3f}: la explicación de Shapley "
        f"describiría una función que no es el motor")
    assert f["error_absoluto_medio"] < 40, f"error medio {f['error_absoluto_medio']:.1f}%"
    return (f"|error| medio {f['error_absoluto_medio']:.1f}%, correlación "
            f"{f['correlacion']:.3f} — apto para explicar")


@prueba("Las políticas se ordenan como la teoría predice")
def t_orden_politicas():
    m = mundo()
    corridas = {}
    for nivel in (0.90, 0.99):
        rs = [simular(m["estado"], {"nivel_servicio": nivel}, dias=90, semilla=900 + i * 71)
              for i in range(6)]
        corridas[nivel] = (float(np.mean([r["fill_rate"] for r in rs])),
                           float(np.mean([r["capital_inmovilizado"] for r in rs])))
    f90, k90 = corridas[0.90]
    f99, k99 = corridas[0.99]
    assert f99 > f90, f"exigir 99% no mejora el fill rate ({f99:.2f}% vs {f90:.2f}%)"
    assert k99 > k90, f"exigir 99% no aumenta el capital inmovilizado"
    return (f"servicio 90% → fill {f90:.2f}%, capital {k90:,.0f} | "
            f"servicio 99% → fill {f99:.2f}%, capital {k99:,.0f}")


@prueba("Reproducibilidad con la misma semilla")
def t_reproducible():
    m = mundo()
    a = simular(m["estado"], {}, dias=60, semilla=31337)["costo_total"]
    b = simular(m["estado"], {}, dias=60, semilla=31337)["costo_total"]
    assert abs(a - b) < 1e-9, f"la misma semilla dio {a} y {b}"
    return f"dos corridas con semilla 31337 dan {a:,.4f} idéntico"


@prueba("Semillas distintas producen resultados distintos")
def t_estocastico():
    m = mundo()
    costos = [simular(m["estado"], {}, dias=60, semilla=400 + i * 97)["costo_total"]
              for i in range(6)]
    cv = float(np.std(costos) / np.mean(costos))
    assert cv > 0.01, f"variación entre réplicas casi nula (CV {cv:.4f}): el modelo no es estocástico"
    return f"6 réplicas, media {np.mean(costos):,.0f}, coef. de variación {cv * 100:.1f}%"


@prueba("Cobertura del intervalo de confianza")
def t_cobertura():
    m = mundo()
    verdadera = ejecutar_escenario(m["estado"], "Base", 60, 40)["costo_total"]
    aciertos, ensayos = 0, 10
    for i in range(ensayos):
        r = ejecutar_escenario(m["estado"], "Base", 60, 8, semilla_base=70000 + i * 3001)
        if r["costo_total_ic"][0] <= verdadera <= r["costo_total_ic"][1]:
            aciertos += 1
    cobertura = aciertos / ensayos * 100
    assert cobertura >= 70, f"cobertura de solo {cobertura:.0f}% frente al 95% nominal"
    return f"{aciertos}/{ensayos} intervalos contienen la media de referencia ({cobertura:.0f}%)"


@prueba("Los umbrales de fidelidad están por ENCIMA del piso irreducible")
def t_umbrales_alcanzables():
    """
    La prueba más importante del archivo.

    Los umbrales originales del proyecto (20% de MAPE, 2 días de diferencia en
    rupturas) estaban por debajo del ruido irreducible del propio motor, o sea
    que ningún modelo —ni siquiera el verdadero— podía cumplirlos. Esta prueba
    impide que vuelvan a fijarse umbrales imposibles.
    """
    m = mundo()
    estado = construir_estado_gemelo(m["catalogo"], m["movimientos"],
                                     dia_corte=FIDELIDAD["dias_ajuste"])
    pisos = pisos_irreducibles(estado, FIDELIDAD["dias_prueba"], n_pares=6)
    assert FIDELIDAD["umbral_mape_relativo"] >= 1.0, (
        "el umbral de MAPE relativo exige ser mejor que el propio motor")
    assert FIDELIDAD["umbral_ruptura_relativa"] >= 1.0, (
        "el umbral de rupturas exige ser mejor que el propio motor")
    assert FIDELIDAD["umbral_error_nivel"] > pisos["nivel"], (
        f"el umbral de nivel ({FIDELIDAD['umbral_error_nivel']}%) está bajo el piso "
        f"({pisos['nivel']:.1f}%): es inalcanzable")
    return (f"pisos: MAPE {pisos['mape']:.0f}%, nivel {pisos['nivel']:.1f}%, "
            f"rupturas {pisos['dias_ruptura']:.1f} d, órdenes {pisos['error_ordenes']:.1f}%. "
            f"Los umbrales relativos (≥1.0×) son alcanzables por construcción.")


@prueba("El gemelo pasa la retro-prueba de fidelidad")
def t_retro_prueba():
    m = mundo()
    r = retro_prueba(m["catalogo"], m["movimientos"], n_replicas=16)["resumen"]
    detalle = " · ".join(f"{c['Criterio']}: {c['Obtenido']}{c['unidad']} "
                         f"{'ok' if c['Cumple'] else 'FALLA'}" for c in r["criterios"])
    assert r["veredicto"], f"solo {r['n_criterios_cumplidos']}/{r['n_criterios']} — {detalle}"
    return f"{r['n_criterios_cumplidos']}/{r['n_criterios']} criterios — {detalle}"


@prueba("La descomposición de escenario suma coherentemente")
def t_descomposicion():
    m = mundo()
    d = descomponer_escenario(m["estado"], "S2", dias=90, n_replicas=5)
    assert d is not None, "S2 no devolvió descomposición"
    assert len(d["filas"]) == 2, f"S2 tiene 2 palancas, se descompusieron {len(d['filas'])}"
    dominante = d["filas"][0]
    assert abs(dominante["Participación (%)"]) > 40, "ninguna palanca domina claramente"
    return (f"{d['nombre']}: efecto conjunto {d['efecto_conjunto']:+,.0f}, suma de "
            f"partes {d['suma_individual']:+,.0f}, interacción {d['interaccion']:+,.0f} "
            f"({d['tipo_interaccion']}). Domina «{dominante['Palanca']}» con "
            f"{dominante['Participación (%)']}%")


@prueba("El tornado ordena por efecto real")
def t_tornado():
    m = mundo()
    tn = tornado_sensibilidad(m["estado"])
    amp = [f["Amplitud (S/.)"] for f in tn["filas"]]
    assert amp == sorted(amp, reverse=True), "el tornado no está ordenado"
    assert amp[0] > amp[-1] * 1.5, "todos los factores tienen efecto parecido"
    return (f"dominante: {tn['filas'][0]['Factor']} (S/. {amp[0]:,.0f}); "
            f"menor: {tn['filas'][-1]['Factor']} (S/. {amp[-1]:,.0f})")


# ==========================================================================
# COPILOTO DE ESCENARIOS
# --------------------------------------------------------------------------
# El copiloto es un módulo OPCIONAL: si sus paquetes no están instalados, estas
# pruebas se omiten en vez de tumbar la verificación del motor. Ninguna toca la
# red ni necesita clave: el modelo de lenguaje se sustituye por uno de mentira.
# ==========================================================================
def _copiloto_instalado():
    return importlib.util.find_spec("langgraph") is not None


OMITIDA = "omitida: el copiloto no está instalado (es un módulo opcional)"


@prueba("Un escenario ad-hoc no cae en el Base en silencio")
def t_adhoc_no_cae_en_base():
    m = mundo()
    cfg = {"codigo": "ADHOC", "nombre": "Demanda +20%", "pregunta": "¿Y si sube 20%?",
           "palancas": {"factor_demanda": 1.20},
           "mecanismo": "Palanca construida fuera de la lista declarada."}
    k = {"dias": 45, "n_replicas": 3, "semilla_base": 2026}

    base = ejecutar_escenario(m["estado"], "Base", **k)
    adhoc = ejecutar_escenario(m["estado"], cfg=cfg, **k)
    s1 = ejecutar_escenario(m["estado"], "S1", **k)

    assert adhoc["codigo"] == "ADHOC", (
        f"el código ad-hoc no se propagó: salió {adhoc['codigo']!r}")
    assert adhoc["costo_total"] != base["costo_total"], (
        "el escenario ad-hoc devolvió el costo del Base: cayó en el fallback")
    assert abs(adhoc["costo_total"] - s1["costo_total"]) < 1e-6, (
        f"con las mismas palancas que S1 el camino ad-hoc dio "
        f"{adhoc['costo_total']:,.0f} y el declarado {s1['costo_total']:,.0f}")

    return (f"ad-hoc S/. {adhoc['costo_total']:,.0f} ≠ Base S/. {base['costo_total']:,.0f}, "
            f"y reproduce S1 exactamente")


@prueba("El cerco numérico rechaza una cifra que el motor nunca produjo")
def t_cerco_rechaza_invencion():
    if not _copiloto_instalado():
        return OMITIDA
    from modulo_copiloto import guardia

    m = mundo()
    cfg = {"codigo": "ADHOC", "nombre": "Demanda +20%", "pregunta": "¿Y si sube 20%?",
           "palancas": {"factor_demanda": 1.20}, "mecanismo": "Prueba."}
    res = ejecutar_escenario(m["estado"], cfg=cfg, dias=45, n_replicas=3, semilla_base=2026)

    # El cerco se arma con lo que se le ENTREGA al redactor, no con los
    # diccionarios completos: un conjunto permitido grande es un cerco débil.
    entregado = (f"Costo total: S/. {res['costo_total']:,.0f}. "
                 f"Fill rate: {res['fill_rate']:.2f}%. "
                 f"Réplicas: {res['n_replicas']}. SKU: {res['n_sku']}.")
    permitidos = {v for _, v in guardia.extraer_numeros(entregado)}
    assert len(permitidos) < 40, (
        f"el conjunto permitido tiene {len(permitidos)} números: demasiado grande "
        f"para que el cerco discrimine")

    fiel = (f"El escenario cierra en S/. {res['costo_total']:,.0f} con un fill rate "
            f"de {res['fill_rate']:.2f}% sobre {res['n_sku']} SKU.")
    informe_fiel = guardia.verificar(fiel, permitidos)
    assert informe_fiel["ok"], (
        f"el cerco rechazó un texto fiel: {[i['literal'] for i in informe_fiel['intrusos']]}")

    inventado = fiel + " El ahorro frente al Base sería de S/. 8,437,219."
    informe_sucio = guardia.verificar(inventado, permitidos)
    assert not informe_sucio["ok"], "el cerco dejó pasar una cifra inventada"
    assert any("8,437,219" in i["literal"] for i in informe_sucio["intrusos"]), (
        f"el cerco rechazó, pero no señaló la cifra inventada: "
        f"{[i['literal'] for i in informe_sucio['intrusos']]}")

    return (f"{len(permitidos)} cifras permitidas; texto fiel aceptado "
            f"({informe_fiel['verificados']} trazadas) y la invención señalada")


@prueba("El grafo corrige un escenario inválido y nunca simula uno malo")
def t_grafo_corrige_escenario_invalido():
    if not _copiloto_instalado():
        return OMITIDA
    from modulo_copiloto.esquemas import EscenarioPropuesto, Traduccion
    from modulo_copiloto.grafo import construir_grafo, estado_inicial
    from langgraph.types import Command

    class _Estructurado:
        def __init__(self, guion):
            self.guion = guion

        def invoke(self, mensajes):
            # El grafo pide Traduccion; el escenario viaja dentro. Construirlo
            # aquí es lo que hace que uno inválido lance ValidationError, igual
            # que cuando el modelo real devuelve algo que no valida.
            return Traduccion(escenario=EscenarioPropuesto(**self.guion.pop(0)))

    class _Respuesta:
        def __init__(self, texto):
            self.content = texto

        @property
        def text(self):
            return self.content

    class _ModeloFalso:
        """No habla con nadie: devuelve un guion. Por eso esto corre sin red."""

        def __init__(self, guion):
            self.guion = list(guion)
            self.vistos = []

        def with_structured_output(self, esquema):
            return _Estructurado(self.guion)

        def invoke(self, mensajes):
            self.vistos.append(mensajes)
            return _Respuesta("Sin cifras que verificar.")

    texto = {"nombre": "Campaña de fiestas", "pregunta": "¿Aguantamos diciembre?",
             "mecanismo": "Sube la demanda media sin tocar la política para ver si el "
                          "stock de seguridad absorbe el pico."}
    # 0.97 no está tabulado en NIVELES_SERVICIO: politicas.py caería al z del SKU
    # y la política reportaría un nivel de servicio que no es el que simuló.
    invalido = {**texto, "nivel_servicio": 0.97}
    valido = {**texto, "factor_demanda": 1.20}

    m = mundo()
    modelo = _ModeloFalso([invalido, valido])
    app = construir_grafo(m["estado"], dias=30, replicas=2, modelo_chat=modelo,
                          semilla_base=2026)
    config = {"configurable": {"thread_id": "verificacion-corrige"}}
    est = app.invoke(estado_inicial("¿Y si la demanda sube 20%?"), config=config)

    assert est.get("__interrupt__"), (
        "el grafo no se detuvo a pedir confirmación: el humano quedó fuera del lazo")
    assert est["intentos_traduccion"] == 2, (
        f"no reintentó tras el rechazo: {est['intentos_traduccion']} intento(s)")
    assert est["propuesta"]["factor_demanda"] == 1.20, "no llegó al escenario válido"
    assert est["propuesta"]["nivel_servicio"] is None, (
        "el nivel de servicio no tabulado sobrevivió a la validación")

    est = app.invoke(Command(resume=True), config=config)
    assert est["resultados"]["propuesto"]["codigo"] != "Base", (
        "simuló el Base en vez del escenario propuesto")

    return ("rechazó nivel_servicio=0.97, corrigió en el segundo intento y esperó "
            "la confirmación del humano antes de simular")


@prueba("Sin clave de API el copiloto se apaga y el motor no se entera")
def t_copiloto_degrada_sin_clave():
    if not _copiloto_instalado():
        return OMITIDA
    import os

    from modulo_copiloto import proveedor

    variable = proveedor.variable_de_clave()
    if not variable:
        return f"proveedor '{proveedor.proveedor_declarado()}' es local: no aplica"

    previo = os.environ.pop(variable, None)
    try:
        listo, motivo = proveedor.disponible()
        assert not listo, "se declaró disponible sin clave configurada"
        assert variable in motivo, (
            f"el motivo no dice qué variable falta: {motivo!r}")
        # Lo que de verdad importa: el motor sigue corriendo.
        m = mundo()
        res = ejecutar_escenario(m["estado"], "Base", dias=30, n_replicas=2,
                                 semilla_base=2026)
        assert res["costo_total"] > 0, "el motor dejó de funcionar sin el copiloto"
    finally:
        if previo is not None:
            os.environ[variable] = previo

    return (f"sin {variable} el copiloto se apaga con un motivo accionable y el "
            f"motor sigue dando S/. {res['costo_total']:,.0f}")


@prueba("El copiloto no manda parámetros que el modelo ya rechaza")
def t_copiloto_sin_parametros_obsoletos():
    if not _copiloto_instalado():
        return OMITIDA
    import os

    from langchain_core.messages import HumanMessage

    from modulo_copiloto import proveedor

    variable = proveedor.variable_de_clave()
    previo = os.environ.get(variable) if variable else None
    if variable:
        os.environ[variable] = previo or "sk-ant-solo-para-construir-el-objeto"
    try:
        # Construir el modelo no toca la red: solo arma el objeto y su payload.
        m = proveedor.modelo()
        payload = m._get_request_payload([HumanMessage("hola")])
    finally:
        if variable:
            if previo is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previo

    # Los modelos Claude 4.7+ devuelven 400 si reciben `temperature`. El detalle
    # que cuesta encontrar: langchain-anthropic no lo manda como argumento
    # nombrado sino DENTRO de extra_body, así que buscarlo en la raíz del
    # payload no lo encuentra y la prueba pasaría en falso.
    extra = payload.get("extra_body") or {}
    for obsoleto in ("temperature", "top_p", "top_k"):
        assert obsoleto not in payload, (
            f"`{obsoleto}` viaja en la raíz del payload: los modelos actuales "
            f"responden 400 invalid_request_error")
        assert obsoleto not in extra, (
            f"`{obsoleto}` viaja dentro de extra_body, que el SDK fusiona al "
            f"JSON tal cual: el API lo va a rechazar con 400")

    return (f"payload limpio para {payload.get('model')}: sin temperature/top_p/top_k "
            f"ni en la raíz ni en extra_body")


@prueba("Ante una pregunta que no es un escenario, el copiloto NO inventa uno")
def t_copiloto_rechaza_en_vez_de_inventar():
    if not _copiloto_instalado():
        return OMITIDA
    from modulo_copiloto.esquemas import Traduccion, catalogo_palancas
    from modulo_copiloto.grafo import construir_grafo, estado_inicial

    class _Estructurado:
        def __init__(self, guion):
            self.guion = guion

        def invoke(self, mensajes):
            return Traduccion(**self.guion.pop(0))

    class _ModeloFalso:
        def __init__(self, guion):
            self.guion = list(guion)

        def with_structured_output(self, esquema):
            assert esquema is Traduccion, (
                f"el grafo pide {esquema.__name__} en vez de Traduccion: sin el canal "
                f"de rechazo, el modelo queda obligado a inventar un escenario")
            return _Estructurado(self.guion)

        def invoke(self, mensajes):
            raise AssertionError("no debía llegar a redactar: no hay nada que redactar")

    m = mundo()
    rechazo = {"motivo": "Eso es un saludo, no una pregunta sobre el inventario. "
                         "Puedo simular qué pasa si cambia la demanda, el proveedor, "
                         "el nivel de servicio, la política de compra o el presupuesto."}
    app = construir_grafo(m["estado"], dias=30, replicas=2,
                          modelo_chat=_ModeloFalso([rechazo]), semilla_base=2026)
    est = app.invoke(estado_inicial("hola que se puede hacer aqui?"),
                     config={"configurable": {"thread_id": "verificacion-rechazo"}})

    assert est.get("no_traducible"), "no registró el rechazo"
    assert est.get("propuesta") is None, (
        "inventó un escenario para una pregunta que no lo es — es el fallo que hace "
        "dudar de las cifras")
    assert not est.get("__interrupt__"), "pidió confirmar un escenario que no existe"
    assert est.get("resultados") is None, "simuló algo que nadie pidió"
    assert est["intentos_traduccion"] == 1, (
        f"reintentó {est['intentos_traduccion']} veces un rechazo legítimo: insistir "
        f"solo empuja al modelo a fabricar un escenario")

    # El esquema no admite las dos salidas a la vez ni ninguna.
    for invalido in ({}, {"motivo": "x" * 20, "escenario": None, "extra": 1}):
        try:
            Traduccion(**invalido)
            raise AssertionError(f"Traduccion aceptó {invalido!r}")
        except AssertionError:
            raise
        except Exception:
            pass

    palancas = catalogo_palancas()
    assert len(palancas) == 8, f"el catálogo muestra {len(palancas)} palancas, no 8"

    return (f"rechazó sin inventar, sin simular y sin reintentar; ofrece las "
            f"{len(palancas)} palancas reales en su lugar")


@prueba("Langflow traduce en el grafo y el modelo directo no calcula el escenario")
def t_langflow_traduce_en_el_grafo():
    if not _copiloto_instalado():
        return OMITIDA
    from modulo_copiloto.grafo import construir_grafo, estado_inicial
    from modulo_copiloto.langflow_cliente import interpretar_plano

    rechazo = interpretar_plano({
        "es_simulable": False,
        "motivo": "Un saludo no mueve ninguna palanca del inventario.",
    })
    assert rechazo.escenario is None, "un rechazo plano se convirtió en escenario"
    assert rechazo.motivo.startswith("Un saludo")

    # Un mecanismo largo no es un escenario inválido: recortarlo evita el
    # reintento que le devolvía a Langflow el aviso del validador y ese aviso
    # terminaba escrito en la pregunta que ve la persona.
    largo = ("Se alarga el tiempo de entrega y también su dispersión. " * 20)
    sucio = (
        "¿Y si el proveedor se demora el triple?\n\n"
        "Tu propuesta anterior fue RECHAZADA por el validador del motor:\n"
        "- mecanismo: String should have at most 400 characters")
    arreglado = interpretar_plano({
        "es_simulable": True,
        "nombre": "Proveedor lento e impredecible",
        "pregunta": sucio,
        "mecanismo": largo,
        "factor_lead_time": "3",
        "factor_desv_lead_time": "2",
    }, pregunta_original="¿Y si el proveedor se demora el triple y se vuelve impredecible?")
    assert len(arreglado.escenario.mecanismo) <= 400
    assert len(arreglado.escenario.mecanismo) >= 20
    assert "RECHAZADA" not in arreglado.escenario.pregunta
    assert arreglado.escenario.factor_lead_time == 3.0
    assert arreglado.escenario.factor_desv_lead_time == 2.0

    texto = {
        "es_simulable": True,
        "motivo": "",
        "nombre": "Campaña de fiestas",
        "pregunta": "¿Aguantamos si la demanda sube?",
        "mecanismo": "Sube la demanda media y deja la política quieta para ver el pico.",
    }
    invalido = {**texto, "nivel_servicio": "0.97"}
    valido = {**texto, "factor_demanda": "1.20", "nivel_servicio": ""}
    guion = [invalido, valido]
    vistos = []

    def traductor(pregunta, errores):
        vistos.append(list(errores))
        return interpretar_plano(guion.pop(0))

    class _ModeloQueNoTraduce:
        def with_structured_output(self, esquema):
            raise AssertionError("el modelo directo no debe traducir cuando hay Langflow")

        def invoke(self, mensajes):
            raise AssertionError("no debía redactar: la simulación espera confirmación")

    m = mundo()
    app = construir_grafo(m["estado"], dias=30, replicas=2,
                          modelo_chat=_ModeloQueNoTraduce(), traductor=traductor,
                          semilla_base=2026)
    est = app.invoke(estado_inicial("¿Y si la demanda sube 20%?"),
                     config={"configurable": {"thread_id": "verificacion-langflow"}})

    assert est.get("__interrupt__"), "no se detuvo a pedir confirmación"
    assert est["propuesta"]["factor_demanda"] == 1.20
    assert est["propuesta"]["nivel_servicio"] is None
    assert len(vistos) == 2, f"no reintentó el JSON inválido: {len(vistos)} llamadas"
    assert vistos[0] == []
    assert vistos[1], "el segundo intento no recibió el error del validador"
    assert "0.97" in vistos[1][0]

    return ("rechazó nivel_servicio=0.97 devuelto por el traductor, corrigió "
            "en el segundo JSON y esperó al humano; el modelo directo no tradujo")


@prueba("El ingenuo semanal copia la última semana y un acierto tiene MASE cero")
def t_crisp_metricas():
    """No ajusta Prophet ni LightGBM: solo las definiciones que el informe usa."""
    from modulo_simulacion.crisp_dm import (
        diebold_mariano, mase, pronostico_ingenuo, wape,
    )

    y = np.arange(1, 15, dtype=float)
    pred = pronostico_ingenuo(y, horizonte=7)
    assert np.allclose(pred, [8, 9, 10, 11, 12, 13, 14])
    tren = np.array([1, 2, 4, 1, 3, 5, 2, 2, 1, 3, 2, 4, 6, 1], dtype=float)
    acierto = np.full(7, 9.0)
    assert mase(acierto, acierto, tren) == 0.0
    assert np.isnan(wape(np.zeros(4), np.ones(4)))
    igual = diebold_mariano(np.ones(30), np.ones(30))
    assert igual["p_valor"] > 0.05
    return "ingenuo repite la semana, MASE 0 si acierta, WAPE vacío si no hubo venta"


# ==========================================================================
def main():
    pruebas = [t_abc, t_censura_existe, t_censura_corregida, t_ss_lead_time,
               t_eficiencia, t_dummy, t_sustituto, t_orden_politicas,
               t_reproducible, t_estocastico, t_cobertura, t_umbrales_alcanzables,
               t_retro_prueba, t_descomposicion, t_tornado,
               t_adhoc_no_cae_en_base, t_cerco_rechaza_invencion,
               t_grafo_corrige_escenario_invalido, t_copiloto_degrada_sin_clave,
               t_copiloto_sin_parametros_obsoletos,
               t_copiloto_rechaza_en_vez_de_inventar,
               t_langflow_traduce_en_el_grafo, t_crisp_metricas]

    print("=" * 78)
    print("  VERIFICACIÓN DEL MOTOR — GEMELO DIGITAL DE INVENTARIO SOAZ")
    print("=" * 78)
    for p in pruebas:
        p()

    for nombre, ok, detalle, dur in RESULTADOS:
        print(f"\n[{'PASA' if ok else 'FALLA'}] {nombre}  ({dur:.1f}s)")
        print(f"        {detalle}")

    fallidas = [n for n, ok, _, _ in RESULTADOS if not ok]
    print("\n" + "=" * 78)
    if fallidas:
        print(f"  {len(fallidas)} de {len(RESULTADOS)} pruebas FALLARON: {', '.join(fallidas)}")
        return 1
    print(f"  Las {len(RESULTADOS)} pruebas pasaron.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
