# modulo_copiloto/grafo.py
"""
El grafo del copiloto: de una pregunta en español a un escenario simulado.

QUÉ HACE Y QUÉ NO
------------------
    traducir  ──┐ (ciclo: el validador devuelve el error y el modelo corrige)
        │  ◄────┘
    confirmar   (interrupt: el humano aprueba las palancas antes de simular)
        │
    simular     (motor real: Base y propuesto, con réplicas e IC)
        │
    redactar  ──┐
        │  ◄────┤ (ciclo: el cerco nombra las cifras inventadas y se reescribe)
     cercar  ───┘

El modelo de lenguaje interviene en dos nodos y en ninguno produce un número:
en `traducir` mapea texto a palancas tipadas, y en `redactar` escribe sobre
resultados que ya existen. Todo lo demás es Python determinista.

Si se inyecta `traductor`, ese callable ocupa el lugar del modelo en `traducir`
—es el flujo de Langflow—. `redactar` sigue usando el modelo directo. Las
cifras las produce el motor en los dos casos.

POR QUÉ NO HAY UN NODO `validar` SEPARADO
------------------------------------------
El plan original lo tenía. Al escribirlo quedó claro que no haría nada: la
validación ocurre al construir el EscenarioPropuesto, dentro de `traducir`.
Un nodo aparte que solo reconstruyera el mismo objeto sería un cuadrito más en
el diagrama y ni una comprobación más en el código. El ciclo de corrección
existe igual, y es real.

POR QUÉ EL ESTADO DEL GEMELO NO ENTRA AL STATE DEL GRAFO
---------------------------------------------------------
Son 24 SKU por 25 campos, y LangGraph serializa el state completo en el
checkpoint en cada paso. Va por closure: el grafo lleva la conversación, el
mundo se queda afuera.

POR QUÉ SE PUEDE INYECTAR EL MODELO
------------------------------------
`construir_grafo(..., modelo_chat=...)` acepta cualquier objeto con la interfaz
de un chat model. Es lo que permite probar el grafo entero —los dos ciclos, el
interrupt, el cerco— con un modelo de mentira, sin clave y sin red. Un módulo
que solo se puede probar con la API viva es un módulo que no se prueba.
"""
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from modulo_simulacion.escenarios import (
    distinguible_del_base, ejecutar_escenario, fila_comparativa,
)
from modulo_simulacion.parametros import NEGOCIO

from . import guardia
from .esquemas import EscenarioPropuesto, NIVELES_VALIDOS, Traduccion

# El primer intento más dos correcciones. Si en tres no salió un escenario
# válido, el problema no es el fraseo: la pregunta no se puede expresar con
# las palancas que hay, y hay que decirlo en vez de seguir insistiendo.
MAX_INTENTOS_TRADUCCION = 3

# El primero más una reescritura. Si el cerco la rechaza dos veces, se descarta
# la envoltura y manda la narrativa determinista. No hay bucle infinito.
MAX_INTENTOS_REDACCION = 2


INSTRUCCION_TRADUCCION = f"""\
Eres el traductor de un gemelo digital de inventario de una comercializadora de
abarrotes en Otuzco, La Libertad. Conviertes la pregunta del negocio en un
escenario ejecutable por el motor de simulación.

NO CALCULAS NADA. No estimas resultados, no das cifras, no opinas sobre qué
pasaría. Eliges qué palancas mover y cuánto, nada más.

El motor simula eventos discretos sobre los SKU de clase A: demanda diaria,
revisión de inventario, llegada del proveedor y vencimiento por lote. Las
únicas palancas son las del esquema.

DEVUELVES UNA DE DOS COSAS, NUNCA LAS DOS:

  `escenario`  cuando la pregunta describe un what-if que las palancas pueden
               expresar.

  `motivo`     cuando NO. Un saludo, una pregunta sobre la herramienta misma
               («¿qué puedo preguntar aquí?»), una pregunta de conocimiento
               general, o algo que el motor no modela —precios, rutas,
               proveedores nuevos, más almacenes, contratar personal—.

Rechazar es una respuesta correcta y esperada, no un fracaso. NUNCA propongas
un escenario de consuelo: inventar una pregunta de negocio que el usuario no
hizo es peor que decirle que no se puede. Si inventas la pregunta, con razón
va a dudar también de las cifras.

Reglas de traducción:
- «sube/baja un X%» la demanda -> factor_demanda = 1 + X/100.
- «se demora el doble» -> factor_lead_time = 2.0. Si además sugiere que el
  proveedor se volvió impredecible, mueve también factor_desv_lead_time.
- «nivel de servicio del 99%» -> nivel_servicio = 0.99. Solo existen
  {", ".join(str(n) for n in NIVELES_VALIDOS)}: elige el más cercano a la intención.
- «comprar una vez por semana» -> tipo_politica = "(R,S)" con
  periodo_revision_dias = 7.
- «solo me alcanza para el 85%» -> fraccion_presupuesto = 0.85.
- Mueve SOLO las palancas que la pregunta pide. Una palanca de más es un
  escenario que responde otra pregunta.

En `mecanismo` explica, en dos oraciones y como máximo 350 caracteres, qué
parte del modelo se activa y por qué eso responde la pregunta. Todavía no se
ha simulado nada: ahí no van resultados.

`nombre` cabe en 80 caracteres. `pregunta` es una sola oración con la pregunta
del negocio: no copies instrucciones, ni un error del validador, ni este texto.
"""


INSTRUCCION_REDACCION = """\
Eres el redactor del gemelo digital de inventario de la Comercializadora SOAZ.
Te entregan resultados YA CALCULADOS y los escribes para la persona que decide
las compras.

REGLA ABSOLUTA: no puedes escribir ninguna cifra que no esté en los resultados
entregados. No sumes, no restes, no promedies, no estimes, no conviertas
unidades. Si mencionas un número, cópialo tal cual. Un texto con una sola cifra
inventada se descarta entero y no llega a pantalla.

Formato: separador de miles con coma y decimal con punto, como «S/. 11,506» y
«97.91%», igual que el resto de la aplicación.

Escribe dos o tres párrafos cortos, en español neutro y profesional:
1. Qué pasó bajo el escenario frente al Base.
2. Si la diferencia es distinguible del ruido del modelo, y qué implica.
3. La advertencia que corresponda para quien va a decidir con esto.

No repitas la interpretación determinista que ya acompaña a la tabla: agrega la
lectura de negocio, no el mismo texto.
"""


class EstadoCopiloto(TypedDict):
    """Lo que viaja por el grafo. El mundo (el estado del gemelo) no está aquí."""
    pregunta: str
    propuesta: Optional[dict]
    no_traducible: Optional[str]
    errores: list
    intentos_traduccion: int
    confirmado: Optional[bool]
    resultados: Optional[dict]
    interp: Optional[dict]
    envoltura: Optional[str]
    intentos_redaccion: int
    cerco: Optional[dict]
    fallo: Optional[str]


def estado_inicial(pregunta):
    return {
        "pregunta": pregunta, "propuesta": None, "no_traducible": None, "errores": [],
        "intentos_traduccion": 0, "confirmado": None, "resultados": None,
        "interp": None, "envoltura": None, "intentos_redaccion": 0,
        "cerco": None, "fallo": None,
    }


def _es_validacion(excepcion):
    """¿El modelo produjo algo mal formado (corregible) o se cayó la red?"""
    vista = excepcion
    for _ in range(5):
        if isinstance(vista, ValidationError):
            return True
        vista = getattr(vista, "__cause__", None) or getattr(vista, "__context__", None)
        if vista is None:
            return False
    return False


def _legible(excepcion):
    """El error de validación en las palabras con que se le puede pedir arreglo."""
    if isinstance(excepcion, ValidationError):
        partes = []
        for error in excepcion.errors():
            campo = ".".join(str(x) for x in error.get("loc", ())) or "(escenario)"
            partes.append(f"- {campo}: {error.get('msg', '')}")
        return "\n".join(partes)
    return str(excepcion)


def _texto(respuesta):
    """El texto de una respuesta, venga como string o como bloques de contenido."""
    texto = getattr(respuesta, "text", None)
    if callable(texto):
        texto = texto()
    if isinstance(texto, str) and texto:
        return texto
    contenido = getattr(respuesta, "content", "")
    if isinstance(contenido, str):
        return contenido
    partes = []
    for bloque in contenido or []:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif isinstance(bloque, dict) and bloque.get("type") == "text":
            partes.append(bloque.get("text", ""))
    return "".join(partes)


def texto_final(estado):
    """
    Qué se muestra al usuario, y si la envoltura del modelo sobrevivió.

    La narrativa determinista se muestra SIEMPRE. La envoltura solo si el cerco
    la aprobó. Esa asimetría es el módulo entero.
    """
    cerco = estado.get("cerco") or {}
    aprobada = bool(estado.get("envoltura")) and bool(cerco.get("ok"))
    return {
        "interp": estado.get("interp"),
        "envoltura": estado["envoltura"] if aprobada else None,
        "envoltura_descartada": estado.get("envoltura") if not aprobada else None,
        "cerco": cerco,
    }


def construir_grafo(estado_gemelo, dias, replicas, modelo_chat=None,
                    semilla_base=None, interpretar=None, checkpointer=None,
                    traductor=None):
    """
    Compila el grafo del copiloto sobre un estado del gemelo concreto.

    `estado_gemelo`, `dias` y `replicas` van por closure. `modelo_chat` se puede
    inyectar para probar sin red; si es None se construye desde proveedor.py la
    primera vez que haga falta. `interpretar` permite cambiar de dónde sale la
    narrativa determinista; por defecto usa narrativa.interpretar_corrida.

    `traductor`, si viene, reemplaza al modelo solo en el nodo traducir. Recibe
    (pregunta, errores) y devuelve un Traduccion. Es el punto por donde entra
    Langflow. Sin él, traducir sigue llamando a `modelo_chat`.
    """
    contenedor = {"modelo": modelo_chat}

    def _modelo():
        if contenedor["modelo"] is None:
            from . import proveedor
            contenedor["modelo"] = proveedor.modelo()
        return contenedor["modelo"]

    def _interpretar(resultado):
        if interpretar is not None:
            return interpretar(resultado)
        from modulo_simulacion import narrativa
        return narrativa.interpretar_corrida(resultado)

    # ------------------------------------------------------------------
    def _traducir(estado):
        intentos = estado.get("intentos_traduccion", 0) + 1
        try:
            if traductor is not None:
                # Langflow ya tiene el sistema en su lienzo. Aquí solo viajan
                # la pregunta y, si el validador rechazó el intento anterior,
                # el error que tiene que corregir.
                traduccion = traductor(estado["pregunta"], list(estado.get("errores") or []))
            else:
                mensajes = [SystemMessage(INSTRUCCION_TRADUCCION),
                            HumanMessage(estado["pregunta"])]
                for error in estado.get("errores") or []:
                    mensajes.append(HumanMessage(
                        f"Tu propuesta anterior fue RECHAZADA por el validador del motor:\n"
                        f"{error}\n"
                        f"Corrige exactamente eso y vuelve a proponer el escenario."))
                estructurado = _modelo().with_structured_output(Traduccion)
                traduccion = estructurado.invoke(mensajes)
        except Exception as excepcion:
            if _es_validacion(excepcion):
                # Corregible: el ciclo devuelve el error al modelo.
                return {"propuesta": None,
                        "errores": list(estado.get("errores") or []) + [_legible(excepcion)],
                        "intentos_traduccion": intentos}
            # Red, autenticación, cuota: reintentar no arregla nada.
            quien = "Langflow" if traductor is not None else "el modelo"
            return {"propuesta": None, "intentos_traduccion": intentos,
                    "fallo": f"No se pudo consultar a {quien}: {excepcion}"}

        if traduccion.escenario is None:
            # Rechazo legítimo: la pregunta no es un what-if simulable. NO se
            # reintenta — insistir solo empuja al modelo a inventar algo.
            return {"propuesta": None, "no_traducible": traduccion.motivo,
                    "errores": [], "intentos_traduccion": intentos}

        return {"propuesta": traduccion.escenario.model_dump(), "errores": [],
                "no_traducible": None, "intentos_traduccion": intentos}

    def _ruta_traduccion(estado):
        if estado.get("fallo"):
            return END
        if estado.get("no_traducible"):
            return END
        if estado.get("propuesta"):
            return "confirmar"
        if estado.get("intentos_traduccion", 0) >= MAX_INTENTOS_TRADUCCION:
            return END
        return "traducir"

    # ------------------------------------------------------------------
    def _confirmar(estado):
        propuesta = EscenarioPropuesto(**estado["propuesta"])
        decision = interrupt({
            "tipo": "confirmar_escenario",
            "nombre": propuesta.nombre,
            "pregunta": propuesta.pregunta,
            "mecanismo": propuesta.mecanismo,
            "filas": propuesta.filas_confirmacion(),
        })
        return {"confirmado": bool(decision)}

    def _ruta_confirmacion(estado):
        return "simular" if estado.get("confirmado") else END

    # ------------------------------------------------------------------
    def _simular(estado):
        propuesta = EscenarioPropuesto(**estado["propuesta"])
        comunes = {"dias": dias, "n_replicas": replicas, "semilla_base": semilla_base}
        base = ejecutar_escenario(estado_gemelo, "Base", **comunes)
        propuesto = ejecutar_escenario(estado_gemelo, cfg=propuesta.a_cfg(), **comunes)
        meses = dias / NEGOCIO["dias_por_mes"]
        return {"resultados": {
            "base": base,
            "propuesto": propuesto,
            "tabla": [fila_comparativa(base, base, meses),
                      fila_comparativa(propuesto, base, meses)],
            "distinguible": distinguible_del_base(propuesto, base),
        }}

    # ------------------------------------------------------------------
    def _redactar(estado):
        resultados = estado["resultados"]
        interp = _interpretar(resultados["propuesto"])
        intentos = estado.get("intentos_redaccion", 0) + 1

        mensajes = [SystemMessage(INSTRUCCION_REDACCION),
                    HumanMessage(_contexto(estado, resultados))]
        cerco_previo = estado.get("cerco")
        if cerco_previo and not cerco_previo.get("ok"):
            cifras = ", ".join(i["literal"] for i in cerco_previo["intrusos"])
            mensajes.append(HumanMessage(
                f"Tu texto anterior fue RECHAZADO por el cerco numérico: las cifras "
                f"{cifras} no aparecen en ningún resultado entregado. Reescríbelo "
                f"usando solo cifras que estén en los datos de arriba."))

        try:
            respuesta = _modelo().invoke(mensajes)
        except Exception as excepcion:
            return {"interp": interp, "envoltura": None,
                    "intentos_redaccion": intentos,
                    "fallo": f"No se pudo redactar la respuesta: {excepcion}"}

        return {"interp": interp, "envoltura": _texto(respuesta),
                "intentos_redaccion": intentos}

    def _cercar(estado):
        """
        Se cerca contra EL CONTEXTO QUE SE LE ENTREGÓ, no contra los resultados
        completos.

        La primera versión admitía todo número que apareciera en los
        diccionarios de resultado: 231 valores, incluidos costos por SKU y
        desviaciones que el modelo nunca vio. Con un conjunto así, una cifra
        inventada como «14.6» encuentra por casualidad un round(14.5875, 1) y
        pasa. Medido: 9.1% de falso positivo por cifra.

        El modelo solo puede citar lo que se le dio, y lo que se le dio es
        exactamente _contexto(). Cercar contra eso es a la vez más estricto y
        más fiel a la regla que se le enunció. Las operaciones derivadas —restar
        dos costos para dar el ahorro— quedan prohibidas, que es justo lo que la
        instrucción de redacción ya le pide.
        """
        contexto = _contexto(estado, estado["resultados"])
        permitidos = {valor for _, valor in guardia.extraer_numeros(contexto)}
        permitidos |= guardia.numeros_permitidos(
            EscenarioPropuesto(**estado["propuesta"]).a_palancas())
        return {"cerco": guardia.verificar(estado.get("envoltura") or "", permitidos)}

    def _ruta_cerco(estado):
        if estado.get("fallo"):
            return END
        if (estado.get("cerco") or {}).get("ok"):
            return END
        if estado.get("intentos_redaccion", 0) >= MAX_INTENTOS_REDACCION:
            return END          # se descarta la envoltura; manda la determinista
        return "redactar"

    # ------------------------------------------------------------------
    grafo = StateGraph(EstadoCopiloto)
    grafo.add_node("traducir", _traducir)
    grafo.add_node("confirmar", _confirmar)
    grafo.add_node("simular", _simular)
    grafo.add_node("redactar", _redactar)
    grafo.add_node("cercar", _cercar)

    grafo.add_edge(START, "traducir")
    grafo.add_conditional_edges("traducir", _ruta_traduccion,
                                {"traducir": "traducir", "confirmar": "confirmar", END: END})
    grafo.add_conditional_edges("confirmar", _ruta_confirmacion,
                                {"simular": "simular", END: END})
    grafo.add_edge("simular", "redactar")
    grafo.add_edge("redactar", "cercar")
    grafo.add_conditional_edges("cercar", _ruta_cerco,
                                {"redactar": "redactar", END: END})

    return grafo.compile(checkpointer=checkpointer or InMemorySaver())


def _contexto(estado, resultados):
    """
    Los resultados que el redactor puede citar, y nada más.

    Se arma a mano en vez de volcar los diccionarios completos: lo que no entra
    aquí es lo que el modelo no puede mencionar, y las series diarias no entran.
    """
    base, propuesto = resultados["base"], resultados["propuesto"]
    lineas = [
        f"PREGUNTA DEL USUARIO: {estado['pregunta']}",
        "",
        f"ESCENARIO SIMULADO: {propuesto['nombre']}",
        f"Mecanismo declarado: {propuesto['mecanismo']}",
        f"Palancas movidas: {EscenarioPropuesto(**estado['propuesta']).a_palancas()}",
        "",
        "RESULTADOS (cifras exactas; no puedes escribir ninguna otra):",
    ]
    for etiqueta, r in (("Base", base), ("Escenario propuesto", propuesto)):
        lineas += [
            f"  {etiqueta}:",
            f"    Costo total: S/. {r['costo_total']:,.0f} "
            f"(IC 95%: [{r['costo_total_ic'][0]:,.0f}, {r['costo_total_ic'][1]:,.0f}])",
            f"    Fill rate: {r['fill_rate']:.2f}%",
            f"    Capital inmovilizado: S/. {r['capital_inmovilizado']:,.0f}",
            f"    Órdenes: {r['ordenes']:.1f}",
            f"    SKU con ruptura: {r['sku_con_ruptura']:.1f} de {r['n_sku']}",
            f"    Unidades perdidas: {r['unidades_perdidas']:.0f}",
            f"    Unidades por merma: {r['unidades_merma']:.0f}",
        ]
    lineas += [
        "",
        f"Réplicas por escenario: {propuesto['n_replicas']}. "
        f"Días simulados: {propuesto['dias']}.",
        f"¿La diferencia de costo es distinguible del ruido? "
        f"{'SÍ' if resultados['distinguible'] else 'NO'} "
        f"(los intervalos de confianza "
        f"{'no se solapan' if resultados['distinguible'] else 'se solapan'}).",
    ]
    return "\n".join(lineas)
