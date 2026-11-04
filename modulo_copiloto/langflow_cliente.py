# modulo_copiloto/langflow_cliente.py
"""
Cliente del flujo «Traductor de escenarios SOAZ» en Langflow.

QUÉ HACE Y QUÉ NO
------------------
Langflow recibe la pregunta en español y devuelve un JSON plano de palancas.
Este módulo lo convierte en un `Traduccion` tipado. No simula, no redacta y
no deja pasar un escenario que `esquemas.py` rechazaría: si el JSON no valida,
la excepción sube al grafo y el ciclo de corrección se la devuelve a Langflow.

El flujo se crea con `crear_flujo_traductor.py`. Sin LANGFLOW_FLOW_ID el
copiloto sigue traduciendo con el modelo directo de proveedor.py: Langflow es
un reemplazo del nodo traducir, no un requisito del gemelo.
"""
import ast
import gzip
import json
import re
import urllib.error
import urllib.request
import uuid

import config

from .esquemas import EscenarioPropuesto, Traduccion

# El agente de Langflow 1.12 arma su salida estructurada con una tabla plana
# (str, bool, float). Un EscenarioPropuesto anidado no cabe ahí, así que las
# palancas viajan como texto y este módulo las convierte. Cadena vacía = palanca
# que la pregunta no pidió.
ESQUEMA_LANGFLOW = [
    {
        "name": "es_simulable",
        "description": "true solo si la pregunta es un what-if que las palancas "
                       "pueden expresar. false para un saludo, una pregunta sobre "
                       "la herramienta, o algo que el motor no modela.",
        "type": "bool",
        "multiple": "False",
    },
    {
        "name": "motivo",
        "description": "Si es_simulable es false, explicación en español de al "
                       "menos 15 caracteres y sin proponer un escenario de consuelo. "
                       "Si es true, cadena vacía.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "nombre",
        "description": "Nombre corto del escenario, 3 a 80 caracteres. Vacío si no es simulable.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "pregunta",
        "description": "Una sola oración, máximo 200 caracteres, con la pregunta del negocio. "
                       "No copies errores del validador ni instrucciones. Vacío si no es simulable.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "mecanismo",
        "description": "Dos oraciones, máximo 350 caracteres: qué parte del modelo se "
                       "activa y por qué. Sin cifras de resultados. Vacío si no es simulable.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "factor_demanda",
        "description": "Multiplicador de la demanda media, entre 0.1 y 5. 1.20 = sube 20%. "
                       "Vacío si la pregunta no habla de la demanda.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "factor_lead_time",
        "description": "Multiplicador del tiempo de entrega, entre 0.1 y 10. Vacío si no aplica.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "factor_desv_lead_time",
        "description": "Multiplicador de la variabilidad del tiempo de entrega, entre 0.1 y 10. "
                       "Vacío si no aplica.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "nivel_servicio",
        "description": "Solo 0.90, 0.95, 0.975 o 0.99. Vacío si la pregunta no cambia el servicio.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "tipo_politica",
        "description": "Solo (s,Q), (R,S) o (R,s,Q). Vacío si no cambia la política.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "periodo_revision_dias",
        "description": "Entero de 1 a 90. Vacío si no aplica. No lo uses con (s,Q).",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "fraccion_presupuesto",
        "description": "Fracción financiable, mayor que 0 y hasta 2. Vacío si no aplica. "
                       "No la combines con presupuesto_mensual.",
        "type": "str",
        "multiple": "False",
    },
    {
        "name": "presupuesto_mensual",
        "description": "Presupuesto mensual de compra en soles. Vacío si no aplica.",
        "type": "str",
        "multiple": "False",
    },
]

INSTRUCCION_CAMPOS = """\
La salida es JSON estructurado, un objeto plano y nada más.

- es_simulable=true solo cuando las palancas alcanzan para la pregunta.
- Si es_simulable=false, llena motivo y deja el resto de campos en cadena vacía.
- Si es_simulable=true, motivo va vacío y llenas solo las palancas que la
  pregunta pide. Una palanca de más responde otra pregunta.
- Los números van con punto decimal, sin unidades y sin texto alrededor:
  1.30, 0.99, 7. Lo que no se mueve se queda en cadena vacía, no en 0 ni en 1.
- `mecanismo`: dos oraciones, máximo 350 caracteres. Si te pasas, el validador
  rechaza la propuesta entera.
- `pregunta`: una oración, máximo 200 caracteres. Es la pregunta del negocio,
  nunca el texto de un rechazo anterior.
- `nombre`: máximo 80 caracteres.
"""

_VACIO = {"", "none", "null", "n/a", "na", "-"}
_FLOTANTES = (
    "factor_demanda", "factor_lead_time", "factor_desv_lead_time",
    "nivel_servicio", "fraccion_presupuesto", "presupuesto_mensual",
)


def activo():
    """Hay un flujo de Langflow declarado y una clave para llamarlo."""
    return bool(config.LANGFLOW_FLOW_ID and config.LANGFLOW_API_KEY and config.LANGFLOW_URL)


def traductor_si_configurado():
    """
    El callable que el grafo inyecta en `traducir`, o None para dejar el modelo directo.

    La firma es (pregunta, errores) -> Traduccion. Los errores son los rechazos
    del validador en intentos anteriores de la misma pregunta.
    """
    if not activo():
        return None

    def _traducir(pregunta, errores):
        return traducir(pregunta, errores)

    return _traducir


def armar_entrada(pregunta, errores):
    """La pregunta más, si los hay, los rechazos que Langflow tiene que corregir."""
    partes = [(pregunta or "").strip()]
    for error in errores or []:
        partes.append(
            "El validador del motor rechazó un campo de la propuesta anterior:\n"
            f"{error}\n"
            "Corrige solo ese campo. `pregunta` sigue siendo la pregunta original "
            "del negocio, en una oración. `mecanismo` en máximo 350 caracteres. "
            "No copies este aviso dentro de la respuesta.")
    return "\n\n".join(partes)


def traducir(pregunta, errores=None):
    """Llama al flujo y devuelve un Traduccion. Puede lanzar ValidationError."""
    if not activo():
        raise RuntimeError(
            "Langflow no está configurado. Hacen falta LANGFLOW_URL, "
            "LANGFLOW_FLOW_ID y LANGFLOW_API_KEY.")

    texto = _ejecutar(armar_entrada(pregunta, errores))
    return interpretar_plano(_como_dict(texto), pregunta_original=pregunta)


def interpretar_plano(datos, pregunta_original=None):
    """
    Un dict con los campos de ESQUEMA_LANGFLOW, ya validado contra el motor.

    Un texto demasiado largo no es un escenario distinto: se recorta aquí, en
    el último punto o espacio que cabe, y no se vuelve a llamar a Langflow.
    Reintentar por el largo solo le devolvía al modelo el aviso del validador,
    y ese aviso terminaba copiado en la pregunta que ve la persona.

    Las palancas vacías se omiten. Un nivel de servicio fuera de tabla, una
    política inexistente o un escenario sin palancas sí revientan: el grafo le
    devuelve ese error a Langflow.
    """
    if not isinstance(datos, dict):
        raise ValueError(f"La salida de Langflow no es un objeto JSON: {type(datos).__name__}")

    if not _flag(datos.get("es_simulable")):
        motivo = _recortar(datos.get("motivo"), 600)
        return Traduccion(motivo=motivo or None)

    palancas = {}
    for clave in _FLOTANTES:
        numero = _flotante(datos.get(clave))
        if numero is not None:
            palancas[clave] = numero
    periodo = _entero(datos.get("periodo_revision_dias"))
    if periodo is not None:
        palancas["periodo_revision_dias"] = periodo
    politica = _texto(datos.get("tipo_politica"))
    if politica:
        palancas["tipo_politica"] = politica

    return Traduccion(escenario=EscenarioPropuesto(
        nombre=_recortar(datos.get("nombre"), 80),
        pregunta=_pregunta_visible(datos.get("pregunta"), pregunta_original),
        mecanismo=_recortar(datos.get("mecanismo"), 400),
        **palancas,
    ))


def _pregunta_visible(del_modelo, original):
    """La pregunta que se muestra. Nunca el aviso de rechazo que se le mandó a Langflow."""
    propia = _texto(del_modelo)
    base = _recortar(original, 200) if original else ""
    if base and (not propia or "RECHAZADA" in propia or "validador del motor" in propia):
        return base
    return _recortar(propia, 200)


def _recortar(texto, maximo):
    """Corta en un límite de oración o de palabra, sin pasarse del máximo del esquema."""
    texto = _texto(texto)
    if len(texto) <= maximo:
        return texto
    corte = texto[:maximo].rstrip()
    for marca in (".", "?", "!"):
        pos = corte.rfind(marca)
        if pos >= maximo // 2:
            return corte[:pos + 1].rstrip()
    espacio = corte.rfind(" ")
    if espacio >= maximo // 2:
        return corte[:espacio].rstrip()
    return corte


def _ejecutar(entrada):
    url = f"{config.LANGFLOW_URL}/api/v1/run/{config.LANGFLOW_FLOW_ID}"
    cuerpo = json.dumps({
        "input_value": entrada,
        "input_type": "chat",
        "output_type": "chat",
        # Cada intento es una sesión nueva: el rechazo ya viaja dentro del texto,
        # y reutilizar el chat duplicaría la pregunta.
        "session_id": f"soaz-{uuid.uuid4()}",
    }).encode("utf-8")
    solicitud = urllib.request.Request(
        url, data=cuerpo,
        headers={
            "Content-Type": "application/json",
            "x-api-key": config.LANGFLOW_API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=120) as respuesta:
            crudo = respuesta.read()
        if crudo[:2] == b"\x1f\x8b":
            crudo = gzip.decompress(crudo)
        crudo = crudo.decode("utf-8")
    except urllib.error.HTTPError as error:
        detalle = error.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"Langflow respondió {error.code}: {detalle}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"No se alcanzó Langflow en {config.LANGFLOW_URL}. "
            f"¿Está encendido en el puerto 7860? ({error.reason})") from error
    return _extraer_texto(json.loads(crudo))


def _extraer_texto(respuesta):
    """El texto del Chat Output, venga como mensaje o como data estructurada."""
    for bloque in respuesta.get("outputs") or []:
        for salida in bloque.get("outputs") or []:
            resultados = salida.get("results") or {}
            mensaje = resultados.get("message") or {}
            if isinstance(mensaje, dict):
                texto = mensaje.get("text")
                if texto:
                    return texto
            for clave in ("data", "structured_response", "json"):
                if clave in resultados and resultados[clave] not in (None, ""):
                    valor = resultados[clave]
                    return valor if isinstance(valor, str) else json.dumps(valor, ensure_ascii=False)
    raise RuntimeError("Langflow terminó la corrida sin devolver una salida de texto.")


def _como_dict(texto):
    if isinstance(texto, dict):
        return texto
    limpio = str(texto).strip()
    cercado = re.search(r"```(?:json)?\s*(\{.*\})\s*```", limpio, re.DOTALL)
    if cercado:
        limpio = cercado.group(1)
    else:
        inicio, fin = limpio.find("{"), limpio.rfind("}")
        if inicio >= 0 and fin > inicio:
            limpio = limpio[inicio:fin + 1]
    try:
        datos = json.loads(limpio)
    except json.JSONDecodeError:
        try:
            datos = ast.literal_eval(limpio)
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                "Langflow no devolvió JSON interpretable. Empieza así: "
                f"{str(texto)[:180]}") from error
    if not isinstance(datos, dict):
        raise ValueError("Langflow devolvió JSON, pero no es un objeto.")
    return datos


def _flag(valor):
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in ("true", "1", "si", "sí", "yes")


def _texto(valor):
    if valor is None:
        return ""
    return str(valor).strip()


def _vacio(valor):
    return _texto(valor).lower() in _VACIO


def _flotante(valor):
    if _vacio(valor):
        return None
    return float(_texto(valor).replace(",", "."))


def _entero(valor):
    if _vacio(valor):
        return None
    return int(float(_texto(valor).replace(",", ".")))
