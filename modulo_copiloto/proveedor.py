# modulo_copiloto/proveedor.py
"""
Acceso al modelo de lenguaje, y la decisión de si el copiloto puede correr.

EL COPILOTO ES OPCIONAL Y ESTE MÓDULO ES QUIEN LO HACE CUMPLIR
---------------------------------------------------------------
El gemelo tiene que arrancar en una máquina sin clave, sin red y sin estos
paquetes instalados — la de la empresa, la de la sustentación. Las otras siete
pestañas no pueden enterarse de que este módulo existe. De ahí tres reglas:

  - los imports de langchain/langgraph van DENTRO de las funciones, nunca arriba;
  - disponible() responde sin importar nada pesado ni tocar la red;
  - ningún fallo de aquí se propaga al motor.

POR QUÉ AQUÍ HAY try/except Y EN EL MOTOR NO
---------------------------------------------
modulo_simulacion no tiene un solo try/except, y está bien así: un error de
cálculo debe reventar y verse, no esconderse tras un valor por defecto. Este
módulo y langflow_cliente.py hablan con un servicio de red, y que el proveedor
esté caído o la clave no esté puesta no es un error del gemelo sino una
condición de operación normal. Atrapar es correcto en esos dos, y en ninguna
otra parte del copiloto.
"""
import importlib.util
import os

import config

# Qué variable de entorno lleva la clave, según el proveedor que declare
# COPILOTO_MODELO. Un modelo local no necesita ninguna.
CLAVE_POR_PROVEEDOR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "ollama": None,
}

# Sin estos paquetes el módulo no existe. Se comprueba por spec, sin importar:
# importar langchain cuesta cerca de un segundo y las otras pestañas no deben
# pagarlo solo para descubrir que el copiloto está apagado.
PAQUETES = ("langgraph", "langchain", "langchain_core")


def proveedor_declarado():
    """El prefijo de COPILOTO_MODELO: 'anthropic' en 'anthropic:claude-sonnet-5'."""
    modelo = (config.COPILOTO_MODELO or "").strip()
    return modelo.split(":", 1)[0] if ":" in modelo else ""


def variable_de_clave():
    """La variable de entorno que este proveedor necesita, o None si es local."""
    return CLAVE_POR_PROVEEDOR.get(proveedor_declarado(), "")


def paquetes_faltantes():
    return [p for p in PAQUETES if importlib.util.find_spec(p) is None]


def disponible():
    """
    (¿puede correr el copiloto?, motivo legible si no).

    El motivo va a pantalla tal cual, así que se escribe para la persona que
    va a sustentar, no para quien depura.
    """
    if not config.COPILOTO_ACTIVO:
        return False, ("El copiloto está apagado por configuración "
                       "(COPILOTO_ACTIVO=0). El resto del gemelo funciona igual.")

    faltantes = paquetes_faltantes()
    if faltantes:
        return False, (
            f"Faltan paquetes del copiloto: {', '.join(faltantes)}. "
            f"Se instalan con `pip install -r requirements.txt`. "
            f"Las otras siete pestañas no los necesitan.")

    proveedor = proveedor_declarado()
    if not proveedor:
        return False, (
            f"COPILOTO_MODELO='{config.COPILOTO_MODELO}' no declara proveedor. "
            f"Debe tener la forma 'proveedor:modelo', por ejemplo "
            f"'anthropic:claude-sonnet-5'.")

    if proveedor not in CLAVE_POR_PROVEEDOR:
        conocidos = ", ".join(sorted(CLAVE_POR_PROVEEDOR))
        return False, (f"Proveedor '{proveedor}' no reconocido. "
                       f"Conocidos: {conocidos}.")

    variable = CLAVE_POR_PROVEEDOR[proveedor]
    if variable and not os.getenv(variable):
        return False, (
            f"Falta la clave de API: la variable de entorno {variable} no está "
            f"definida. En Windows se fija con `setx {variable} \"...\"` y hay "
            f"que reabrir la consola. Sin ella, esta pestaña queda inactiva y "
            f"el gemelo sigue funcionando igual.")

    return True, f"Copiloto listo con {config.COPILOTO_MODELO}."


def modelo():
    """
    El chat model configurado. Lanza RuntimeError si el copiloto no puede correr.

    AQUÍ NO SE PASA `temperature`, Y NO ES UN OLVIDO
    -------------------------------------------------
    Los modelos Claude 4.7 en adelante —Sonnet 5 incluido— rechazan ese
    parámetro con un 400: «temperature is deprecated for this model». Solo
    aceptan su valor por defecto, incluso con el pensamiento apagado.

    El camino por el que se colaba no es evidente, y conviene dejarlo escrito
    porque cuesta media hora encontrarlo: `anthropic>=1` quitó `temperature`
    de los argumentos nombrados de Messages.create, así que langchain-anthropic
    lo REUBICA en `extra_body`, que el SDK fusiona al JSON tal cual (ver
    langchain_anthropic/_sdk_compat.py). Inspeccionar el payload buscando la
    clave `temperature` no muestra nada: viaja dentro de extra_body y llega
    igual al API.

    Se pedía temperatura 0 para que traducir dos veces la misma pregunta diera
    el mismo escenario. Esa garantía ya no está disponible por esta vía, y hay
    que decirlo en vez de fingir que sigue ahí. Lo que protege de una
    traducción caprichosa no es el muestreo: es el validador tipado de
    esquemas.py y, sobre todo, que una persona apruebe las palancas en pantalla
    antes de simular nada. Esa siempre fue la garantía fuerte; la temperatura
    era el cinturón, no el arnés.

    COPILOTO_ESFUERZO regula la profundidad de razonamiento en los modelos que
    lo soportan — el reemplazo moderno de temperature. Sin declarar no se envía
    nada, que es lo seguro: un parámetro no soportado devuelve 400 y tumba la
    pestaña completa.
    """
    listo, motivo = disponible()
    if not listo:
        raise RuntimeError(motivo)

    from langchain.chat_models import init_chat_model

    opciones = {}
    if config.COPILOTO_ESFUERZO:
        opciones["reasoning_effort"] = config.COPILOTO_ESFUERZO
    return init_chat_model(config.COPILOTO_MODELO, **opciones)
