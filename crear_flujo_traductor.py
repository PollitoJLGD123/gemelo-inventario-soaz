# crear_flujo_traductor.py
"""
Crea o actualiza en el Langflow local el flujo «Traductor de escenarios SOAZ».

El flujo es Chat Input → Agente (OpenRouter) → Chat Output. El agente devuelve
el JSON plano que modulo_copiloto/langflow_cliente.py convierte en palancas.
No simula inventario: eso sigue en el motor.

Requisitos: Langflow en http://localhost:7860 y, en el entorno o en .env,
LANGFLOW_API_KEY. El modelo sale del mismo catálogo OpenRouter que ya usa el
flujo Tutor de idiomas; la clave de OpenRouter vive en las variables de
Langflow, no en este repositorio.

Uso:
    python crear_flujo_traductor.py
"""
import copy
import gzip
import json
import os
import urllib.error
import urllib.request

import config
from modulo_copiloto.grafo import INSTRUCCION_TRADUCCION
from modulo_copiloto.langflow_cliente import ESQUEMA_LANGFLOW, INSTRUCCION_CAMPOS

NOMBRE = "Traductor de escenarios SOAZ"
ENDPOINT = "traductor_escenarios_soaz"
CARPETA = "cbc755d5-d85f-4a80-90df-502bf18a2fe6"
DONANTE = "56cb8e79-4a81-4628-960f-e6b7b15b8edb"
MODELO = "anthropic/claude-sonnet-4.5"
DESCRIPCION = (
    "Traduce una pregunta de abastecimiento de la Comercializadora SOAZ a las "
    "palancas del gemelo digital. No calcula costos ni niveles de servicio: "
    "devuelve el JSON que el motor valida antes de simular."
)


def _clave():
    clave = os.getenv("LANGFLOW_API_KEY") or config.LANGFLOW_API_KEY
    if not clave:
        raise SystemExit(
            "Falta LANGFLOW_API_KEY en el entorno o en .env. "
            "Es la clave del servidor Langflow, no la de Anthropic.")
    return clave


def _pedir(metodo, ruta, clave, cuerpo=None):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    solicitud = urllib.request.Request(
        f"{config.LANGFLOW_URL}{ruta}", data=datos,
        headers={"Content-Type": "application/json", "x-api-key": clave},
        method=metodo,
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=60) as respuesta:
            crudo = respuesta.read()
        if crudo[:2] == b"\x1f\x8b":
            crudo = gzip.decompress(crudo)
        crudo = crudo.decode("utf-8")
    except urllib.error.HTTPError as error:
        detalle = error.read().decode("utf-8", "replace")[:800]
        raise SystemExit(f"{metodo} {ruta} → {error.code}: {detalle}") from error
    return json.loads(crudo) if crudo else None


def _asa(objeto):
    """Langflow guarda los handles con œ en lugar de comillas."""
    return json.dumps(objeto, separators=(",", ":")).replace('"', "\u0153")


def _clonar(nodo, nuevo_id, x, y):
    nodo = copy.deepcopy(nodo)
    viejo = nodo["id"]
    crudo = json.dumps(nodo).replace(viejo, nuevo_id)
    nodo = json.loads(crudo)
    nodo["id"] = nuevo_id
    nodo["position"] = {"x": x, "y": y}
    nodo["data"]["id"] = nuevo_id
    for sobra in ("measured", "selected", "dragging"):
        nodo.pop(sobra, None)
    return nodo


def _arista(origen, destino, origen_handle, destino_handle):
    salida = _asa(origen_handle)
    entrada = _asa(destino_handle)
    return {
        "animated": False,
        "className": "",
        "selected": False,
        "source": origen,
        "target": destino,
        "sourceHandle": salida,
        "targetHandle": entrada,
        "data": {"sourceHandle": origen_handle, "targetHandle": destino_handle},
        "id": f"reactflow__edge-{origen}{salida}-{destino}{entrada}",
    }


def _modelo(opciones):
    for opcion in opciones:
        if isinstance(opcion, dict) and opcion.get("name") == MODELO:
            return opcion
    raise SystemExit(
        f"Langflow no ofrece {MODELO}. Revisa el proveedor OpenRouter del agente.")


def _por_tipo(donante):
    return {n["data"]["type"]: n for n in donante["data"]["nodes"]}


def _arista_donante(donante, tipo_origen, tipo_destino):
    ids = {n["id"]: n["data"]["type"] for n in donante["data"]["nodes"]}
    for arista in donante["data"]["edges"]:
        if ids.get(arista["source"]) == tipo_origen and ids.get(arista["target"]) == tipo_destino:
            return arista
    raise SystemExit(f"El flujo donante no conecta {tipo_origen} con {tipo_destino}.")


def _reescribir_handle(handle, id_viejo, id_nuevo, **cambios):
    if isinstance(handle, str):
        handle = json.loads(handle.replace("\u0153", '"'))
    handle = dict(handle)
    if handle.get("id") == id_viejo:
        handle["id"] = id_nuevo
    handle.update(cambios)
    return handle


def construir_datos(donante):
    nodos = _por_tipo(donante)
    for tipo in ("ChatInput", "Agent", "ChatOutput"):
        if tipo not in nodos:
            raise SystemExit(f"El flujo donante no tiene un nodo {tipo}.")

    id_entrada = nodos["ChatInput"]["id"]
    id_agente = nodos["Agent"]["id"]
    id_salida = nodos["ChatOutput"]["id"]
    entrada = _clonar(nodos["ChatInput"], "ChatInput-soaz1", 80, 180)
    agente = _clonar(nodos["Agent"], "Agent-soaz1", 520, 80)
    salida = _clonar(nodos["ChatOutput"], "ChatOutput-soaz1", 1100, 180)

    plantilla = agente["data"]["node"]["template"]
    plantilla["system_prompt"]["value"] = (
        INSTRUCCION_TRADUCCION.strip() + "\n\n" + INSTRUCCION_CAMPOS.strip())
    plantilla["output_schema"]["value"] = ESQUEMA_LANGFLOW
    plantilla["format_instructions"]["value"] = (
        "Devuelve un único objeto JSON con los campos del esquema. "
        "Las palancas que la pregunta no pide van en cadena vacía. " + INSTRUCCION_CAMPOS)
    plantilla["model"]["value"] = [_modelo(plantilla["model"].get("options") or [])]
    plantilla["add_current_date_tool"]["value"] = False
    plantilla["add_calculator_tool"]["value"] = False
    if "tools" in plantilla:
        plantilla["tools"]["value"] = []
    agente["data"]["node"]["display_name"] = "Traductor SOAZ"
    agente["data"]["selected_output"] = "structured_response"
    entrada["data"]["node"]["display_name"] = "Pregunta del negocio"
    salida["data"]["node"]["display_name"] = "Escenario en JSON"

    hacia_agente = _arista_donante(donante, "ChatInput", "Agent")
    hacia_salida = _arista_donante(donante, "Agent", "ChatOutput")
    aristas = [
        _arista(
            "ChatInput-soaz1", "Agent-soaz1",
            _reescribir_handle(hacia_agente["data"]["sourceHandle"], id_entrada, "ChatInput-soaz1"),
            _reescribir_handle(hacia_agente["data"]["targetHandle"], id_agente, "Agent-soaz1"),
        ),
        _arista(
            "Agent-soaz1", "ChatOutput-soaz1",
            _reescribir_handle(
                hacia_salida["data"]["sourceHandle"], id_agente, "Agent-soaz1",
                name="structured_response", output_types=["Data", "JSON"]),
            _reescribir_handle(hacia_salida["data"]["targetHandle"], id_salida, "ChatOutput-soaz1"),
        ),
    ]
    return {
        "nodes": [entrada, agente, salida],
        "edges": aristas,
        "viewport": {"x": 0, "y": 0, "zoom": 0.8},
    }


def _guardar_env(flow_id, clave):
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(ruta):
        return
    with open(ruta, encoding="utf-8") as archivo:
        texto = archivo.read()
    for nombre, valor in (
        ("LANGFLOW_URL", config.LANGFLOW_URL),
        ("LANGFLOW_FLOW_ID", flow_id),
        ("LANGFLOW_API_KEY", clave),
    ):
        linea = f"{nombre}={valor}"
        if f"{nombre}=" in texto:
            lineas = []
            for existente in texto.splitlines():
                lineas.append(linea if existente.startswith(f"{nombre}=") else existente)
            texto = "\n".join(lineas) + ("\n" if texto.endswith("\n") else "")
        else:
            if texto and not texto.endswith("\n"):
                texto += "\n"
            texto += linea + "\n"
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write(texto)


def main():
    clave = _clave()
    donante = _pedir("GET", f"/api/v1/flows/{DONANTE}", clave)
    datos = construir_datos(donante)
    existentes = _pedir("GET", "/api/v1/flows/?get_all=true", clave)
    previo = next((f for f in existentes if f.get("name") == NOMBRE), None)
    cuerpo = {
        "name": NOMBRE,
        "description": DESCRIPCION,
        "data": datos,
        "folder_id": CARPETA,
        "endpoint_name": ENDPOINT,
        "mcp_enabled": True,
        "action_name": ENDPOINT,
        "action_description": DESCRIPCION,
    }
    if previo:
        flujo = _pedir("PATCH", f"/api/v1/flows/{previo['id']}", clave, cuerpo)
        print(f"Flujo actualizado: {flujo['id']}")
    else:
        flujo = _pedir("POST", "/api/v1/flows/", clave, cuerpo)
        print(f"Flujo creado: {flujo['id']}")
    _guardar_env(flujo["id"], clave)
    print(f"Endpoint: {ENDPOINT}")
    print("Listo. .env quedó con LANGFLOW_URL, LANGFLOW_FLOW_ID y LANGFLOW_API_KEY.")
    return flujo["id"]


if __name__ == "__main__":
    main()
