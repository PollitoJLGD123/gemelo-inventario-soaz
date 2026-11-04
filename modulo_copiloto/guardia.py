# modulo_copiloto/guardia.py
"""
Cerco numérico sobre el texto que produce el modelo de lenguaje.

LA REGLA QUE ESTE MÓDULO HACE CUMPLIR
--------------------------------------
El LLM nunca produce un número. Redacta, y solo puede citar cifras que las
herramientas del motor devolvieron de verdad. Este módulo lo verifica: extrae
cada número del texto y exige que sea trazable a un resultado real. Si uno no
lo es, el texto completo se descarta y queda la narrativa determinista sola.

Por eso el módulo no lanza excepciones ni corrige: informa. Decidir qué hacer
con un texto sucio es del grafo, no del cerco.

POR QUÉ SE EXCLUYEN LAS SERIES DIARIAS
---------------------------------------
El resultado de ejecutar_escenario trae _replica_muestra, con la traza día a
día de cada SKU: del orden de miles de números entre 0 y unos pocos cientos.
Si eso entrara al conjunto permitido, casi cualquier cifra inventada
encontraría por casualidad un número igual y pasaría el cerco.

UN CONJUNTO PERMITIDO GRANDE ES UN CERCO DÉBIL. La fuerza de esta verificación
no está en el algoritmo sino en mantener chico y pertinente lo que se admite:
los agregados, sus intervalos y la tabla por SKU. Nada más.

LO QUE ESTE CERCO NO PRUEBA
----------------------------
Que un número sea trazable no significa que el modelo lo haya usado bien: puede
citar el costo del escenario y llamarlo costo del Base. El cerco ataca la
fabricación de cifras, no la atribución equivocada. Contra eso protege que la
narrativa determinista se muestre siempre, y que sea ella la que manda.
"""
import bisect
import re
from numbers import Number

import numpy as np
import pandas as pd

# Claves cuyo contenido no entra al conjunto permitido. Ver el encabezado.
CLAVES_EXCLUIDAS = ("_replica_muestra", "serie", "por_sku_serie")

# Enteros chicos que el texto puede usar para enumerar («los 3 SKU más caros»)
# sin que sean una cifra de resultado.
LIBRES = (0.0, 1.0, 2.0, 3.0)

# Captura «1,234», «11,506.75», «97.91», «0.95», «-79». La alternativa con
# separador de miles va primero para que no se parta en dos.
_PATRON = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

_PROFUNDIDAD_MAX = 12


def _numerico(valor):
    """Descarta booleanos, NaN e infinitos. True es un int en Python."""
    if isinstance(valor, bool) or valor is None:
        return None
    if not isinstance(valor, Number):
        return None
    v = float(valor)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _recorrer(objeto, acumulador, excluir, profundidad):
    if profundidad > _PROFUNDIDAD_MAX:
        return
    v = _numerico(objeto)
    if v is not None:
        acumulador.add(v)
        return
    if isinstance(objeto, str) or objeto is None:
        return
    if isinstance(objeto, pd.DataFrame):
        for columna in objeto.columns:
            _recorrer(objeto[columna], acumulador, excluir, profundidad + 1)
    elif isinstance(objeto, pd.Series):
        for x in objeto.tolist():
            _recorrer(x, acumulador, excluir, profundidad + 1)
    elif isinstance(objeto, np.ndarray):
        for x in objeto.ravel().tolist():
            _recorrer(x, acumulador, excluir, profundidad + 1)
    elif isinstance(objeto, dict):
        for clave, x in objeto.items():
            if clave in excluir:
                continue
            _recorrer(x, acumulador, excluir, profundidad + 1)
    elif isinstance(objeto, (list, tuple, set, frozenset)):
        for x in objeto:
            _recorrer(x, acumulador, excluir, profundidad + 1)


def numeros_permitidos(*objetos, excluir=CLAVES_EXCLUIDAS):
    """
    Todo número que aparezca en los resultados de las herramientas.

    Recorre dicts, listas, DataFrames y arrays. Las claves en `excluir` se
    saltan enteras: ver el encabezado del módulo.
    """
    acumulador = set()
    for objeto in objetos:
        _recorrer(objeto, acumulador, set(excluir), 0)
    return acumulador


def _formas(valor):
    """
    Las formas en que un número puede aparecer escrito.

    El motor devuelve 11505.73 y el texto dirá «S/. 11,506». Una fracción como
    0.95 se escribe «95%». El cerco tiene que reconocer esas reescrituras sin
    aflojar la exigencia de trazabilidad.
    """
    formas = {valor}
    for decimales in (0, 1, 2):
        formas.add(round(valor, decimales))
    if -10.0 <= valor <= 10.0:
        for decimales in (0, 1, 2):
            formas.add(round(valor * 100.0, decimales))
    return formas


def extraer_numeros(texto):
    """
    Cada número del texto, como (literal, valor).

    Se descartan los que vienen pegados a un identificador — el «0012» de
    «SKU-0012» no es una cifra de resultado.
    """
    encontrados = []
    for coincidencia in _PATRON.finditer(texto or ""):
        inicio = coincidencia.start()
        if inicio > 0 and (texto[inicio - 1].isalnum() or texto[inicio - 1] in "_-"):
            continue
        literal = coincidencia.group()
        try:
            encontrados.append((literal, float(literal.replace(",", ""))))
        except ValueError:
            continue
    return encontrados


def verificar(texto, permitidos, libres=LIBRES, rel_tol=0.001, abs_tol=0.01):
    """
    ¿Todo número del texto es trazable a un resultado real?

    Devuelve {ok, intrusos, verificados, total}. No lanza ni corrige: informa.
    Un intruso es un número que el modelo escribió y que ninguna herramienta
    devolvió — es decir, se lo inventó.
    """
    formas = set()
    for valor in permitidos:
        formas |= _formas(valor)
    ordenadas = sorted(formas)
    libres = set(libres)

    intrusos = []
    verificados = 0
    for literal, valor in extraer_numeros(texto):
        if _trazable(valor, formas, ordenadas, libres, rel_tol, abs_tol):
            verificados += 1
        else:
            intrusos.append({"literal": literal, "valor": valor})

    return {
        "ok": not intrusos,
        "intrusos": intrusos,
        "verificados": verificados,
        "total": verificados + len(intrusos),
    }


def _trazable(valor, formas, ordenadas, libres, rel_tol, abs_tol):
    if valor in libres or valor in formas:
        return True
    if not ordenadas:
        return False
    # Solo hay que mirar los dos vecinos: la lista está ordenada.
    i = bisect.bisect_left(ordenadas, valor)
    for j in (i - 1, i):
        if 0 <= j < len(ordenadas):
            candidato = ordenadas[j]
            if abs(valor - candidato) <= max(abs_tol, rel_tol * abs(candidato)):
                return True
    return False


def resumen(informe):
    """Una línea legible para la pantalla y para el log de la sustentación."""
    if informe["ok"]:
        return (f"Cerco numérico: {informe['verificados']} de {informe['total']} "
                f"cifras trazadas a resultados del motor.")
    cifras = ", ".join(i["literal"] for i in informe["intrusos"])
    return (f"Cerco numérico: RECHAZADO. {len(informe['intrusos'])} de "
            f"{informe['total']} cifras no aparecen en ningún resultado ({cifras}).")
