# -*- mode: python ; coding: utf-8 -*-
"""
Spec de PyInstaller del Gemelo Digital de Inventario — SOAZ.

    venv\\Scripts\\pyinstaller.exe --noconfirm gemelo_soaz.spec

Produce dist/GemeloInventarioSOAZ/GemeloInventarioSOAZ.exe (modo carpeta).

Dos cosas que un spec normal no necesita y este sí:

1. Streamlit se ejecuta a sí mismo leyendo el script del usuario como ARCHIVO
   fuente, no como módulo importado. Por eso app_motor.py, config.py y
   modulo_simulacion/ viajan como DATOS (carpeta `app/`), no congelados.

2. Nada del lanzador importa numpy/pandas/scipy/plotly/simpy: los importa
   app_motor.py en runtime, fuera del grafo de dependencias que PyInstaller
   puede ver. Van declarados a mano en `hiddenimports`.
"""
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

NOMBRE = "GemeloInventarioSOAZ"
RAIZ = os.path.abspath(os.getcwd())

datas = []
binaries = []
hiddenimports = []

# --- Paquetes con datos propios (assets, esquemas JSON, metadata) -----------
# Streamlit necesita su carpeta static/ y su metadata (lee su propia versión
# con importlib.metadata). altair y plotly traen esquemas Vega/JSON.
for paquete in ("streamlit", "altair", "plotly", "narwhals", "pyarrow", "pydeck"):
    d, b, h = collect_all(paquete)
    datas += d
    binaries += b
    hiddenimports += h

# --- Motor de simulación: importado por el script, no por el lanzador -------
hiddenimports += [
    "numpy",
    "pandas",
    "scipy",
    "scipy.stats",
    "scipy.special",
    "scipy.optimize",
    "simpy",
    "plotly.graph_objects",
    "plotly.express",
]

# --- Copiloto de escenarios: OPCIONAL ---------------------------------------
# Estos paquetes pueden no estar instalados, y el .exe tiene que construirse
# igual: el copiloto es opcional por diseño y en ejecución se desactiva solo.
# Van por collect_all y no por hiddenimports porque langchain_core y langgraph
# leen su propia versión con importlib.metadata, y sin los .dist-info revientan
# al importar. PyInstaller ya trae hook-langchain.py y hook-pydantic.py.
for paquete in ("langchain", "langchain_core", "langgraph",
                "langchain_anthropic", "anthropic", "pydantic", "langsmith"):
    try:
        d, b, h = collect_all(paquete)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h

# --- Servidor: uvicorn/starlette resuelven sus protocolos por nombre --------
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "starlette",
    "websockets",
    "websockets.legacy",
    "h11",
    "httptools",
    "anyio",
    "click",
    "toml",
    "packaging",
]

# --- El código de la aplicación, como datos --------------------------------
# Solo los .py: el __pycache__ del desarrollador no tiene por qué viajar al
# ejecutable, donde además nadie puede regenerarlo.
datas += [
    (os.path.join(RAIZ, "app_motor.py"), "app"),
    (os.path.join(RAIZ, "config.py"), "app"),
]
for paquete in ("modulo_simulacion", "modulo_copiloto"):
    carpeta = os.path.join(RAIZ, paquete)
    if not os.path.isdir(carpeta):
        continue
    for archivo in sorted(os.listdir(carpeta)):
        if archivo.endswith(".py"):
            datas.append((
                os.path.join(carpeta, archivo),
                os.path.join("app", paquete),
            ))

icono = os.path.join(RAIZ, "icono.ico")
icono = icono if os.path.isfile(icono) else None

a = Analysis(
    ["lanzador.py"],
    pathex=[RAIZ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PySide2", "IPython", "pytest"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NOMBRE,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # la consola es el control del motor: cerrarla lo apaga
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icono,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=NOMBRE,
)
