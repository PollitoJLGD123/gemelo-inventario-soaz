# lanzador.py
"""
Punto de entrada del ejecutable de Windows del Gemelo Digital de Inventario.

No lo ejecuta el usuario final: es lo que PyInstaller congela en
GemeloInventarioSOAZ.exe. Levanta el servidor de Streamlit sobre
`app_motor.py` (empaquetado como dato dentro del bundle) y abre el navegador.

Se puede probar sin congelar nada:

    python lanzador.py
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser

PUERTO_PREFERIDO = 8501
DIRECCION = "localhost"


def consola_utf8() -> None:
    """Sin esto la consola de Windows escribe «Comercializadora SOAZ» con basura:
    su página de códigos por defecto no es UTF-8."""
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


def directorio_base() -> str:
    """Raíz de los recursos: el bundle si está congelado, el repo si no."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # noqa: SLF001 — lo define PyInstaller en runtime
    return os.path.dirname(os.path.abspath(__file__))


def directorio_app() -> str:
    """Carpeta que contiene app_motor.py, config.py y modulo_simulacion/."""
    base = directorio_base()
    empaquetado = os.path.join(base, "app")
    return empaquetado if os.path.isdir(empaquetado) else base


def puerto_libre(preferido: int) -> int:
    """El puerto preferido si está libre; si no, uno que el sistema asigne."""
    for candidato in (preferido, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((DIRECCION, candidato))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferido


def abrir_navegador_cuando_responda(url: str, puerto: int, espera_max: float = 60.0) -> None:
    """Abre el navegador recién cuando el servidor acepta conexiones."""
    inicio = time.time()
    while time.time() - inicio < espera_max:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((DIRECCION, puerto)) == 0:
                time.sleep(0.6)  # deja que Streamlit termine de servir el index
                webbrowser.open(url)
                return
        time.sleep(0.4)
    print(f"\nEl servidor tardó en responder. Abre manualmente: {url}\n")


def main() -> None:
    consola_utf8()
    app_dir = directorio_app()
    script = os.path.join(app_dir, "app_motor.py")

    if not os.path.isfile(script):
        print(f"ERROR: no se encontró app_motor.py en {app_dir}")
        input("Pulsa Enter para cerrar...")
        sys.exit(1)

    # El script y sus módulos (config, modulo_simulacion) se importan desde aquí.
    sys.path.insert(0, app_dir)
    os.chdir(app_dir)

    # Home escribible para las credenciales y la caché de Streamlit: dentro del
    # bundle no se puede escribir, y %USERPROFILE% siempre existe en Windows.
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    puerto = puerto_libre(PUERTO_PREFERIDO)
    url = f"http://{DIRECCION}:{puerto}"

    from streamlit.web import bootstrap

    opciones = {
        "server.port": puerto,
        "server.address": DIRECCION,
        "server.headless": True,          # el navegador lo abrimos nosotros
        "server.fileWatcherType": "none",  # el código es fijo dentro del .exe
        "server.runOnSave": False,
        "global.developmentMode": False,
        # Explícito: congelado, Streamlit no reconoce su instalación como
        # release y cae a DEBUG, que llena la consola del usuario de ruido y
        # esconde los errores que esa consola existe para mostrar.
        "logger.level": "info",
        "browser.gatherUsageStats": False,
        "browser.serverAddress": DIRECCION,
        "browser.serverPort": puerto,
    }
    # Y no un set_option() por clave: `bootstrap.run` no aplica sus flag_options
    # —de eso se encarga la CLI antes de llamarlo—, y varios ajustes (el nivel
    # de log entre ellos) solo se releen cuando se dispara la señal de config
    # parseada. Esta es la misma llamada que hace `streamlit run`.
    bootstrap.load_config_options(opciones)

    print("=" * 66)
    print("  Gemelo Digital de Inventario — Comercializadora SOAZ")
    print("  Universidad Nacional de Trujillo — Ingeniería de Sistemas")
    print("=" * 66)
    print(f"\n  Abriendo la interfaz en {url}")
    print("  Deja ESTA ventana abierta mientras uses el gemelo.")
    print("  Para cerrar el motor: cierra esta ventana o pulsa Ctrl+C.\n")

    threading.Thread(
        target=abrir_navegador_cuando_responda, args=(url, puerto), daemon=True
    ).start()

    try:
        bootstrap.run(script, False, [], opciones)
    except KeyboardInterrupt:
        print("\nMotor detenido.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
