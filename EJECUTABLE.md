# 🖥️ Ejecutable de Windows — Gemelo Digital de Inventario SOAZ

Cómo se compila el `.exe`, qué produce y por qué está armado así.

---

## Para quien solo va a usarlo

1. Copiar la carpeta **`GemeloInventarioSOAZ`** completa a la máquina destino.
2. Doble clic en **`GemeloInventarioSOAZ.exe`**.
3. Se abre una ventana negra (el motor) y, unos segundos después, el navegador
   con la interfaz en `http://localhost:8501`.
4. Para cerrar el gemelo: cerrar la ventana negra. Cerrar solo el navegador
   **no** apaga el motor.

No requiere Python instalado, ni PostgreSQL, ni conexión a internet.

> **La carpeta entera es el programa.** El `.exe` suelto no funciona: necesita
> la carpeta `_internal\` que va a su lado. Si hay que mandarlo por correo o
> USB, comprimir la carpeta completa en un `.zip`.

### Lo que Windows va a decir la primera vez

- **SmartScreen** («Windows protegió su PC»): el ejecutable no está firmado
  digitalmente. *Más información → Ejecutar de todas formas*. Firmarlo requiere
  un certificado de code signing, que se compra aparte.
- **Firewall**: pide permiso de red. El servidor escucha solo en `localhost`;
  basta con permitir en redes privadas, o incluso cancelar — sigue funcionando.
- **Antivirus**: algunos marcan los binarios de PyInstaller por heurística. Es
  un falso positivo conocido del *bootloader*, no del código de la tesis.

---

## Para quien lo compila

### Requisitos

- Windows 10/11 de 64 bits
- Python 3.13 (el `venv` del proyecto ya lo usa)

### Compilar

```powershell
powershell -ExecutionPolicy Bypass -File construir_exe.ps1
```

El script crea el `venv` si no existe, instala PyInstaller, borra
`build\` y `dist\` anteriores y compila. Tarda entre 2 y 5 minutos.

Resultado: **`dist\GemeloInventarioSOAZ\`** (~600 MB — numpy, scipy, pandas,
pyarrow y plotly pesan).

Equivalente a mano:

```powershell
venv\Scripts\python.exe -m PyInstaller --noconfirm --clean gemelo_soaz.spec
```

### Archivos que hacen posible el ejecutable

| Archivo | Rol |
|---|---|
| [`lanzador.py`](lanzador.py) | Punto de entrada congelado: elige puerto, levanta el servidor de Streamlit sobre `app_motor.py` y abre el navegador |
| [`gemelo_soaz.spec`](gemelo_soaz.spec) | Receta de PyInstaller: qué datos empaquetar y qué imports declarar a mano |
| [`construir_exe.ps1`](construir_exe.ps1) | Compilación de un comando, con limpieza y verificación |

El lanzador se puede probar sin congelar nada, que es como se depura un
problema de arranque:

```powershell
venv\Scripts\python.exe lanzador.py
```

---

## Por qué el spec no es un spec normal

Empaquetar Streamlit no es como empaquetar un script cualquiera. Dos cosas
rompen el flujo por defecto de PyInstaller:

**1. Streamlit lee el script del usuario como archivo, no como módulo.**
El *runner* de Streamlit abre `app_motor.py`, lo compila con `compile()` y lo
ejecuta. Un módulo congelado dentro del `.pyz` no le sirve: necesita el archivo
en disco. Por eso `app_motor.py`, `config.py` y `modulo_simulacion/` viajan
como **datos** en la carpeta `app\`, no como código congelado. El lanzador
mete esa carpeta en `sys.path` antes de arrancar el servidor.

Efecto lateral útil: el código del motor queda legible dentro de
`_internal\app\`, lo que sirve para la sustentación de la tesis.

**2. PyInstaller no puede ver las dependencias del motor.**
El grafo de imports arranca en `lanzador.py`, que solo importa Streamlit. Nadie
en ese grafo importa numpy, pandas, scipy, simpy ni plotly — los importa
`app_motor.py`, en runtime, fuera del alcance del análisis estático. Van
declarados explícitamente en `hiddenimports` del spec. **Si el motor gana una
dependencia nueva, hay que agregarla ahí o el `.exe` fallará al abrir la
pestaña que la usa** (no al arrancar: el error aparece tarde y confunde).

Además:

- `collect_all("streamlit")` trae su carpeta `static\` (el frontend compilado)
  y su *metadata* — Streamlit lee su propia versión con `importlib.metadata`.
- `collect_submodules("uvicorn")` es necesario porque uvicorn resuelve sus
  protocolos HTTP y WebSocket por nombre en texto, no con `import`.
- `server.fileWatcherType = "none"`: dentro del `.exe` el código es fijo, y el
  watcher solo gastaría CPU vigilando archivos que nadie va a editar.
- Modo carpeta y no archivo único: en `--onefile` Streamlit descomprime ~600 MB
  en `%TEMP%` en cada arranque, lo que agrega medio minuto de espera y a veces
  choca con el antivirus.
- `console=True`: la ventana negra es el control de apagado del motor y el
  lugar donde aparece cualquier error. Ocultarla deja al usuario sin forma de
  cerrar el servidor ni de reportar qué falló.

---

## Detalles de comportamiento

**Puerto.** Usa el 8501. Si está ocupado — por otra instancia, o por un
`streamlit run` abierto — pide uno libre al sistema y abre el navegador ahí.
Dos copias del gemelo pueden correr a la vez sin pisarse.

**Datos.** El histórico es sintético y se genera en memoria en cada corrida
(`FUENTE_DATOS = "sintetico"`). El ejecutable no escribe nada en disco ni
guarda estado entre sesiones: cerrar y volver a abrir es empezar de cero.

**Ícono.** Si se coloca un archivo `icono.ico` en la raíz del proyecto antes de
compilar, el spec lo detecta y lo usa. Sin él, queda el ícono genérico de
PyInstaller.

**PostgreSQL.** Cuando el sistema web de la tesis esté operativo y haya que
cambiar `FUENTE_DATOS` a `"postgres"`, hay que agregar `psycopg2` y
`sqlalchemy` tanto a `requirements.txt` como a `hiddenimports` en el spec,
y recompilar.

---

## Si algo falla

| Síntoma | Causa habitual |
|---|---|
| La ventana negra aparece y se cierra al instante | Error de arranque. Abrir PowerShell en la carpeta y correr `.\GemeloInventarioSOAZ.exe` para leer el traceback antes de que desaparezca |
| `ModuleNotFoundError` al usar una pestaña | Falta esa dependencia en `hiddenimports` del spec |
| El navegador abre y queda en blanco | Faltó la carpeta `static\` de Streamlit — verificar que `collect_all("streamlit")` esté en el spec |
| «No se encontró app_motor.py» | La carpeta `_internal\app\` no se empaquetó; revisar la sección `datas` del spec |
