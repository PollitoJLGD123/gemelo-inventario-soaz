# construir_exe.ps1 — Compila el ejecutable de Windows del Gemelo Digital SOAZ.
#
#   powershell -ExecutionPolicy Bypass -File construir_exe.ps1
#
# Resultado: dist\GemeloInventarioSOAZ\GemeloInventarioSOAZ.exe
# La carpeta dist\GemeloInventarioSOAZ completa es lo que se distribuye: el .exe
# solo no funciona, necesita el _internal\ que lo acompaña.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No existe el entorno virtual. Creandolo..." -ForegroundColor Yellow
    python -m venv venv
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
}

Write-Host "Verificando dependencias de compilacion..." -ForegroundColor Cyan
& $python -m pip install --quiet --upgrade pyinstaller
if (-not $?) { throw "No se pudo instalar PyInstaller" }

# Una copia del gemelo abierta bloquea dist\ y hace fallar el borrado con un
# "Device or resource busy" que no dice de que se trata.
$abiertos = Get-Process GemeloInventarioSOAZ -ErrorAction SilentlyContinue
if ($abiertos) {
    Write-Host "Cerrando $($abiertos.Count) instancia(s) del gemelo en ejecucion..." -ForegroundColor Yellow
    $abiertos | Stop-Process -Force
    Start-Sleep -Seconds 2
}

Write-Host "Limpiando compilaciones anteriores..." -ForegroundColor Cyan
foreach ($carpeta in @("build", "dist")) {
    if (Test-Path $carpeta) { Remove-Item -Recurse -Force $carpeta }
}

Write-Host "Compilando (tarda entre 2 y 5 minutos)..." -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean gemelo_soaz.spec
if (-not $?) { throw "La compilacion fallo" }

$exe = Join-Path $PSScriptRoot "dist\GemeloInventarioSOAZ\GemeloInventarioSOAZ.exe"
if (-not (Test-Path $exe)) { throw "No se genero el ejecutable en $exe" }

$tam = "{0:N0} MB" -f ((Get-ChildItem -Recurse "dist\GemeloInventarioSOAZ" |
    Measure-Object -Property Length -Sum).Sum / 1MB)

Write-Host ""
Write-Host "Listo." -ForegroundColor Green
Write-Host "  Ejecutable : $exe"
Write-Host "  Distribuir : dist\GemeloInventarioSOAZ\  (carpeta completa, $tam)"
Write-Host ""
