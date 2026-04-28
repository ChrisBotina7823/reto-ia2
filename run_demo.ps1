param(
  [switch]$NoAgent,
  [switch]$NoInstall,
  [switch]$KillPorts
)

$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "[demo] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[demo] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[demo] $msg" -ForegroundColor Red; exit 1 }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Info "Root: $root"

if (-not (Test-Path ".env")) {
  Warn "No existe .env. Copiando .env.example -> .env"
  if (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env" -Force
    Warn "Edita .env y pega tu GOOGLE_API_KEY antes de continuar."
  } else {
    Fail "No existe .env ni .env.example"
  }
}

if (-not (Test-Path ".venv")) {
  Info "Creando entorno virtual (.venv)"
  py -3.11 -m venv .venv
}

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Fail "No se encontró Python del venv en: $py"
}

if (-not $NoInstall) {
  Info "Instalando dependencias (requirements.txt)"
  & $py -m pip install -r requirements.txt | Out-Host
} else {
  Info "Saltando instalación de dependencias (-NoInstall)"
}

function Stop-ListeningPort([int]$port) {
  try {
    $pids = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique)
  } catch {
    $pids = @()
  }

  if (-not $pids -or $pids.Count -eq 0) {
    return
  }

  foreach ($procId in $pids) {
    try {
      $proc = Get-Process -Id $procId -ErrorAction Stop
      Warn "Liberando puerto $port (PID=$procId, $($proc.ProcessName))"
      Stop-Process -Id $procId -Force -ErrorAction Stop
    } catch {
      Warn "No pude detener PID=$procId para puerto ${port}: $($_.Exception.Message)"
    }
  }
}

function Start-McpService([string]$name, [string]$path, [int]$port) {
  # Usar EncodedCommand para evitar problemas de escape con espacios en rutas.
  $cmdText = "& '$py' '$path'"
  $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmdText))
  $title = "MCP-$name :$port"
  Info "Iniciando $title"
  Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoExit",
    "-EncodedCommand",
    $encoded
  ) -WindowStyle Normal | Out-Null
}

# Si los puertos ya están ocupados por corridas previas, puedes liberarlos automáticamente.
if ($KillPorts) {
  Info "Liberando puertos 8001/8002/8003 (-KillPorts)"
  Stop-ListeningPort 8001
  Stop-ListeningPort 8002
  Stop-ListeningPort 8003
  Start-Sleep -Seconds 1
}

# Levanta los 3 MCP en ventanas separadas.
Start-McpService -name "emociones" -path "services\emociones\main.py" -port 8001
Start-McpService -name "metricas" -path "services\metricas\main.py" -port 8002
Start-McpService -name "propagacion" -path "services\propagacion\main.py" -port 8003

Info "Servicios iniciados. Esperando 2s para que levanten..."
Start-Sleep -Seconds 2

if (-not $NoAgent) {
  Info "Iniciando chat del agente (CTRL+C para salir)"
  & $py "agent\chat.py"
} else {
  Info "Listo. (No se inició el agente por -NoAgent)"
  Info "Para abrir el chat: $py agent\chat.py"
}

