#!/usr/bin/env bash
set -euo pipefail

NO_INSTALL=false
NO_AGENT=false
KILL_PORTS=false

for arg in "$@"; do
  case $arg in
    --no-install) NO_INSTALL=true ;;
    --no-agent)   NO_AGENT=true ;;
    --kill-ports) KILL_PORTS=true ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

info()  { echo "[demo] $*"; }
warn()  { echo "[demo] WARN: $*"; }
fail()  { echo "[demo] ERROR: $*" >&2; exit 1; }

if [ ! -f ".env" ]; then
  warn "No existe .env."
  if [ -f ".env.example" ]; then
    cp .env.example .env
    warn "Copiado .env.example -> .env. Edita .env y pega tu GOOGLE_API_KEY."
  else
    fail "No existe .env ni .env.example"
  fi
fi

if [ ! -d ".venv" ]; then
  info "Creando entorno virtual (.venv)"
  python3 -m venv .venv
fi

PY="$ROOT/.venv/bin/python"
[ -f "$PY" ] || fail "No se encontró Python del venv en: $PY"

if [ "$NO_INSTALL" = false ]; then
  info "Instalando dependencias"
  "$PY" -m pip install -r requirements.txt --quiet
else
  info "Saltando instalación (--no-install)"
fi

kill_port() {
  local port=$1
  local pid
  pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    warn "Liberando puerto $port (PID=$pid)"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

if [ "$KILL_PORTS" = true ]; then
  info "Liberando puertos 8001/8002/8003"
  kill_port 8001
  kill_port 8002
  kill_port 8003
  sleep 1
fi

start_service() {
  local name=$1
  local path=$2
  info "Iniciando $name"
  "$PY" "$path" &
}

start_service "MCP Emociones  :8001" "services/emociones/main.py"
start_service "MCP Metricas   :8002" "services/metricas/main.py"
start_service "MCP Propagacion:8003" "services/propagacion/main.py"

info "Servicios iniciados. Esperando 2s..."
sleep 2

if [ "$NO_AGENT" = false ]; then
  info "Iniciando chat del agente (CTRL+C para salir)"
  "$PY" agent/chat.py
else
  info "Listo. Para abrir el chat: $PY agent/chat.py"
  wait
fi
