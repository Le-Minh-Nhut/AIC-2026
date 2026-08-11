#!/usr/bin/env bash
# Usage: DEVICE=cuda PE_CHECKPOINT=/models/pe.pt VLM_CHECKPOINT=/models/qwen bash bash/run_web.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${WEB_HOST:-127.0.0.1}"
PORT="${WEB_PORT:-8000}"
WEB_MODE="${WEB_MODE:-backend}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

export AIC_API_REPO_ROOT="$REPO_ROOT"
export AIC_API_DATA_ROOT="${DATA_ROOT:-data}"
export AIC_API_DEVICE="${DEVICE:-cpu}"
export AIC_API_PE_CHECKPOINT="${PE_CHECKPOINT:-}"
export AIC_API_VLM_CHECKPOINT="${VLM_CHECKPOINT:-}"
export AIC_API_FG_MODEL_ID="${FG_MODEL_ID:-}"
export AIC_API_FG_REVISION="${FG_REVISION:-}"

if [[ "$WEB_MODE" == "full" ]]; then
  (cd web/frontend && npm run dev -- --host "$HOST" --port "$FRONTEND_PORT") &
  frontend_pid=$!
  trap 'kill "$frontend_pid" 2>/dev/null || true' EXIT INT TERM
fi

reload_args=()
if [[ "${WEB_RELOAD:-0}" == "1" ]]; then
  reload_args=(--reload)
fi

if [[ "$WEB_MODE" == "full" ]]; then
  "$PYTHON_BIN" -m uvicorn api.app:app --app-dir "$REPO_ROOT/src" \
    --host "$HOST" --port "$PORT" "${reload_args[@]}"
elif [[ "$WEB_MODE" == "backend" ]]; then
  exec "$PYTHON_BIN" -m uvicorn api.app:app --app-dir "$REPO_ROOT/src" \
    --host "$HOST" --port "$PORT" "${reload_args[@]}"
else
  echo "WEB_MODE must be backend or full" >&2
  exit 2
fi
