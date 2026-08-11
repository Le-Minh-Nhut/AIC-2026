#!/usr/bin/env bash
# Usage: PYTHON_BIN=python3 bash bash/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,btc-clip,fgclip2,pecore,faiss,refinement,qwen3-vl,web}"
PERCEPTION_MODELS_PATH="${PERCEPTION_MODELS_PATH:-}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e ".[${INSTALL_EXTRAS}]"

if [[ -n "$PERCEPTION_MODELS_PATH" ]]; then
  "$VENV_PYTHON" -m pip install -e "$PERCEPTION_MODELS_PATH"
fi
