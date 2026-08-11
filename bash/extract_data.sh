#!/usr/bin/env bash
# Usage: DATA_ROOT=data EXTRACT_SCOPE=all bash bash/extract_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
EXTRACT_SCOPE="${EXTRACT_SCOPE:-all}"
DELETE_ARCHIVES="${DELETE_ARCHIVES:-0}"

args=(scripts/extract_data.py --data-root "$DATA_ROOT" --only "$EXTRACT_SCOPE")
[[ "$DELETE_ARCHIVES" == "1" ]] && args+=(--delete-archives)

exec "$PYTHON_BIN" "${args[@]}" "$@"
