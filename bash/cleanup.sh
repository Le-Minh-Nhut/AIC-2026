#!/usr/bin/env bash
# Usage: CLEANUP_TARGET=pecore_g14_448 bash bash/cleanup.sh; add CLEANUP_DELETE=1 to delete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
CLEANUP_TARGET="${CLEANUP_TARGET:-}"
CLEANUP_DELETE="${CLEANUP_DELETE:-0}"

args=(scripts/cleanup_storage.py --data-root "$DATA_ROOT")
[[ -n "$CLEANUP_TARGET" ]] && args+=(--encoder "$CLEANUP_TARGET")
[[ "$CLEANUP_DELETE" == "1" ]] && args+=(--delete)

exec "$PYTHON_BIN" "${args[@]}" "$@"
