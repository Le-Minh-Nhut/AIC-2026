#!/usr/bin/env bash
# Usage: DATA_ROOT=data bash bash/build_fg_index.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
FG_EMBEDDING_MANIFEST="${FG_EMBEDDING_MANIFEST:-}"
FG_INDEX_OUTPUT="${FG_INDEX_OUTPUT:-}"
FG_INDEX_BATCH_SIZE="${FG_INDEX_BATCH_SIZE:-}"
FG_INDEX_OVERWRITE="${FG_INDEX_OVERWRITE:-0}"

args=(scripts/build_indexes.py --encoder fgclip2_large --data-root "$DATA_ROOT")
[[ -n "$FG_EMBEDDING_MANIFEST" ]] && args+=(--embedding-manifest "$FG_EMBEDDING_MANIFEST")
[[ -n "$FG_INDEX_OUTPUT" ]] && args+=(--output "$FG_INDEX_OUTPUT")
[[ -n "$FG_INDEX_BATCH_SIZE" ]] && args+=(--batch-size "$FG_INDEX_BATCH_SIZE")
[[ "$FG_INDEX_OVERWRITE" == "1" ]] && args+=(--overwrite)

exec "$PYTHON_BIN" "${args[@]}" "$@"
