#!/usr/bin/env bash
# Usage: DATA_ROOT=data bash bash/build_pe_index.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
PE_EMBEDDING_MANIFEST="${PE_EMBEDDING_MANIFEST:-}"
PE_INDEX_OUTPUT="${PE_INDEX_OUTPUT:-}"
PE_INDEX_BATCH_SIZE="${PE_INDEX_BATCH_SIZE:-}"
PE_INDEX_OVERWRITE="${PE_INDEX_OVERWRITE:-0}"

args=(scripts/build_indexes.py --encoder pecore_g14_448 --data-root "$DATA_ROOT")
[[ -n "$PE_EMBEDDING_MANIFEST" ]] && args+=(--embedding-manifest "$PE_EMBEDDING_MANIFEST")
[[ -n "$PE_INDEX_OUTPUT" ]] && args+=(--output "$PE_INDEX_OUTPUT")
[[ -n "$PE_INDEX_BATCH_SIZE" ]] && args+=(--batch-size "$PE_INDEX_BATCH_SIZE")
[[ "$PE_INDEX_OVERWRITE" == "1" ]] && args+=(--overwrite)

exec "$PYTHON_BIN" "${args[@]}" "$@"
