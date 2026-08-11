#!/usr/bin/env bash
# Usage: DEVICE=cuda bash bash/encode_fgclip2.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-cuda}"
FG_OUTPUT="${FG_OUTPUT:-}"
FG_MODEL_ID="${FG_MODEL_ID:-}"
FG_REVISION="${FG_REVISION:-}"
FG_BATCH_SIZE="${FG_BATCH_SIZE:-}"
FG_SHARD_SIZE="${FG_SHARD_SIZE:-}"
FG_STORAGE_DTYPE="${FG_STORAGE_DTYPE:-float16}"
FG_NO_RESUME="${FG_NO_RESUME:-0}"

args=(scripts/encode_keyframes.py --encoder fgclip2_large --data-root "$DATA_ROOT" --device "$DEVICE" --storage-dtype "$FG_STORAGE_DTYPE")
[[ -n "$FG_OUTPUT" ]] && args+=(--output "$FG_OUTPUT")
[[ -n "$FG_MODEL_ID" ]] && args+=(--model-id "$FG_MODEL_ID")
[[ -n "$FG_REVISION" ]] && args+=(--revision "$FG_REVISION")
[[ -n "$FG_BATCH_SIZE" ]] && args+=(--batch-size "$FG_BATCH_SIZE")
[[ -n "$FG_SHARD_SIZE" ]] && args+=(--shard-size "$FG_SHARD_SIZE")
[[ "$FG_NO_RESUME" == "1" ]] && args+=(--no-resume)

exec "$PYTHON_BIN" "${args[@]}" "$@"
