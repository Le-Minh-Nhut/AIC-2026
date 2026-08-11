#!/usr/bin/env bash
# Usage: DEVICE=cuda PE_CHECKPOINT=/models/PE-Core-G14-448.pt bash bash/encode_pecore.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-cuda}"
PE_OUTPUT="${PE_OUTPUT:-}"
PE_CHECKPOINT="${PE_CHECKPOINT:-}"
PE_MODEL_CONFIG="${PE_MODEL_CONFIG:-}"
PE_BATCH_SIZE="${PE_BATCH_SIZE:-}"
PE_SHARD_SIZE="${PE_SHARD_SIZE:-}"
PE_STORAGE_DTYPE="${PE_STORAGE_DTYPE:-float16}"
PE_NO_RESUME="${PE_NO_RESUME:-0}"

args=(scripts/encode_keyframes.py --encoder pecore_g14_448 --data-root "$DATA_ROOT" --device "$DEVICE" --storage-dtype "$PE_STORAGE_DTYPE")
[[ -n "$PE_OUTPUT" ]] && args+=(--output "$PE_OUTPUT")
[[ -n "$PE_CHECKPOINT" ]] && args+=(--checkpoint "$PE_CHECKPOINT")
[[ -n "$PE_MODEL_CONFIG" ]] && args+=(--model-config "$PE_MODEL_CONFIG")
[[ -n "$PE_BATCH_SIZE" ]] && args+=(--batch-size "$PE_BATCH_SIZE")
[[ -n "$PE_SHARD_SIZE" ]] && args+=(--shard-size "$PE_SHARD_SIZE")
[[ "$PE_NO_RESUME" == "1" ]] && args+=(--no-resume)

exec "$PYTHON_BIN" "${args[@]}" "$@"
