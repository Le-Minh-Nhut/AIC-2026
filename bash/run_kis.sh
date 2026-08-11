#!/usr/bin/env bash
# Usage: QUERY="a person opens a door" DEVICE=cuda bash bash/run_kis.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
QUERY="${QUERY:?Set QUERY to a KIS event description}"
ENCODER="${ENCODER:-fg_pe_fusion}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-2026}"
TOP_K="${TOP_K:-100}"
PE_CHECKPOINT="${PE_CHECKPOINT:-}"
COARSE_ONLY="${COARSE_ONLY:-0}"
NO_TEMPORAL_NMS="${NO_TEMPORAL_NMS:-0}"
DEBUG_OUTPUT="${DEBUG_OUTPUT:-}"

args=(scripts/run_kis.py --query "$QUERY" --encoder "$ENCODER" --data-root "$DATA_ROOT" --device "$DEVICE" --seed "$SEED" --top-k "$TOP_K")
[[ -n "$PE_CHECKPOINT" ]] && args+=(--pe-checkpoint "$PE_CHECKPOINT")
[[ "$COARSE_ONLY" == "1" ]] && args+=(--coarse-only)
[[ "$NO_TEMPORAL_NMS" == "1" ]] && args+=(--no-temporal-nms)
[[ -n "$DEBUG_OUTPUT" ]] && args+=(--debug-output "$DEBUG_OUTPUT")

exec "$PYTHON_BIN" "${args[@]}" "$@"
