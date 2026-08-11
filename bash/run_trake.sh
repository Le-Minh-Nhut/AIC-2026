#!/usr/bin/env bash
# Usage: QUERY="approach -> takeoff -> landing" DEVICE=cuda bash bash/run_trake.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
QUERY="${QUERY:?Set QUERY to an ordered TRAKE event description}"
ENCODER="${ENCODER:-fg_pe_fusion}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-2026}"
PE_CHECKPOINT="${PE_CHECKPOINT:-}"
COARSE_ONLY="${COARSE_ONLY:-0}"
DEBUG_OUTPUT="${DEBUG_OUTPUT:-}"

args=(scripts/run_trake.py --query "$QUERY" --encoder "$ENCODER" --data-root "$DATA_ROOT" --device "$DEVICE" --seed "$SEED")
[[ -n "$PE_CHECKPOINT" ]] && args+=(--pe-checkpoint "$PE_CHECKPOINT")
[[ "$COARSE_ONLY" == "1" ]] && args+=(--coarse-only)
[[ -n "$DEBUG_OUTPUT" ]] && args+=(--debug-output "$DEBUG_OUTPUT")

exec "$PYTHON_BIN" "${args[@]}" "$@"
