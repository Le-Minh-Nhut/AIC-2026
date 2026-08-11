#!/usr/bin/env bash
# Usage: EVENT_DESCRIPTION="..." QUESTION="..." VLM_CHECKPOINT=/models/Qwen3-VL bash bash/run_qna.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
EVENT_DESCRIPTION="${EVENT_DESCRIPTION:?Set EVENT_DESCRIPTION for retrieval}"
QUESTION="${QUESTION:?Set QUESTION for the VLM}"
ENCODER="${ENCODER:-fg_pe_fusion}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-2026}"
PE_CHECKPOINT="${PE_CHECKPOINT:-}"
VLM_CHECKPOINT="${VLM_CHECKPOINT:-}"
QUERY_ID="${QUERY_ID:-}"
COARSE_ONLY="${COARSE_ONLY:-0}"
DEBUG_OUTPUT="${DEBUG_OUTPUT:-}"

args=(scripts/run_qna.py --event-description "$EVENT_DESCRIPTION" --question "$QUESTION" --encoder "$ENCODER" --data-root "$DATA_ROOT" --device "$DEVICE" --seed "$SEED")
[[ -n "$PE_CHECKPOINT" ]] && args+=(--pe-checkpoint "$PE_CHECKPOINT")
[[ -n "$VLM_CHECKPOINT" ]] && args+=(--vlm-checkpoint "$VLM_CHECKPOINT")
[[ -n "$QUERY_ID" ]] && args+=(--query-id "$QUERY_ID")
[[ "$COARSE_ONLY" == "1" ]] && args+=(--coarse-only)
[[ -n "$DEBUG_OUTPUT" ]] && args+=(--debug-output "$DEBUG_OUTPUT")

exec "$PYTHON_BIN" "${args[@]}" "$@"
