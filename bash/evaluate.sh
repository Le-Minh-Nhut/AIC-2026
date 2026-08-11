#!/usr/bin/env bash
# Usage: GROUND_TRUTH=data/dev_ground_truth.json SUBMISSION=outputs/submissions/dev.json bash bash/evaluate.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GROUND_TRUTH="${GROUND_TRUTH:?Set GROUND_TRUTH to local development ground truth JSON}"
SUBMISSION="${SUBMISSION:?Set SUBMISSION to versioned submission JSON}"
TASK="${TASK:-}"
EVALUATION_OUTPUT="${EVALUATION_OUTPUT:-}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
DATASET_SNAPSHOT="${DATASET_SNAPSHOT:-}"
RUNTIME_METADATA="${RUNTIME_METADATA:-}"
PIPELINE_LATENCY_MS="${PIPELINE_LATENCY_MS:-}"
NOTES="${NOTES:-}"

args=(scripts/evaluate.py --ground-truth "$GROUND_TRUTH" --submission "$SUBMISSION")
[[ -n "$TASK" ]] && args+=(--task "$TASK")
[[ -n "$EVALUATION_OUTPUT" ]] && args+=(--output "$EVALUATION_OUTPUT")
[[ -n "$EXPERIMENT_LOG" ]] && args+=(--experiment-log "$EXPERIMENT_LOG")
[[ -n "$EXPERIMENT_ID" ]] && args+=(--experiment-id "$EXPERIMENT_ID")
[[ -n "$DATASET_SNAPSHOT" ]] && args+=(--dataset-snapshot "$DATASET_SNAPSHOT")
[[ -n "$RUNTIME_METADATA" ]] && args+=(--runtime-metadata "$RUNTIME_METADATA")
[[ -n "$PIPELINE_LATENCY_MS" ]] && args+=(--pipeline-latency-ms "$PIPELINE_LATENCY_MS")
[[ -n "$NOTES" ]] && args+=(--notes "$NOTES")

exec "$PYTHON_BIN" "${args[@]}" "$@"
