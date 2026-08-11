#!/usr/bin/env bash
# Usage: bash bash/profile.sh python scripts/run_kis.py --coarse-only --query "..."
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${SEED:-2026}"
STORAGE_PATH="${STORAGE_PATH:-data/processed}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-outputs/profiles/pipeline.json}"

if [[ "$#" -eq 0 ]]; then
  echo "ERROR: pass the pipeline command to profile after bash/profile.sh" >&2
  exit 2
fi

exec "$PYTHON_BIN" scripts/profile_pipeline.py --data-root "$DATA_ROOT" --seed "$SEED" --storage-path "$STORAGE_PATH" --output "$PROFILE_OUTPUT" -- "$@"
