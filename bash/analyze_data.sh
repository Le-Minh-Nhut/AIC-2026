#!/usr/bin/env bash
# Usage: DATA_ROOT=data ANALYZE_STRICT=1 bash bash/analyze_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
ANALYZE_STRICT="${ANALYZE_STRICT:-1}"
ANALYSIS_REPORT="${ANALYSIS_REPORT:-}"
SAMPLE_DECODE_VIDEOS="${SAMPLE_DECODE_VIDEOS:-}"
SAMPLE_IMAGES="${SAMPLE_IMAGES:-}"

args=(scripts/analyze_data.py --data-root "$DATA_ROOT")
[[ "$ANALYZE_STRICT" == "1" ]] && args+=(--strict)
[[ -n "$ANALYSIS_REPORT" ]] && args+=(--report "$ANALYSIS_REPORT")
[[ -n "$SAMPLE_DECODE_VIDEOS" ]] && args+=(--sample-decode-videos "$SAMPLE_DECODE_VIDEOS")
[[ -n "$SAMPLE_IMAGES" ]] && args+=(--sample-images "$SAMPLE_IMAGES")

exec "$PYTHON_BIN" "${args[@]}" "$@"
