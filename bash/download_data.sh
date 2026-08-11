#!/usr/bin/env bash
# Usage: DATA_ROOT=data DOWNLOAD_SCOPE=all bash bash/download_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
DOWNLOAD_SCOPE="${DOWNLOAD_SCOPE:-all}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-2}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-}"
SOURCE_URL="${SOURCE_URL:-}"

args=(scripts/download_data.py --data-root "$DATA_ROOT" --only "$DOWNLOAD_SCOPE" --workers "$DOWNLOAD_WORKERS")
[[ -n "$DOWNLOAD_TIMEOUT" ]] && args+=(--timeout "$DOWNLOAD_TIMEOUT")
[[ -n "$DOWNLOAD_RETRIES" ]] && args+=(--retries "$DOWNLOAD_RETRIES")
[[ -n "$SOURCE_URL" ]] && args+=(--source-url "$SOURCE_URL")

exec "$PYTHON_BIN" "${args[@]}" "$@"
