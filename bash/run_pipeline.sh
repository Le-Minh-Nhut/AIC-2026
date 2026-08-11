#!/usr/bin/env bash
# Usage: RUN_ENCODE_PE=0 DEVICE=cuda bash bash/run_pipeline.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data}"
export PYTHON_BIN DATA_ROOT

RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
RUN_ENCODE_FG="${RUN_ENCODE_FG:-1}"
RUN_ENCODE_PE="${RUN_ENCODE_PE:-1}"
RUN_BUILD_FG_INDEX="${RUN_BUILD_FG_INDEX:-1}"
RUN_BUILD_PE_INDEX="${RUN_BUILD_PE_INDEX:-1}"

run_stage() {
  local name="$1"
  local enabled="$2"
  local script_path="$3"
  case "${enabled,,}" in
    1|true|yes)
      echo "==> Running ${name}"
      bash "$script_path"
      ;;
    0|false|no)
      echo "==> Skipping ${name}"
      ;;
    *)
      echo "ERROR: ${name} toggle must be 1/0, true/false, or yes/no; received ${enabled}" >&2
      exit 2
      ;;
  esac
}

run_stage download "$RUN_DOWNLOAD" "$REPO_ROOT/bash/download_data.sh"
run_stage extract "$RUN_EXTRACT" "$REPO_ROOT/bash/extract_data.sh"
run_stage analyze "$RUN_ANALYZE" "$REPO_ROOT/bash/analyze_data.sh"
run_stage encode_fgclip2 "$RUN_ENCODE_FG" "$REPO_ROOT/bash/encode_fgclip2.sh"
run_stage encode_pecore "$RUN_ENCODE_PE" "$REPO_ROOT/bash/encode_pecore.sh"
run_stage build_fg_index "$RUN_BUILD_FG_INDEX" "$REPO_ROOT/bash/build_fg_index.sh"
run_stage build_pe_index "$RUN_BUILD_PE_INDEX" "$REPO_ROOT/bash/build_pe_index.sh"
