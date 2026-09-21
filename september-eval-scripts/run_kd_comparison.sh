#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Only the two requested epoch_1 checkpoints, in reasoning mode.
run_root="${EVAL_OUTPUT_ROOT:-./eval_outputs/september-kd-comparison/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
"${PYTHON:-python3}" -m custom_eval.run_eval \
  --config "$SCRIPT_DIR/qwen_kd_comparison-think.yaml" \
  --output-dir "$run_root/think" "$@"
