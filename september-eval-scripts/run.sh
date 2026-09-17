#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mode="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  all) modes=(no_think think) ;;
  think|no_think) modes=("$mode") ;;
  *) echo "Usage: bash september-eval-scripts/run.sh [all|think|no_think] [eval CLI options]" >&2; exit 2 ;;
esac

# Separate each invocation's results, including smoke checks and repeated runs.
run_root="${EVAL_OUTPUT_ROOT:-./eval_outputs/september/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
for selected_mode in "${modes[@]}"; do
  "${PYTHON:-python3}" -m custom_eval.run_eval \
    --config "$SCRIPT_DIR/qwen_2B-${selected_mode}.yaml" \
    --output-dir "$run_root/$selected_mode" "$@"
done
