#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

stage="${1:-all}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/train_logit_kd.sh [all|initial|continue]" >&2
  exit 2
fi
case "$stage" in
  all) configs=(qwen_logit_kd qwen_logit_kd_continue) ;;
  initial) configs=(qwen_logit_kd) ;;
  continue) configs=(qwen_logit_kd_continue) ;;
  *) echo "Usage: bash scripts/train_logit_kd.sh [all|initial|continue]" >&2; exit 2 ;;
esac

"${PYTHON:-python3}" -c 'import torch; assert torch.cuda.is_available(), "Logit KD training requires a CUDA GPU with the current model loaders."'

for config in "${configs[@]}"; do
  "${PYTHON:-python3}" scripts/train.py --config "configs/${config}.yaml"
done
