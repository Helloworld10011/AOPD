#!/usr/bin/env bash
# Original EvalPlus/vLLM route: greedy pass@1; sampled pass@10 requires a separate n>=10 run.
set -euo pipefail
model=${1:?Usage: bash evaluation/humaneval.sh MODEL [extra EvalPlus options]}
shift
exec evalplus.evaluate --model "$model" --dataset humaneval --backend vllm \
  --tp "${NUM_GPUS:-1}" --greedy --root results/humaneval "$@"
