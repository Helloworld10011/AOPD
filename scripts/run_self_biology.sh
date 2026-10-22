#!/usr/bin/env bash
# AOPD biology with a reference-conditioned self-teacher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model=${1:-Qwen/Qwen3-4B-Instruct-2507}
if (( $# )); then shift; fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --multi_gpu --num_processes "${NUM_GPUS:-8}" --mixed_precision bf16 \
  --module training.self_distillation.train --task biology --model "$model" "$@"
