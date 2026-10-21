#!/usr/bin/env bash
# Launch external-teacher AOPD on math; select free GPUs with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --multi_gpu --num_processes "${NUM_GPUS:-8}" --module external_teacher.train_math \
  --model_path "${MODEL:-Qwen/Qwen3-1.7B}" \
  --exp "${RUN_NAME:-aopd_external_math}" \
  --output_dir_base "${OUTPUT_DIR:-outputs}" \
  --mu "${MU:-.08}" "$@"
