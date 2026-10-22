#!/usr/bin/env bash
# AOPD code with an external teacher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model=${1:-Qwen/Qwen3-1.7B}
if (( $# )); then shift; fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
num_gpus=${NUM_GPUS:-8}
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NUM_GPUS must be a positive integer.' >&2
  exit 2
fi
launch=(--num_processes "$num_gpus" --mixed_precision bf16)
if (( num_gpus > 1 )); then launch+=(--multi_gpu); fi
accelerate launch "${launch[@]}" \
  --module training.external_teacher.aopd_train --task code --model "$model" "$@"
