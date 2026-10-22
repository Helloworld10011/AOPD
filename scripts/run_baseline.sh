#!/usr/bin/env bash
# Launch a separate OPD, OPSD, or GRPO entry point with the selected task arguments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
baseline=${1:?Usage: run_baseline.sh opd|opsd|grpo MODEL --task TASK [arguments]}
model=${2:?Supply the student model as the second argument.}
shift 2
case "$baseline" in
  opd) module=training.external_teacher.opd_train ;;
  opsd) module=training.self_distillation.opsd_train ;;
  grpo) module=training.grpo_train ;;
  *) echo 'Choose opd, opsd, or grpo.' >&2; exit 2 ;;
esac
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
num_gpus=${NUM_GPUS:-8}
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then echo 'NUM_GPUS must be a positive integer.' >&2; exit 2; fi
launch=(--num_processes "$num_gpus" --mixed_precision bf16)
if (( num_gpus > 1 )); then launch+=(--multi_gpu); fi
accelerate launch "${launch[@]}" --module "$module" --model "$model" "$@"
