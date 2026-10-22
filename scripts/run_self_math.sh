#!/usr/bin/env bash
# AOPD math with a reference-conditioned self-teacher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model=${1:-Qwen/Qwen3-1.7B}
if (( $# )); then shift; fi
model_name=${model##*/}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=${WANDB_PROJECT:-RLVR}
num_gpus=${NUM_GPUS:-8}
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer." >&2
    exit 2
fi
launch=(--num_processes "$num_gpus" --mixed_precision bf16)
if (( num_gpus > 1 )); then launch+=(--multi_gpu); fi

accelerate launch "${launch[@]}" \
    --module training.self_distillation.aopd_train \
    --task math \
    --model_name_or_path "$model" \
    --output_dir "outputs/aopd_math_${model_name}" \
    --run_name "aopd_math_${model_name}" \
    --fixed_teacher True \
    --student_thinking False \
    --teacher_thinking False \
    --reason_first False \
    --use_ema_teacher False \
    --learning_rate 2e-5 \
    --max_steps 400 \
    --max_length 35000 \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 4 \
    --dataloader_drop_last True \
    --atd_num_generations 4 \
    --top_k_loss 0 \
    --atd_w_clip 4 \
    --atd_mu 0 \
    --beta 0 \
    --lmbda 1 \
    --temperature .8 \
    --top_p 1 \
    --top_k -1 \
    --lr_scheduler_type linear \
    --warmup_ratio 0 \
    --weight_decay 0 \
    --max_grad_norm .1 \
    --bf16 True \
    --gradient_checkpointing True \
    --disable_dropout True \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout .05 \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
    --attn_implementation flash_attention_2 \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization .3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_sleep_mode False \
    --eval_strategy no \
    --save_steps 25 \
    --save_only_model True \
    --logging_steps 2 \
    --seed 42 \
    --report_to wandb \
    "$@"
