#!/usr/bin/env bash
# AOPD math with an external teacher.
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
    --module training.external_teacher.aopd_train \
    --task math \
    --model_name_or_path "$model" \
    --output_dir "outputs/aopd_external_math" \
    --run_name "aopd_external_math" \
    --teacher_model Qwen/Qwen3-4B-Instruct-2507 \
    --learning_rate 2e-5 \
    --max_steps 450 \
    --max_length 9500 \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --num_generations 4 \
    --loss_top_k 0 \
    --loss_add_tail True \
    --student_thinking False \
    --atd_w_clip 4 \
    --atd_mu .08 \
    --beta 0 \
    --lmbda 1 \
    --temperature .8 \
    --top_p 1 \
    --top_k -1 \
    --lr_scheduler_type cosine \
    --warmup_ratio .05 \
    --weight_decay 0 \
    --max_grad_norm .1 \
    --bf16 True \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --disable_dropout True \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
    --attn_implementation flash_attention_2 \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization .2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_sleep_mode False \
    --eval_strategy no \
    --save_steps 25 \
    --save_only_model True \
    --logging_steps 10 \
    --seed 42 \
    --report_to wandb \
    --logging_first_step True \
    "$@"
