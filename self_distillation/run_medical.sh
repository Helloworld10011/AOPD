#!/usr/bin/env bash
# Retained released-MedMCQA launcher; model and output are configurable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch \
    --config_file self_distillation/accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 12949 \
    --module self_distillation.train_medical \
    --model_name_or_path "${MODEL:-Qwen/Qwen3-4B-Instruct-2507}" \
    --learning_rate 2e-5 \
    --max_grad_norm 0.1 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --atd_num_generations 4 \
    --steps_per_generation 4 \
    --dataloader_drop_last True \
    --output_dir "${OUTPUT_DIR:-outputs}" \
    --run_config atd_opsd_medqa_qwen1_7b \
    --max_steps 300 \
    --max_completion_length 2048 \
    --eval_max_new_tokens 2048 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 0.8 \
    --top_p 1.0 \
    --top_k -1 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.00 \
    --student_thinking False \
    --teacher_thinking False \
    --top_k_loss 64 \
    --atd_w_clip 4.0 \
    --eval_strategy no \
    --save_only_model True \
    --wandb_project RLVR \
    "$@"
