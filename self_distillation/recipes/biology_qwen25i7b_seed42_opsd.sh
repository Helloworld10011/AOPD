#!/usr/bin/env bash
# Recorded main recipe: biology_qwen25i7b_seed42_opsd. No training was run during packaging.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch \
  --config_file self_distillation/accelerate.yaml \
  --num_processes 8 \
  --gradient_accumulation_steps 1 \
  --main_process_port 12949 \
  --module self_distillation.baseline_biology \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --model_revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --learning_rate 5e-06 \
  --lr_scheduler_type constant \
  --warmup_ratio 0.0 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --output_dir outputs \
  --run_config biology_qwen25i7b_seed42_opsd \
  --max_steps 400 \
  --max_completion_length 1024 \
  --save_steps 25 \
  --logging_steps 2 \
  --seed 42 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.4 \
  --vllm_tensor_parallel_size 1 \
  --lora_r 64 \
  --lora_alpha 128 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k -1 \
  --lmbda 1 \
  --jsd_token_clip 0 \
  --student_thinking False \
  --teacher_thinking False \
  --top_k_loss 0 \
  --eval_strategy no \
  --save_only_model False \
  --wandb_project RLVR \
  --reward_eval_steps 25 \
  --eval_max_new_tokens 4096 \
  --report_to wandb \
  --disable_tqdm True \
  --science_data_dir data/biology \
  --gradient_checkpointing --use_vllm \
  --use_peft --fixed_teacher \
  --lora_target_modules q_proj \
  k_proj v_proj \
  o_proj gate_proj \
  up_proj down_proj \
  "$@"
