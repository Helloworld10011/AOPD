# Plain on-policy distillation from an external teacher, with forward or reverse KL.
import argparse
from pathlib import Path
from .tasks import load_training_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['math', 'code'], required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--divergence', choices=['fkl', 'rkl'], default='fkl')
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--data_dir', default='data/external_code')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--top_k_loss', type=int, default=0)
    parser.add_argument('--report_to', default='none')
    args = parser.parse_args(argv)
    args.max_steps = (450 if args.task == 'math' else 350) if args.max_steps is None else args.max_steps
    args.output_dir = args.output_dir or f'outputs/opd_{args.divergence}_{args.task}_{args.model.split("/")[-1]}'
    if min(args.max_steps, args.batch_size, args.accumulation) < 1 or not 0 < args.learning_rate < float('inf') or args.top_k_loss < 0:
        parser.error('Steps, batch size, accumulation, and learning rate must be positive; loss top-k must be nonnegative.')
    return args


def build_config(args, local_rank):
    from trl.experimental.distillation import DistillationConfig
    extra = {'max_prompt_length':2048} if args.task == 'code' else {}
    return DistillationConfig(
        output_dir=args.output_dir, run_name=Path(args.output_dir).name,
        seed=args.seed, learning_rate=args.learning_rate, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        # beta=0 is KL(teacher || student); beta=1 is KL(student || teacher).
        beta=0. if args.divergence == 'fkl' else 1., lmbda=1., num_generations=1,
        loss_top_k=args.top_k_loss, loss_add_tail=True,
        max_completion_length=8192 if args.task == 'math' else 4096,
        max_length=9500 if args.task == 'math' else 6144,
        temperature=1., top_p=1., top_k=-1,
        use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=.2,
        vllm_max_model_length=9500 if args.task == 'math' else 6144,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        disable_dropout=True, dataloader_drop_last=True,
        deepspeed={'bf16':{'enabled':'auto'}, 'zero_optimization':{'stage':2},
                   'train_micro_batch_size_per_gpu':'auto', 'gradient_accumulation_steps':'auto', 'train_batch_size':'auto'},
        lr_scheduler_type='cosine', warmup_ratio=.05, weight_decay=0., max_grad_norm=.1,
        eval_strategy='no', save_steps=25, save_only_model=True, logging_steps=10, report_to=args.report_to,
        model_init_kwargs={'dtype':'bfloat16', 'use_cache':False, 'attn_implementation':'flash_attention_2'},
        teacher_model_init_kwargs={'dtype':'bfloat16', 'attn_implementation':'flash_attention_2', 'device_map':{'':local_rank}},
        **extra)


def main():
    args = parse_args()
    import os
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl.experimental.distillation import DistillationTrainer

    if not torch.cuda.is_available():
        raise RuntimeError('OPD training requires CUDA GPUs with BF16 support.')
    rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_template = tokenizer.apply_chat_template
    def nonthinking(*values, **kwargs):
        kwargs.setdefault('enable_thinking', False)
        return original_template(*values, **kwargs)
    tokenizer.apply_chat_template = nonthinking
    dataset, _ = load_training_data(args, tokenizer)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0., bias='none', task_type='CAUSAL_LM',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj'])
    trainer = DistillationTrainer(
        model=args.model, teacher_model='Qwen/Qwen3-4B-Instruct-2507' if args.task == 'math' else 'Qwen/Qwen3-8B',
        args=build_config(args, rank), processing_class=tokenizer, peft_config=lora, train_dataset=dataset)
    # The custom mean loss needs Trainer's gradient-accumulation scaling.
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
