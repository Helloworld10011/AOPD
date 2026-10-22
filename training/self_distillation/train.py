# Train self-distillation AOPD on any of the four main paper tasks.
import argparse
from pathlib import Path
import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl.experimental.gold import GOLDConfig
from .trainer import AOPDTrainer
from .tasks import AOPDDataCollator, get_reward, load_training_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['math', 'medical', 'biology', 'code'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_completion_length', type=int, default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--num_generations', type=int, default=4)
    parser.add_argument('--mu', type=float, default=0.)
    parser.add_argument('--report_to', default='wandb')
    args = parser.parse_args(argv)
    if args.task != 'math' and args.mu:
        parser.error('--mu is used only by the all-wrong math recipe.')
    args.learning_rate = args.learning_rate or (1e-5 if args.task == 'biology' and 'Qwen3' in args.model else 2e-5)
    args.max_steps = args.max_steps or (300 if args.task == 'medical' else 400)
    args.max_completion_length = args.max_completion_length or {'math':8192, 'medical':2048, 'biology':4096, 'code':4096}[args.task]
    args.output_dir = args.output_dir or f'outputs/aopd_{args.task}_{args.model.split("/")[-1]}'
    return args


def build_config(args):
    # ZeRO-2 is embedded here so no separate configuration file is needed.
    deepspeed = {'train_batch_size':'auto', 'train_micro_batch_size_per_gpu':'auto',
                 'gradient_accumulation_steps':'auto', 'gradient_clipping':'auto',
                 'bf16':{'enabled':'auto'}, 'zero_optimization':{'stage':2, 'overlap_comm':True, 'contiguous_gradients':True}}
    return GOLDConfig(
        output_dir=args.output_dir, run_name=Path(args.output_dir).name,
        learning_rate=args.learning_rate, max_steps=args.max_steps, seed=args.seed,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        steps_per_generation=args.accumulation, dataloader_drop_last=True,
        max_length=35000 if args.task == 'math' else 20000,
        max_completion_length=args.max_completion_length,
        lr_scheduler_type={'math':'linear','medical':'linear','biology':'constant','code':'cosine'}[args.task],
        warmup_ratio=0. if args.task == 'biology' else (0. if args.task in ('math','medical') else .1),
        max_grad_norm=.1, gradient_checkpointing=True,
        beta=0., lmbda=1., temperature=.8, top_p=1., top_k=-1,
        use_vllm=True, vllm_mode='colocate', vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=.3 if args.task == 'math' else .4,
        deepspeed=deepspeed, bf16=True, eval_strategy='no',
        save_steps=25, logging_steps=2, save_only_model=args.task in ('math','medical'),
        report_to=args.report_to, dataset_kwargs={'skip_prepare_dataset':True},
        model_init_kwargs={'revision':args.revision, 'dtype':torch.bfloat16,
                           'attn_implementation':'flash_attention_2', 'use_cache':False})


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side='left')
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    config = build_config(args)
    config.student_model_revision = args.revision
    collator = AOPDDataCollator(tokenizer, args.task, config.max_length, config.max_completion_length)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=.05, bias='none', task_type='CAUSAL_LM',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    trainer = AOPDTrainer(
        model=args.model, args=config, train_dataset=load_training_data(args.task, args.data_dir),
        processing_class=tokenizer, data_collator=collator, peft_config=lora,
        fixed_teacher=True, reason_first=False, use_ema_teacher=False,
        student_thinking=False, teacher_thinking=False,
        top_k_loss=None if args.task == 'math' else 64, jsd_token_clip=None,
        reward_func=get_reward(args.task), atd_num_generations=args.num_generations,
        w_clip=4., mu=args.mu)
    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == '__main__':
    main()
