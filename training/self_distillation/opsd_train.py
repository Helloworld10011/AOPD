# Plain OPSD: match the frozen reference-conditioned base model without advantage weighting.
import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['math', 'medical', 'biology', 'code'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_completion_length', type=int, default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--top_k_loss', type=int, default=0)
    parser.add_argument('--report_to', default='none')
    args = parser.parse_args(argv)
    if args.learning_rate is None:
        args.learning_rate = 2e-5 if args.task == 'biology' else 5e-6
    if args.max_steps is None:
        args.max_steps = 300 if args.task == 'medical' else 400
    if args.max_completion_length is None:
        args.max_completion_length = {'math':8192, 'medical':2048, 'biology':4096, 'code':4096}[args.task]
    args.output_dir = args.output_dir or f'outputs/opsd_{args.task}_{args.model.split("/")[-1]}'
    if min(args.max_steps, args.max_completion_length, args.batch_size, args.accumulation) < 1 or not 0 < args.learning_rate < float('inf') or args.top_k_loss < 0:
        parser.error('Steps, token limit, batch size, accumulation, and learning rate must be positive; loss top-k must be nonnegative.')
    if args.max_completion_length >= (35000 if args.task == 'math' else 20000):
        parser.error('Completion length must leave room for the prompt.')
    return args


def build_config(args):
    from trl.experimental.gold import GOLDConfig
    return GOLDConfig(
        output_dir=args.output_dir, run_name=Path(args.output_dir).name, seed=args.seed,
        learning_rate=args.learning_rate, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        steps_per_generation=1, max_length=35000 if args.task == 'math' else 20000,
        max_completion_length=args.max_completion_length, beta=0., lmbda=1.,
        temperature=1.1 if args.task in ('math','medical') else .8,
        top_p=.95 if args.task in ('math','medical') else 1., top_k=20 if args.task in ('math','medical') else -1,
        use_vllm=True, vllm_mode='colocate', vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=.3 if args.task == 'math' else .4,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        deepspeed={'bf16':{'enabled':'auto'}, 'zero_optimization':{'stage':2},
                   'train_micro_batch_size_per_gpu':'auto', 'gradient_accumulation_steps':'auto', 'train_batch_size':'auto'},
        lr_scheduler_type={'math':'linear','medical':'linear','biology':'constant','code':'cosine'}[args.task],
        warmup_ratio=.1 if args.task == 'code' else 0., max_grad_norm=.1,
        remove_unused_columns=False, dataset_kwargs={'skip_prepare_dataset':True},
        eval_strategy='no', save_steps=25, save_only_model=True, logging_steps=2, report_to=args.report_to,
        model_init_kwargs={'revision':args.revision, 'dtype':'bfloat16', 'use_cache':False, 'attn_implementation':'flash_attention_2'})


def main():
    args = parse_args()
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from .opsd_base import OPSDTrainer
    from .tasks import AOPDDataCollator, load_training_data

    if not torch.cuda.is_available():
        raise RuntimeError('OPSD training requires CUDA GPUs with BF16 support.')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side='left')
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    config = build_config(args)
    config.student_model_revision = args.revision
    prompts = AOPDDataCollator(tokenizer, args.task, config.max_length, config.max_completion_length)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=.05, bias='none', task_type='CAUSAL_LM',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    trainer = OPSDTrainer(
        model=args.model, args=config, train_dataset=load_training_data(args.task, args.data_dir),
        processing_class=tokenizer, peft_config=lora,
        # Reuse the task prompts directly, without AOPD's buffered collator or reward computation.
        data_collator=prompts.collate_microbatch, fixed_teacher=True, reason_first=False,
        use_ema_teacher=False, student_thinking=False, teacher_thinking=False,
        top_k_loss=args.top_k_loss, jsd_token_clip=None)
    # The custom mean loss needs Trainer's gradient-accumulation scaling.
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == '__main__':
    main()
