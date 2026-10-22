# Standard TRL GRPO using the corresponding task's student prompts and terminal rewards.
import argparse
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=['external', 'self'], required=True)
    parser.add_argument('--task', choices=['math', 'medical', 'biology', 'code'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_completion_length', type=int, default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=None)
    parser.add_argument('--num_generations', type=int, default=None)
    parser.add_argument('--report_to', default='none')
    args = parser.parse_args(argv)
    if args.setting == 'external' and args.task not in ('math', 'code'):
        parser.error('External-teacher task data is available for math and code.')
    if args.learning_rate is None:
        args.learning_rate = {'math':5e-6, 'medical':1e-5, 'biology':2e-5, 'code':2e-5}[args.task]
    if args.max_steps is None:
        args.max_steps = (450 if args.task == 'math' else 350) if args.setting == 'external' else (300 if args.task == 'medical' else 400)
    if args.max_completion_length is None:
        args.max_completion_length = {'math':8192, 'medical':2048, 'biology':4096, 'code':4096}[args.task]
    if args.num_generations is None:
        args.num_generations = 8 if args.setting == 'external' else 4
    if args.accumulation is None:
        args.accumulation = 8 if args.setting == 'external' else 4
    args.data_dir = args.data_dir or ('data/external_code' if args.setting == 'external' else f'data/{args.task}')
    args.output_dir = args.output_dir or f'outputs/grpo_{args.setting}_{args.task}_{args.model.split("/")[-1]}'
    if min(args.max_steps, args.max_completion_length, args.batch_size, args.accumulation) < 1 or not 0 < args.learning_rate < float('inf'):
        parser.error('Steps, token limit, batch size, accumulation, and learning rate must be positive.')
    world = int(os.environ.get('WORLD_SIZE', 1))
    if args.num_generations < 2 or args.batch_size * args.accumulation * world % args.num_generations:
        parser.error('num_generations must be at least two and divide the global generation batch.')
    context = 35000 if args.task == 'math' else (6144 if args.task == 'code' else 20000)
    if args.max_completion_length >= context:
        parser.error('Completion length must leave room for the prompt.')
    return args


def build_config(args):
    from trl import GRPOConfig
    # The external and self-distillation environments pin different TRL versions.
    extra = {'max_prompt_length':None} if 'max_prompt_length' in GRPOConfig.__dataclass_fields__ else {}
    return GRPOConfig(
        output_dir=args.output_dir, run_name=Path(args.output_dir).name,
        seed=args.seed, learning_rate=args.learning_rate, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        num_generations=args.num_generations, max_completion_length=args.max_completion_length,
        beta=0., loss_type='grpo', scale_rewards='none', mask_truncated_completions=True,
        temperature=1., top_p=1., top_k=-1,
        use_vllm=True, vllm_mode='colocate', vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=.35 if args.task in ('biology','code') else .45,
        vllm_max_model_length=35000 if args.task == 'math' else (6144 if args.task == 'code' else 20000),
        chat_template_kwargs={'enable_thinking':False},
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        disable_dropout=True, remove_unused_columns=False, dataloader_drop_last=True,
        deepspeed={'bf16':{'enabled':'auto'}, 'zero_optimization':{'stage':2},
                   'train_micro_batch_size_per_gpu':'auto', 'gradient_accumulation_steps':'auto', 'train_batch_size':'auto'},
        lr_scheduler_type='cosine', warmup_ratio=0. if args.task == 'biology' else .05,
        max_grad_norm=1., weight_decay=.05 if args.task == 'biology' else 0.,
        eval_strategy='no', save_steps=25, save_only_model=True, logging_steps=10, report_to=args.report_to,
        model_init_kwargs={'dtype':'bfloat16', 'use_cache':False, 'attn_implementation':'flash_attention_2'},
        **extra)


def prepare_dataset(dataset, tokenizer, task, answer_lookup=None):
    def prepare(row):
        if answer_lookup is not None:
            messages = row['messages']
        elif task == 'math':
            messages = [{'role':'user', 'content':f"Problem: {row['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
        else:
            messages = row['problem']
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        answer = answer_lookup[prompt] if answer_lookup is not None else row['Answer']
        return {'prompt':prompt, 'ground_truth':answer}
    # Teacher references and solutions never enter a GRPO model prompt.
    return dataset.map(prepare, remove_columns=dataset.column_names)


def main():
    args = parse_args()
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError('GRPO training requires CUDA GPUs with BF16 support.')
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if args.setting == 'external':
        from training.external_teacher.tasks import load_training_data, get_reward
        dataset, lookup = load_training_data(args, tokenizer)
    else:
        from training.self_distillation.tasks import load_training_data, get_reward
        dataset, lookup = load_training_data(args.task, args.data_dir), None
    dataset = prepare_dataset(dataset, tokenizer, args.task, lookup)
    reward = get_reward(args.task)
    def correct_reward(prompts, completions, ground_truth, **kwargs):
        return reward(prompts, completions, ground_truth)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0., bias='none', task_type='CAUSAL_LM',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj'])
    trainer = GRPOTrainer(model=args.model, args=build_config(args), processing_class=tokenizer,
                          peft_config=lora, train_dataset=dataset, reward_funcs=[correct_reward])
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
