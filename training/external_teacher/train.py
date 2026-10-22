# Train external-teacher AOPD on math or coding with the same trainer.
import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl.experimental.distillation import DistillationConfig
from .trainer import AOPDTrainer

MATH_INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'
CODE_INSTRUCTION = 'Write Python code to solve the problem. Present the code in a Python code block at the end. Think first, then write the Python code in a ```python code block.'


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['math','code'], required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--data_dir', default='data/external_code')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--mu', type=float, default=None)
    parser.add_argument('--report_to', default='wandb')
    args = parser.parse_args(argv)
    args.max_steps = args.max_steps or (450 if args.task == 'math' else 350)
    args.mu = (.08 if args.task == 'math' else 0.) if args.mu is None else args.mu
    args.output_dir = args.output_dir or f'outputs/aopd_external_{args.task}'
    return args


def render_example(example, task):
    if task == 'math':
        messages = [{'role':'system','content':MATH_INSTRUCTION}, {'role':'user','content':example['problem']}]
    else:
        problem = example['question']
        if example.get('starter_code'):
            problem += f"\n\nStarter code:\n```python\n{example['starter_code']}\n```"
        messages = [{'role':'user','content':problem + '\n\n' + CODE_INSTRUCTION}]
    return {'messages':messages}


def load_training_data(args, tokenizer):
    if args.task == 'math':
        raw = load_dataset('siyanzhao/Openthoughts_math_30k_opsd', split='train').shuffle(seed=args.seed)
        raw = raw.select(range(len(raw)//2))
        train = raw.select(range(256,len(raw)))
    else:
        raw = load_from_disk(args.data_dir)
        def fits(example):
            text = tokenizer.apply_chat_template(render_example(example,'code')['messages'], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            return len(tokenizer(text)['input_ids']) <= 2048
        raw = raw.filter(fits).shuffle(seed=args.seed)
        train = raw.select(range(256,min(5856,len(raw))))
    lookup = {tokenizer.apply_chat_template(render_example(row,args.task)['messages'], tokenize=False,
               add_generation_prompt=True, enable_thinking=False):row['Answer' if args.task=='math' else 'ground_truth'] for row in raw}
    return train.map(lambda row:render_example(row,args.task), remove_columns=train.column_names), lookup


def get_reward(task):
    if task == 'math':
        from eval.math_grader import boxed_reward_fn, timeout
        def reward(prompts, completions, answers):
            scores = []
            for text, answer in zip(completions, answers):
                try:
                    with timeout(3):
                        _, score = boxed_reward_fn(text, answer, fast=True)
                    scores.append(score)
                except Exception:
                    scores.append(0.)
            return scores
        return reward
    from .code_reward import compute_score
    pool = ThreadPoolExecutor(max_workers=4)
    semaphore = threading.Semaphore(20)
    def score(text, tests):
        if '```' not in text:
            return 0.
        tests = json.loads(tests) if isinstance(tests,str) else tests
        selected = json.dumps({key:tests[key][:5] for key in ('inputs','outputs')})
        try:
            result, _ = compute_score('http://127.0.0.1:8080/run_code', semaphore, 1024, text, selected, continuous=False, timeout=10)
            return float(result)
        except Exception:
            return 0.
    return lambda prompts,completions,answers:list(pool.map(score,completions,answers))


def build_config(args, local_rank):
    cap = 8192 if args.task == 'math' else 4096
    deepspeed = {'train_batch_size':'auto', 'train_micro_batch_size_per_gpu':'auto',
                 'gradient_accumulation_steps':'auto', 'gradient_clipping':'auto',
                 'zero_allow_untested_optimizer':True, 'bf16':{'enabled':'auto'},
                 'zero_optimization':{'stage':2, 'allgather_partitions':True, 'allgather_bucket_size':5e8,
                  'overlap_comm':False, 'reduce_scatter':True, 'reduce_bucket_size':5e8,
                  'contiguous_gradients':True, 'round_robin_gradients':True}}
    extra = {} if args.task == 'math' else {'max_prompt_length':2048, 'vllm_max_model_length':6144}
    return DistillationConfig(
        output_dir=args.output_dir, run_name=Path(args.output_dir).name, deepspeed=deepspeed,
        seed=args.seed, learning_rate=args.learning_rate, max_steps=args.max_steps,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        disable_dropout=True, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, num_generations=4,
        lmbda=1., beta=0., loss_top_k=64, loss_add_tail=True,
        max_completion_length=cap, max_length=9500 if args.task=='math' else 6144,
        temperature=.8, top_p=1., top_k=-1, use_vllm=True, vllm_mode='colocate',
        vllm_gpu_memory_utilization=.2, logging_steps=10, logging_first_step=True,
        eval_strategy='no', save_strategy='steps', save_steps=25, save_only_model=True,
        dataloader_drop_last=True, max_grad_norm=.1, weight_decay=0.,
        lr_scheduler_type='cosine', warmup_ratio=.05, report_to=args.report_to,
        model_init_kwargs={'dtype':'bfloat16','use_cache':False,'attn_implementation':'flash_attention_2'},
        teacher_model_init_kwargs={'dtype':'bfloat16','attn_implementation':'flash_attention_2','device_map':{'':local_rank}},
        **extra)


def main():
    args = parse_args()
    rank = int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(rank)
    random.seed(args.seed); torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_template = tokenizer.apply_chat_template
    def nonthinking(*values, **kwargs):
        kwargs.setdefault('enable_thinking',False)
        return original_template(*values,**kwargs)
    tokenizer.apply_chat_template = nonthinking
    dataset, lookup = load_training_data(args,tokenizer)
    lora = LoraConfig(r=64,lora_alpha=128,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj'])
    trainer = AOPDTrainer(model=args.model,
        teacher_model='Qwen/Qwen3-4B-Instruct-2507' if args.task=='math' else 'Qwen/Qwen3-8B',
        args=build_config(args,rank),processing_class=tokenizer,peft_config=lora,
        train_dataset=dataset,reward_func=get_reward(args.task),answer_lookup=lookup,mu=args.mu)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
