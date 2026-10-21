# Train AOPD on OpenThoughts with a separate frozen Qwen teacher.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
from pathlib import Path
# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse, random
import pandas as pd
from datasets import Dataset, load_dataset
from trl.experimental.distillation import DistillationConfig
from external_teacher.aopd_trainer import ATDDistillationTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
import re
from evaluation.math_grader import boxed_reward_fn, timeout
from transformers import TrainerCallback
import gc
import os, torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
import wandb
from vllm import SamplingParams

os.environ["NCCL_TIMEOUT"] = "1800"

p = argparse.ArgumentParser()
p.add_argument("--learning_rate", type=float, default=2e-5)
p.add_argument("--max_steps", type=int, default=450)
p.add_argument("--per_device_train_batch_size", type=int, default=1)
p.add_argument("--gradient_accumulation_steps", type=int, default=4)
p.add_argument("--mu", type=float, default=0.08, help="Forward KL only when every rollout in a prompt group fails.")
p.add_argument("--exp", type=str, default="test")
p.add_argument("--model_path", type=str, default=None)
p.add_argument("--output_dir_base", type=str, default="outputs")
args_cli = p.parse_args()
torch.cuda.set_device(local_rank)

output_dir = f"{args_cli.output_dir_base}/{args_cli.exp}"
exp_name = f"{args_cli.exp}"
model_id = args_cli.model_path
seed = 42

# 1) Build a simple "prompt" dataset from preference data
random.seed(seed)
torch.manual_seed(seed)
val_size = 256

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
ds = ds.shuffle(seed=seed)
half_size = len(ds) // 2
ds = ds.select(range(half_size))

tokenizer = AutoTokenizer.from_pretrained(model_id)
_orig_apply = tokenizer.apply_chat_template
def _apply_no_think(*args, **kwargs):
    kwargs.setdefault("enable_thinking", False)   # only sets if caller didn't
    return _orig_apply(*args, **kwargs)
tokenizer.apply_chat_template = _apply_no_think

def to_chat(ex):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["problem"]},
    ]}

proc = ds.map(to_chat, remove_columns=ds.column_names)
val_ds = proc.select(range(val_size))
val_raw = ds.select(range(val_size))
train_ds = proc.select(range(val_size, len(proc)))

class SafeSaveCallback(TrainerCallback):
    """Ensures clean state before checkpoint saving."""

    def on_save(self, args, state, control, **kwargs):
        """Called before saving checkpoint."""
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        # Synchronize all ranks before save
        if dist.is_initialized():
            dist.barrier()

        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at end of epoch."""
        # Sync before potential save
        gc.collect()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        return control

def get_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if (
        isinstance(completion, list)
        and completion
        and isinstance(completion[0], dict)
    ):
        return completion[0].get("content", "") or ""
    return ""

def correct_reward(prompts, completions, ground_truth, **kwargs):
    """Safe wrapper for DR-GRPO's boxed_reward_fn."""
    rewards = []

    for i, comp in enumerate(completions):
        try:
            # Extract content
            content = get_completion_text(comp)

            if not content or i >= len(ground_truth):
                rewards.append(0.0)
                continue

            # Wrap ENTIRE reward computation in timeout
            try:
                with timeout(3):  # 3 second timeout for entire computation
                    _, reward = boxed_reward_fn(
                        content,
                        ground_truth[i],
                        fast=True  # Use fast mode to skip slow is_latex_equal
                    )
                    rewards.append(reward)
            except TimeoutError:
                rewards.append(0.0)

        except Exception:
            rewards.append(0.0)

    return rewards


class VLLMRewardEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, problems, answers, aime_problems, aime_answers, max_new_tokens=8192):
        self.answers = list(answers)
        self.aime_answers = list(aime_answers)
        self.max_new_tokens = max_new_tokens

        self.prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for p in problems
        ]

        self.aime_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for p in aime_problems
        ]

        self.trainer = None

    def _evaluate(self, llm, prompts, answers, rank, world):
        if len(prompts) != len(answers):
            raise ValueError("Prompt and answer counts do not match.")

        self.sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.00, max_tokens=self.max_new_tokens, n = self.n)

        my_prompts = prompts[rank::world]
        my_answers = answers[rank::world]

        outs = llm.generate(my_prompts, self.sp, use_tqdm=False)

        texts = [x.text for o in outs for x in o.outputs]
        repeated_answers = [a for a in my_answers for _ in range(self.n)]
        rewards = correct_reward(None, texts, repeated_answers)

        local_correct = float(sum(rewards))
        local_samples = float(len(rewards))

        stats = torch.tensor([local_correct, local_samples], device="cuda")
        if dist.is_initialized():
            dist.all_reduce(stats)

        return (stats[0] / stats[1]).item()

    def on_evaluate(self, args, state, control, **kwargs):
        llm = getattr(getattr(self.trainer, "vllm_generation", None), "llm", None)
        if llm is None:
            print("WARNING: LLM not found in trainer. Skipping evaluation.")
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1

        self.n=1
        accuracy = self._evaluate(llm, self.prompts, self.answers, rank, world)
        self.n=4
        aime_accuracy = self._evaluate(llm, self.aime_prompts, self.aime_answers, rank, world)

        if state.is_world_process_zero:
            self.trainer.log({
                "eval_correct_reward": accuracy,
                "eval_aime_correct_reward": aime_accuracy,
            })


peft_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.00, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]
)

# 4) GRPO config (vLLM-backed)
args = DistillationConfig(
    deepspeed=str(Path(__file__).resolve().parent / "zero2.json"),
    output_dir= output_dir,
    seed = seed,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    disable_dropout=True,
    per_device_train_batch_size=args_cli.per_device_train_batch_size, gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
    num_generations=4,
    learning_rate=args_cli.learning_rate,
    lmbda=1.0, beta=0.0, loss_top_k=64, loss_add_tail=True,
    max_completion_length=8192, max_length = 9500,
    temperature=0.8, top_p=1.0, top_k=-1,
    use_vllm=True, vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.2,
    logging_strategy="steps", logging_steps=10, logging_first_step=True,
    eval_strategy="steps", eval_steps=25, per_device_eval_batch_size=4,
    save_strategy="epoch", save_only_model= True,
    report_to="wandb", run_name=exp_name,
    dataloader_drop_last=True,
    max_grad_norm=0.1, weight_decay=0.0,
    lr_scheduler_type="cosine", warmup_ratio=0.05,
    num_train_epochs=1, max_steps=args_cli.max_steps,
    load_best_model_at_end=False,
    eval_on_start=True,
    model_init_kwargs={
        "dtype": "bfloat16",
        "use_cache": False,
        "attn_implementation": "flash_attention_2",
    },
    teacher_model_init_kwargs={
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "device_map": {"": local_rank},
    },
)

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
if dist.is_initialized():
    dist.barrier()

answer_lookup = {
    tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": ex["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ): ex["Answer"]
    for ex in ds
}

trainer = ATDDistillationTrainer(
    model=model_id,
    teacher_model="Qwen/Qwen3-4B-Instruct-2507",
    args=args,
    processing_class = tokenizer,
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[SafeSaveCallback()],
    reward_func=correct_reward,
    answer_lookup=answer_lookup,
    mu=args_cli.mu
    # w_clip = 10.0,
)

aime_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
aime_problems = aime_dataset["problem"]
aime_answers = aime_dataset["answer"]

eval_callback = VLLMRewardEvalCallback(
    tokenizer,
    val_raw["problem"],
    val_raw["Answer"],
    aime_problems,
    aime_answers,
    max_new_tokens=16192,
)

eval_callback.trainer = trainer
trainer.add_callback(eval_callback)

trainer.train()
