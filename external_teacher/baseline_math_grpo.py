# Original GRPO math baseline and boxed-answer reward.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
from pathlib import Path
# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse, random
import pandas as pd
from datasets import Dataset, load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
import re
from evaluation.math_grader import boxed_reward_fn, timeout
from transformers import TrainerCallback
import gc

os.environ["NCCL_TIMEOUT"] = "1800"

p = argparse.ArgumentParser()
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
val_size = 100

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
# ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
ds = ds.shuffle(seed=seed)
half_size = len(ds) // 2
ds = ds.select(range(half_size))

tokenizer = AutoTokenizer.from_pretrained(model_id)

def to_chat(ex):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["problem"]},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return {
        "prompt": prompt,
        "ground_truth": ex["Answer"],
    }

proc = ds.map(to_chat, remove_columns=ds.column_names)
val_ds = proc.select(range(val_size))
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

class BaseDriftCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.ref = {n: p.detach().clone() for n, p in model.named_parameters()
                    if "lora_" not in n and ("layers.0." in n or "layers.14." in n)}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 10 == 0:
            drift = max((p.detach() - self.ref[n]).abs().max().item()
                        for n, p in model.named_parameters() if n in self.ref)
            print(f"[base-drift] step {state.global_step}: {drift:.3e}")


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


def boxed_format_reward(prompts, completions, **kwargs):
    """Reward function that checks if the completion contains only one boxed answer."""
    pattern = r"\\boxed\{([^}]*)\}"
    rewards = []
    for completion in completions:
        try:
            content = get_completion_text(completion)
            matches = re.findall(pattern, content)
            rewards.append(1.0 if len(matches) == 1 else 0.0)
        except Exception as e:
            print(f"[boxed_format_reward] Error: {e}")
            rewards.append(0.0)
    return rewards


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


peft_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.00, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]
)

# 4) GRPO config (vLLM-backed)
args = GRPOConfig(
    deepspeed=str(Path(__file__).resolve().parent / "zero2.json"),
    output_dir= output_dir,
    seed = seed,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    disable_dropout=True,
    per_device_train_batch_size=4, gradient_accumulation_steps=16,
    # per_device_train_batch_size=16, gradient_accumulation_steps=4,
    num_generations=8,
    learning_rate= 5e-6,
    # beta= 1e-7,
    beta= 0.0,
    # max_prompt_length=None,
    max_completion_length=8192,
    temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
    use_vllm=True, vllm_mode="colocate",
    loss_type="grpo",
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.45,
    logging_strategy="steps", logging_steps=10, logging_first_step=True,
    eval_strategy="steps", eval_steps=50, per_device_eval_batch_size=4,
    save_strategy="epoch", save_only_model= True,
    report_to="wandb", run_name=exp_name,
    dataloader_drop_last=True,
    scale_rewards= False,
    mask_truncated_completions=True,
    max_grad_norm=1.0,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=2,
    load_best_model_at_end=False,
    reward_weights=[0.0, 1.0],
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "attn_implementation": "flash_attention_2",
    }
)

trainer = GRPOTrainer(
    model=model_id,
    args=args,
    reward_funcs=[boxed_format_reward, correct_reward],
    reward_processing_classes=None,
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[SafeSaveCallback(), BaseDriftCallback()],
)

trainer.train()
