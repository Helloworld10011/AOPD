# Original GRPO coding baseline with SandboxFusion reward.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse, random
import pandas as pd
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
import re
from transformers import TrainerCallback
import gc
import json
from datasets import load_dataset, concatenate_datasets, load_from_disk
import logging, threading
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from external_teacher.sandbox_fusion import compute_score as code_compute_score

logging.getLogger("sandbox_fusion").setLevel(logging.WARNING)
logging.getLogger("sandbox_fusion.utils").setLevel(logging.WARNING)

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
val_size = 256
training_size = 5600

SYSTEM = """Write Python code to solve the problem. Present the code in a Python code block at the end. Think first, then write the Python code in a ```python code block."""


DATA_DIR = BASE_DIR / "taco_stdin_clean"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"{DATA_DIR} does not exist. Run prep_taco.sh first."
    )

raw = load_from_disk(str(DATA_DIR)).shuffle(seed=seed)

if len(raw) <= val_size:
    raise ValueError(f"Dataset only contains {len(raw)} examples.")

val_raw = raw.select(range(val_size))
train_raw = raw.select(range(val_size, min(val_size + training_size, len(raw))))


def to_chat(ex):
    problem = ex["question"]

    if ex.get("starter_code"):
        problem += f"\n\nStarter code:\n```python\n{ex['starter_code']}\n```"

    return {
        "prompt": [{"role": "user", "content": problem + "\n\n" + SYSTEM}],
        "ground_truth": ex["ground_truth"],
    }

train_ds = train_raw.map(to_chat, remove_columns=train_raw.column_names)
val_ds = val_raw.map(to_chat, remove_columns=val_raw.column_names)


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

N_REWARD_TESTS = 5

def select_reward_tests(tests):
    tests = json.loads(tests) if isinstance(tests, str) else tests
    return json.dumps({
        "inputs": tests["inputs"][:N_REWARD_TESTS],
        "outputs": tests["outputs"][:N_REWARD_TESTS],
    })

SANDBOX_URL = "http://127.0.0.1:8080/run_code"

REWARD_WORKERS = 4
_SANDBOX_SEM = threading.Semaphore(REWARD_WORKERS * N_REWARD_TESTS)
_REWARD_POOL = ThreadPoolExecutor(max_workers=REWARD_WORKERS)

def _score_one(completion, tests):
    text = get_completion_text(completion)

    if "```" not in text:
        return 0.0

    try:
        score, _ = code_compute_score(
            sandbox_fusion_url=SANDBOX_URL,
            concurrent_semaphore=_SANDBOX_SEM,
            memory_limit_mb=1024,
            completion=text,
            test_cases=select_reward_tests(tests),
            continuous=False,
            timeout=10,
        )
        return float(score)
    except Exception as e:
        print(f"[code_reward] {type(e).__name__}: {e}")
        return 0.0

def correct_reward(prompts, completions, ground_truth, **kwargs):
    return list(_REWARD_POOL.map(_score_one, completions, ground_truth))


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
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    num_generations=8,
    learning_rate= 2e-5,
    beta= 0.0,
    max_completion_length=4096,
    temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
    use_vllm=True, vllm_mode="colocate",
    loss_type="grpo",
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.35,
    logging_strategy="steps", logging_steps=10, logging_first_step=True,
    eval_strategy="steps", eval_steps=25, per_device_eval_batch_size=4, eval_on_start=True,
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
    model_init_kwargs={
        "dtype": "bfloat16",
        "use_cache": False,
        "attn_implementation": "flash_attention_2",
    },
    chat_template_kwargs={"enable_thinking": False}
)

trainer = GRPOTrainer(
    model=model_id,
    args=args,
    reward_funcs=[correct_reward],
    reward_processing_classes=None,
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[SafeSaveCallback()],
)

trainer.train()
# # trainer.train(resume_from_checkpoint=True)
