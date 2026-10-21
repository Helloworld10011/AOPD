# Train external-teacher AOPD on verified TACO stdin problems.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
# train_atd_code_qwen.py
import torch, torch.distributed as dist
import os, argparse, random
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from trl.experimental.distillation import DistillationConfig
from external_teacher.aopd_trainer import ATDDistillationTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
import re
from transformers import TrainerCallback
import gc
import os, torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
import wandb
from vllm import SamplingParams

# CODE TASK: sandbox reward instead of math_grader (same imports as the GRPO script)
import json, logging, threading, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from external_teacher.sandbox_fusion import compute_score as code_compute_score
logging.getLogger("sandbox_fusion").setLevel(logging.WARNING)
logging.getLogger("sandbox_fusion.utils").setLevel(logging.WARNING)

os.environ["NCCL_TIMEOUT"] = "1800"

p = argparse.ArgumentParser()
p.add_argument("--learning_rate", type=float, default=2e-5)
p.add_argument("--max_steps", type=int, default=350)
p.add_argument("--per_device_train_batch_size", type=int, default=1)
p.add_argument("--gradient_accumulation_steps", type=int, default=4)
p.add_argument("--mu", type=float, default=0.0, help="Forward KL only when every rollout in a prompt group fails.")
p.add_argument("--exp", type=str, default="test")
p.add_argument("--model_path", type=str, default=None)
p.add_argument("--output_dir_base", type=str, default="outputs")
p.add_argument("--temperature", type=float, default=1.0)
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
MAX_PROMPT = 2048      # prompt budget; must match the GRPO script
MAX_COMPLETION = 4096         # same as the GRPO run

# CODE TASK: same prompt and data as the GRPO script
SYSTEM = """Write Python code to solve the problem. Present the code in a Python code block at the end. Think first, then write the Python code in a ```python code block."""

DATA_DIR = BASE_DIR / "taco_stdin_clean"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"{DATA_DIR} does not exist. Run prep_taco.sh first."
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)
_orig_apply = tokenizer.apply_chat_template
def _apply_no_think(*args, **kwargs):
    kwargs.setdefault("enable_thinking", False)   # only sets if caller didn't
    return _orig_apply(*args, **kwargs)
tokenizer.apply_chat_template = _apply_no_think

def to_chat(ex):
    problem = ex["question"]

    if ex.get("starter_code"):
        problem += f"\n\nStarter code:\n```python\n{ex['starter_code']}\n```"

    # CODE TASK: trainer expects "messages" (prompt-only chat); content identical to GRPO's "prompt"
    return {"messages": [{"role": "user", "content": problem + "\n\n" + SYSTEM}]}

def prompt_tokens(ex):
    text = tokenizer.apply_chat_template(to_chat(ex)["messages"], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    return len(tokenizer(text)["input_ids"])

raw = load_from_disk(str(DATA_DIR))
n0 = len(raw)
raw = raw.filter(lambda ex: prompt_tokens(ex) <= MAX_PROMPT)   # filter BEFORE shuffle -> same split in all scripts
print(f"prompt-length filter (<= {MAX_PROMPT} tokens): kept {len(raw)}/{n0}")
raw = raw.shuffle(seed=seed)

if len(raw) <= val_size:
    raise ValueError(f"Dataset only contains {len(raw)} examples.")

val_raw = raw.select(range(val_size))
train_raw = raw.select(range(val_size, min(val_size + training_size, len(raw))))

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

# CODE TASK: reward copied verbatim from the GRPO script
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

class VLLMRewardEvalCallback(TrainerCallback):
    # CODE TASK: prompts built through the same to_chat as training; scored against ground_truth
    def __init__(self, tokenizer, examples, max_new_tokens=4096):
        self.answers = list(examples["ground_truth"])
        self.sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.00, max_tokens=max_new_tokens)
        self.prompts = [
            tokenizer.apply_chat_template(
                to_chat(ex)["messages"],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for ex in examples
        ]
        self.trainer = None  # set after trainer construction

    def on_evaluate(self, args, state, control, **kwargs):
        llm = getattr(getattr(self.trainer, "vllm_generation", None),
                          "llm", None)
        if llm is None:
            print("WARNING: LLM not found in trainer. Skipping evaluation.")
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1

        # each rank scores an interleaved shard on its own engine
        my_prompts = self.prompts[rank::world]
        my_answers = self.answers[rank::world]
        outs = llm.generate(my_prompts, self.sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        local = correct_reward(None, texts, my_answers)

        stats = torch.tensor([float(sum(local)), float(len(local))], device="cuda")
        if dist.is_initialized():
            dist.all_reduce(stats)   # balanced collective, all ranks arrive ~together

        if state.is_world_process_zero:
            accuracy = (stats[0] / stats[1]).item()
            self.trainer.log({"eval_correct_reward": accuracy})

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
    max_prompt_length=MAX_PROMPT, max_completion_length=MAX_COMPLETION, max_length = MAX_COMPLETION + MAX_PROMPT,
    vllm_max_model_length=MAX_PROMPT + MAX_COMPLETION,
    temperature=args_cli.temperature, top_p=1.0, top_k=-1,
    use_vllm=True, vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.2,
    logging_strategy="steps", logging_steps=10, logging_first_step=True,
    eval_strategy="steps", eval_steps=25, per_device_eval_batch_size=4,
    save_strategy="steps", save_steps=25, save_only_model= True,
    report_to="wandb", run_name=exp_name,
    dataloader_drop_last=True,
    max_grad_norm=0.1, weight_decay=0.0,
    lr_scheduler_type="cosine", warmup_ratio=0.05,
    num_train_epochs=2, max_steps=args_cli.max_steps,
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

# CODE TASK: rendered prompt -> test cases; keys built through the same to_chat as training/eval
answer_lookup = {
    tokenizer.apply_chat_template(
        to_chat(ex)["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ): ex["ground_truth"]
    for ex in raw
}

trainer = ATDDistillationTrainer(
    model=model_id,
    teacher_model="Qwen/Qwen3-8B",
    args=args,
    processing_class = tokenizer,
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[SafeSaveCallback()],
    reward_func=correct_reward,
    answer_lookup=answer_lookup,
    mu=args_cli.mu,
)

rewardcb= VLLMRewardEvalCallback(tokenizer, val_raw)
rewardcb.trainer = trainer
trainer.add_callback(rewardcb)

trainer.train()
