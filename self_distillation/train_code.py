# Train AOPD on prepared TACO problems with the direct-code prompt and raw-Python reward.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import os
import wandb
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from vllm import SamplingParams

from datasets import load_from_disk
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from self_distillation.buffered_trainer import BufferedATDOPSDTrainer, BufferedATDSelfDistillationDataCollator
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional

VALID_LETTERS = {"A", "B", "C", "D"}

def _normalize_letter(snippet: str, case_sensitive: bool = False) -> Optional[str]:
    if not snippet:
        return None
    cleaned = re.sub(r"[\*\(\)\[\]\.\:`\"']", " ", snippet).strip()

    if case_sensitive:
        if cleaned in VALID_LETTERS:
            return cleaned
        m = re.match(r"\s*([ABCD])\b", cleaned)
        return m.group(1) if m else None
    else:
        if cleaned.upper() in VALID_LETTERS:
            return cleaned.upper()
        m = re.match(r"\s*([ABCD])\b", cleaned, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None


def extract_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D from a model response with multiple fallbacks."""
    if not text:
        return None

    # Strip chain-of-thought blocks for reasoning models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 1. Strict: last <answer>...</answer> block
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        cand = _normalize_letter(matches[-1], case_sensitive=False)
        if cand:
            return cand

    # 2. Unclosed <answer> tag (truncated generation)
    m = re.search(r"<answer>\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = _normalize_letter(m.group(1), case_sensitive=False)
        if cand:
            return cand

    # 3. Explicit cue words (case-insensitive — context disambiguates)
    cue_patterns = [
        r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:\-])\s*\(?\*?\*?([ABCDabcd])\b",
        r"\boption\s*\(?\*?\*?([ABCDabcd])\b",
        r"\b([ABCDabcd])\s*(?:is\s+(?:the\s+)?correct|is\s+right)",
        r"\\boxed\{\s*\(?([ABCDabcd])\)?\s*\}",
    ]
    for pat in cue_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 4. Leading "A." / "(A)" on a line — uppercase only
    m = re.search(r"^\s*\(?\*?\*?([ABCD])\)?\s*[\.\):]", text, flags=re.MULTILINE)
    if m:
        return m.group(1)

    # 5. Last-resort bare letter — uppercase only (avoids matching article "a")
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("*`()[].:\"' ")
        if stripped in VALID_LETTERS:
            return stripped

    return None

def get_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return ""


def correct_reward(prompts, completions, ground_truth, **kwargs):
    if len(completions) != len(ground_truth):
        raise ValueError("Completion and answer counts differ.")

    rewards = []
    for completion, answer in zip(completions, ground_truth):
        text = get_completion_text(completion)
        pred = extract_answer(text)
        rewards.append(float(pred == answer.strip().upper()))

    return rewards


class ScienceBufferedATDDataCollator(BufferedATDSelfDistillationDataCollator):
    def __init__(self, tokenizer, max_length, reason_first=False, student_thinking=False, teacher_thinking=False):
        super().__init__(tokenizer=tokenizer, max_length=max_length, reason_first=reason_first, student_thinking=student_thinking, teacher_thinking=teacher_thinking)
        self.science_max_length = max_length

    def collate_microbatch(self, features):
        student_prompts, teacher_prompts = [], []
        for feature in features:
            student = feature["problem"]
            if not student or student[-1]["role"] != "user" or any(m["role"] not in ("system", "user") for m in student):
                raise ValueError("The student prompt must contain only the question/instructions, not a reference answer.")
            reference = feature["solution"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("ATD requires a nonempty reference solution in each training record.")
            teacher = [dict(message) for message in student]
            teacher[-1]["content"] += f"\n\nReference solution:\n{reference}\n\nWrite Python code to solve the original problem in the requested format."
            student_prompts.append(self.tokenizer.apply_chat_template(student, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            teacher_prompts.append(self.tokenizer.apply_chat_template(teacher, tokenize=False, add_generation_prompt=True, enable_thinking=False))

        batch = {}
        for role, prompts in (("student", student_prompts), ("teacher", teacher_prompts)):
            encoded = self.tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
            if encoded.input_ids.size(1) > self.science_max_length:
                raise ValueError(f"{role} prompt exceeds max_length - max_completion_length; increase the context budget instead of cutting the reference.")
            batch[f"{role}_prompts"] = encoded.input_ids
            batch[f"{role}_prompt_attention_mask"] = encoded.attention_mask
            batch[f"{role}_prompt_length"] = encoded.input_ids.size(1)
            batch[f"{role}_prompt_lengths_per_example"] = encoded.attention_mask.sum(-1)

        batch["ground_truth"] = [feature["Answer"] for feature in features]
        batch["atd_group_ids"] = torch.tensor([feature["atd_group_id"] for feature in features], dtype=torch.long)
        return batch


class VLLMRewardEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, examples, eval_steps=25, max_new_tokens=2048):
        self.answers = list(examples["answer"])
        self.prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) for messages in examples["prompt"]]
        if not self.prompts or len(self.prompts) != len(self.answers):
            raise ValueError("Evaluation prompts and answers must be nonempty and aligned.")
        self.sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=max_new_tokens, n=1)
        self.eval_steps = eval_steps
        self.trainer = None

    def _run_eval(self, state):
        trainer = self.trainer
        llm = trainer.vllm_engine
        if getattr(trainer, "_last_vllm_sync_step", -1) != state.global_step:
            trainer._move_model_to_vllm()
            trainer._last_vllm_sync_step = state.global_step

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        prompts, answers = self.prompts[rank::world], self.answers[rank::world]
        outs = llm.generate(prompts, self.sp, use_tqdm=False) if prompts else []
        local = correct_reward(None, [o.outputs[0].text for o in outs], answers)
        stats = torch.tensor([float(sum(local)), float(len(local))], device=trainer.accelerator.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(stats)

        if state.is_world_process_zero:
            metrics = {"train/global_step": state.global_step, "eval_correct_reward": (stats[0] / stats[1]).item()}
            print(metrics)
            if wandb.run is not None:
                wandb.log(metrics)

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_eval(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._run_eval(state)


# Task-only adapters; original trainer and loss implementations are unchanged.
from self_distillation.code_reward import correct_reward
from evaluation.code_callback import VLLMTacoEvalCallback as VLLMRewardEvalCallback

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    science_data_dir: str = field(default="data/science")
    reward_eval_steps: int = field(default=25)
    eval_max_new_tokens: int = field(default=4096)

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default False for this non-thinking run."
        },
    )
    atd_num_generations: int = field(default=4)
    atd_w_clip: float = field(default=4.0)
    atd_mu: float = field(default=0.0)


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if not script_args.fixed_teacher or not model_args.use_peft or script_args.use_ema_teacher or script_args.reason_first:
        raise ValueError("Use fixed_teacher=True, use_peft=True, use_ema_teacher=False and reason_first=False for this science ATD experiment.")
    if training_args.eval_strategy != "no" or training_args.vllm_mode != "colocate" or training_args.vllm_tensor_parallel_size != 1 or training_args.vllm_enable_sleep_mode:
        raise ValueError("Keep eval_strategy=no, colocated vLLM, tensor parallel size 1, and sleep mode disabled for the existing callback.")
    if script_args.reward_eval_steps < 1 or script_args.eval_max_new_tokens < 1:
        raise ValueError("Evaluation interval and token budget must be positive.")
    if script_args.student_thinking or script_args.teacher_thinking:
        raise ValueError("This entry point requires student_thinking=False and teacher_thinking=False.")

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_config = f"atd_science_{model_name}_lr{lr_str}_bs{effective_batch_size}_tok{training_args.max_completion_length}"
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    print(f"\n{'='*80}")
    print("RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters.")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_steps": training_args.max_steps,
                "lr_scheduler_type": str(training_args.lr_scheduler_type),
                "warmup_ratio": training_args.warmup_ratio,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "jsd_token_clip": script_args.jsd_token_clip,
                "student_thinking": script_args.student_thinking,
                "teacher_thinking": script_args.teacher_thinking,
                "atd_num_generations": script_args.atd_num_generations,
                "atd_w_clip": script_args.atd_w_clip,
                "atd_mu": script_args.atd_mu,
            },
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("eval_correct_reward", step_metric="train/global_step")

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs
    training_args.dataset_kwargs = {**(training_args.dataset_kwargs or {}), "skip_prepare_dataset": True}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _orig_apply = tokenizer.apply_chat_template
    def _apply_no_think(*args, **kwargs):
        kwargs["enable_thinking"] = False
        return _orig_apply(*args, **kwargs)
    tokenizer.apply_chat_template = _apply_no_think

    training_args.presence_penalty = script_args.presence_penalty

    data_dir = Path(script_args.science_data_dir)
    train_dataset = load_from_disk(str(data_dir / "train_data"))
    val_dataset = load_from_disk(str(data_dir / "eval_data"))
    train_dataset = train_dataset.rename_columns({"messages": "problem", "output_text": "solution"})

    if "Answer" not in train_dataset.column_names:
        if "answer" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("answer", "Answer")
        else:
            def add_answer(example):
                solution = example["solution"]
                if "<answer>" not in solution or "</answer>" not in solution:
                    raise ValueError("Science ATD needs an answer for every training record; expected an answer column or <answer>...</answer> in output_text.")
                return {"Answer": solution.split("<answer>")[-1].split("</answer>")[0].strip()}
            train_dataset = train_dataset.map(add_answer)

    train_dataset = train_dataset.add_column("atd_group_id", list(range(len(train_dataset))))
    print(f"Science: {len(train_dataset)} training questions, {len(val_dataset)} evaluation questions")

    data_collator = ScienceBufferedATDDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length - training_args.max_completion_length,
        reason_first=script_args.reason_first,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    if script_args.atd_mu != 0:
        raise ValueError("These task recipes use mu=0. All-wrong KL is exposed by train_math.py only.")

    trainer = BufferedATDOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=False,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=False,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        reward_func=correct_reward,
        atd_num_generations=script_args.atd_num_generations,
        w_clip=script_args.atd_w_clip,
        mu=script_args.atd_mu,
    )

    eval_callback = VLLMRewardEvalCallback(tokenizer, val_dataset, eval_steps=script_args.reward_eval_steps, max_new_tokens=script_args.eval_max_new_tokens)
    eval_callback.trainer = trainer
    trainer.add_callback(eval_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
