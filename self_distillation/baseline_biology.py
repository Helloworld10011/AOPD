# Original OPSD biology baseline and task-specific reference prompts.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import os
import wandb
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from vllm import SamplingParams

from datasets import load_dataset, load_from_disk
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
from self_distillation.opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional


VALID_LETTERS = {'A', 'B', 'C', 'D'}

def _normalize_letter(snippet: str, case_sensitive: bool=False) -> Optional[str]:
    if not snippet:
        return None
    cleaned = re.sub('[\\*\\(\\)\\[\\]\\.\\:`\\"\']', ' ', snippet).strip()
    if case_sensitive:
        if cleaned in VALID_LETTERS:
            return cleaned
        m = re.match('\\s*([ABCD])\\b', cleaned)
        return m.group(1) if m else None
    else:
        if cleaned.upper() in VALID_LETTERS:
            return cleaned.upper()
        m = re.match('\\s*([ABCD])\\b', cleaned, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None


def extract_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D from a model response with multiple fallbacks."""
    if not text:
        return None
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    matches = re.findall('<answer>\\s*(.*?)\\s*</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        cand = _normalize_letter(matches[-1], case_sensitive=False)
        if cand:
            return cand
    m = re.search('<answer>\\s*(.*)', text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = _normalize_letter(m.group(1), case_sensitive=False)
        if cand:
            return cand
    cue_patterns = ['(?:final\\s+answer|the\\s+answer\\s+is|answer\\s*[:\\-])\\s*\\(?\\*?\\*?([ABCDabcd])\\b', '\\boption\\s*\\(?\\*?\\*?([ABCDabcd])\\b', '\\b([ABCDabcd])\\s*(?:is\\s+(?:the\\s+)?correct|is\\s+right)', '\\\\boxed\\{\\s*\\(?([ABCDabcd])\\)?\\s*\\}']
    for pat in cue_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search('^\\s*\\(?\\*?\\*?([ABCD])\\)?\\s*[\\.\\):]', text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip('*`()[].:"\' ')
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


class ReferenceDataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        student_prompts, teacher_prompts = [], []
        for feature in features:
            student = feature["problem"]
            if not student or student[-1]["role"] != "user" or any(m["role"] not in ("system", "user") for m in student):
                raise ValueError("The student prompt must contain only the question/instructions, not a reference answer.")
            reference = feature["solution"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("OPSD requires a nonempty reference solution in each training record.")
            teacher = [dict(message) for message in student]
            teacher[-1]["content"] = f"\n{student[-1]['content']}\n\nThe verified correct answer to the question is:\n{reference}\n\nNow answer with a response of your own, including the thinking process.\n"
            student_prompts.append(self.tokenizer.apply_chat_template(student, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            teacher_prompts.append(self.tokenizer.apply_chat_template(teacher, tokenize=False, add_generation_prompt=True, enable_thinking=False))

        batch = {}
        for role, prompts in (("student", student_prompts), ("teacher", teacher_prompts)):
            encoded = self.tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
            if encoded.input_ids.size(1) > self.max_length:
                raise ValueError(f"{role} prompt exceeds max_length - max_completion_length; increase the context budget instead of cutting the reference.")
            batch[f"{role}_prompts"] = encoded.input_ids
            batch[f"{role}_prompt_attention_mask"] = encoded.attention_mask
            batch[f"{role}_prompt_length"] = encoded.input_ids.size(1)
            batch[f"{role}_prompt_lengths_per_example"] = encoded.attention_mask.sum(-1)
        return batch


class VLLMRewardEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, examples, eval_steps=25, max_new_tokens=1024):
        self.examples = list(examples)
        self.prompts = [tokenizer.apply_chat_template(x['prompt'], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in self.examples]
        self.sp = SamplingParams(temperature=0., top_p=1., top_k=-1, max_tokens=max_new_tokens, n=1, seed=7319)
        self.eval_steps = eval_steps
        self.trainer = None
    def _run_eval(self, state):
        import json
        trainer = self.trainer
        if getattr(trainer, '_last_vllm_sync_step', -1) != state.global_step:
            trainer._move_model_to_vllm()
            trainer._last_vllm_sync_step = state.global_step
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        examples = self.examples[rank::world]
        outs = trainer.vllm_engine.generate(self.prompts[rank::world], self.sp, use_tqdm=False)
        assert len(outs) == len(examples)
        rows = []
        for x, o in zip(examples, outs):
            v = o.outputs[0]
            rows.append(dict(uid=x['uid'], context_uid=x['context_uid'], gold=x['answer'], prediction=extract_answer(v.text), completion=v.text, correct=correct_reward(None,[v.text],[x['answer']])[0], tokens=len(v.token_ids), finish_reason=v.finish_reason))
        dest = Path(trainer.args.output_dir) / 'science_screen'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'generations_step_{state.global_step:06d}_rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        stats = torch.tensor([sum(x['correct'] for x in rows),len(rows),sum(x['finish_reason']=='length' for x in rows),sum(x['tokens'] for x in rows)],device=trainer.accelerator.device,dtype=torch.float64)
        if dist.is_initialized(): dist.all_reduce(stats)
        if state.is_world_process_zero:
            metrics = {'train/global_step':state.global_step,'eval_correct_reward':(stats[0]/stats[1]).item(),'eval_truncation_fraction':(stats[2]/stats[1]).item(),'eval_mean_tokens':(stats[3]/stats[1]).item()}
            (dest / f'metrics_step_{state.global_step:06d}.json').write_text(json.dumps(metrics)+'\n')
            print(metrics,flush=True)
            if wandb.run is not None: wandb.log(metrics)
    def on_train_begin(self,args,state,control,**kwargs): self._run_eval(state)
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step>0 and state.global_step % self.eval_steps==0: self._run_eval(state)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Existing OPSD arguments plus local data/evaluation paths."""

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


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if not script_args.fixed_teacher or not model_args.use_peft or script_args.use_ema_teacher or script_args.reason_first:
        raise ValueError("Use fixed_teacher=True, use_peft=True, use_ema_teacher=False and reason_first=False for this reference-OPSD experiment.")
    if training_args.eval_strategy != "no" or training_args.vllm_mode != "colocate" or training_args.vllm_tensor_parallel_size != 1 or training_args.vllm_enable_sleep_mode:
        raise ValueError("Keep eval_strategy=no, colocated vLLM, tensor parallel size 1, and sleep mode disabled for the existing callback.")
    if script_args.reward_eval_steps < 1 or script_args.eval_max_new_tokens < 1:
        raise ValueError("Evaluation interval and token budget must be positive.")
    if script_args.student_thinking or script_args.teacher_thinking:
        raise ValueError("This entry point requires student_thinking=False and teacher_thinking=False.")

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
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
                "jsd_token_clip": script_args.jsd_token_clip,
                "student_thinking": script_args.student_thinking,
                "teacher_thinking": script_args.teacher_thinking,
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
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            },
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("eval_correct_reward", step_metric="train/global_step")

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
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
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs
    training_args.dataset_kwargs = {**(training_args.dataset_kwargs or {}), "skip_prepare_dataset": True}

    # No separate teacher model needed - we use the same model with privileged info

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

    ################
    # Dataset
    ################
    # Load the existing local dataset and its reference responses
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    data_dir = Path(script_args.science_data_dir)
    train_dataset = load_from_disk(str(data_dir / "train_data"))
    val_dataset = load_from_disk(str(data_dir / "eval_data"))
    train_dataset = train_dataset.rename_columns({"messages": "problem", "output_text": "solution"})
    print(f"Science: {len(train_dataset)} training questions, {len(val_dataset)} evaluation questions")

    data_collator = ReferenceDataCollator(tokenizer, training_args.max_length - training_args.max_completion_length)

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    trainer.model_accepts_loss_kwargs = False

    eval_callback = VLLMRewardEvalCallback(tokenizer, val_dataset, eval_steps=script_args.reward_eval_steps, max_new_tokens=script_args.eval_max_new_tokens)
    eval_callback.trainer = trainer
    trainer.add_callback(eval_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
