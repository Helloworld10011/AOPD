# Train self-distillation AOPD on OpenThoughts using the all-wrong math variant.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import os
import wandb
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from vllm import SamplingParams
from evaluation.math_grader import boxed_reward_fn, timeout

from datasets import load_dataset
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
from self_distillation.buffered_trainer import BufferedATDSelfDistillationDataCollator
from self_distillation.all_wrong_trainer import AllWrongKLATDOPSDTrainer
from dataclasses import dataclass, field


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
    def __init__(self, tokenizer, problems, answers, aime_problems, aime_answers, eval_steps=25, max_new_tokens=8192, aime_max_new_tokens=16192):
        self.answers = list(answers)
        self.aime_answers = list(aime_answers)
        self.eval_steps = eval_steps
        self.max_new_tokens = max_new_tokens
        self.aime_max_new_tokens = aime_max_new_tokens
        self.trainer = None

        def render(items):
            return [tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Problem: {p}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for p in items]

        self.prompts = render(problems)
        self.aime_prompts = render(aime_problems)
        if not self.prompts or not self.aime_prompts or len(self.prompts) != len(self.answers) or len(self.aime_prompts) != len(self.aime_answers):
            raise ValueError("Evaluation sets must be nonempty with matching prompt and answer counts.")

    def _evaluate(self, llm, prompts, answers, rank, world, n, max_tokens):
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=max_tokens, n=n)
        my_prompts, my_answers = prompts[rank::world], answers[rank::world]
        outs = llm.generate(my_prompts, sp, use_tqdm=False) if my_prompts else []
        texts = [completion.text for output in outs for completion in output.outputs]
        repeated_answers = [answer for answer in my_answers for _ in range(n)]
        rewards = correct_reward(None, texts, repeated_answers)
        stats = torch.tensor([float(sum(rewards)), float(len(rewards))], device=self.trainer.accelerator.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (stats[0] / stats[1]).item()

    def _run_eval(self, state):
        trainer = self.trainer
        llm = trainer.vllm_engine
        if getattr(trainer, "_last_vllm_sync_step", -1) != state.global_step:
            trainer._move_model_to_vllm()
            trainer._last_vllm_sync_step = state.global_step

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        accuracy = self._evaluate(llm, self.prompts, self.answers, rank, world, n=1, max_tokens=self.max_new_tokens)
        aime_accuracy = self._evaluate(llm, self.aime_prompts, self.aime_answers, rank, world, n=4, max_tokens=self.aime_max_new_tokens)
        if state.is_world_process_zero:
            metrics = {"train/global_step": state.global_step, "eval_correct_reward": accuracy, "eval_aime_correct_reward": aime_accuracy}
            print(metrics)
            if wandb.run is not None:
                wandb.log(metrics)

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_eval(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._run_eval(state)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

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
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )
    atd_num_generations: int = field(default=4)
    atd_w_clip: float = field(default=4.0)
    atd_mu: float = field(default=0.0, metadata={"help": "Forward-KL coefficient on all-wrong groups only; use 0 for plain ATD."})


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

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

    full_wandb_run_config += f"_allwrong_mu{script_args.atd_mu:g}"

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
    if os.environ.get("RANK", "0") == "0":
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
                "prompts_per_update": effective_batch_size // script_args.atd_num_generations,
                "atd_num_generations": script_args.atd_num_generations,
                "atd_mu": script_args.atd_mu,
                "max_steps": training_args.max_steps,
                "lr_scheduler_type": str(training_args.lr_scheduler_type),
                "lora_dropout": model_args.lora_dropout,
                "disable_dropout": training_args.disable_dropout,
                "num_train_epochs": training_args.num_train_epochs,
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
        wandb.define_metric("eval_aime_correct_reward", step_metric="train/global_step")

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

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"]

    split = dataset.train_test_split(test_size=512, seed=42)
    train_dataset = split["train"]
    train_dataset = train_dataset.add_column("atd_group_id", list(range(len(train_dataset))))

    val_dataset = split["test"]
    val_problems = val_dataset["problem"]
    val_answers = val_dataset["Answer"]


    data_collator = BufferedATDSelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        reason_first=script_args.reason_first,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    trainer = AllWrongKLATDOPSDTrainer(
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

    aime_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    eval_callback = VLLMRewardEvalCallback(
        tokenizer=tokenizer,
        problems=val_problems,
        answers=val_answers,
        aime_problems=aime_dataset["problem"],
        aime_answers=aime_dataset["answer"],
        eval_steps=25,
        max_new_tokens=8192,
        aime_max_new_tokens=16192,
    )
    eval_callback.trainer = trainer
    trainer.add_callback(eval_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
