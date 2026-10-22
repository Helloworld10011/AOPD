# Shared script options; TRL keeps the original model and training arguments available.
from dataclasses import dataclass
import sys

ZERO2 = {
    "bf16": {"enabled": "auto"},
    "gradient_clipping": "auto",
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
}

EXTERNAL_ZERO2 = {
    **ZERO2,
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
    },
}


@dataclass
class TaskArguments:
    task: str = "math"
    data_dir: str | None = None
    student_thinking: bool = False
    resume_checkpoint: str | None = None


@dataclass
class ExternalArguments(TaskArguments):
    teacher_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    teacher_revision: str | None = None


@dataclass
class ExternalAOPDArguments(ExternalArguments):
    atd_w_clip: float = 4.0
    atd_mu: float = 0.0


@dataclass
class OPDArguments(ExternalArguments):
    divergence: str = "fkl"


@dataclass
class SelfDistillationArguments(TaskArguments):
    fixed_teacher: bool = True
    teacher_thinking: bool = False
    reason_first: bool = False
    use_ema_teacher: bool = False
    ema_decay: float = 0.999
    use_tinker_loss: bool = False
    top_k_loss: int = 0
    jsd_token_clip: float = 0.0
    presence_penalty: float = 0.0
    atd_num_generations: int = 4
    atd_w_clip: float = 4.0
    atd_mu: float = 0.0


def parse_arguments(
    script_class, config_class, argv=None, task_defaults=None, **defaults
):
    from trl import ModelConfig, TrlParser

    parser = TrlParser((script_class, config_class, ModelConfig))
    parser.set_defaults(
        report_to="wandb",
        use_peft=True,
        lora_r=64,
        lora_alpha=128,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        **defaults,
    )
    aliases = {
        "--model": "--model_name_or_path",
        "--revision": "--model_revision",
        "--batch_size": "--per_device_train_batch_size",
        "--accumulation": "--gradient_accumulation_steps",
        "--w_clip": "--atd_w_clip",
        "--mu": "--atd_mu",
    }
    if script_class is ExternalAOPDArguments:
        aliases["--top_k_loss"] = "--loss_top_k"
    if script_class is SelfDistillationArguments:
        aliases["--num_generations"] = "--atd_num_generations"
    tokens = list(sys.argv[1:] if argv is None else argv)
    tokens = [aliases.get(token, token) for token in tokens]
    if task_defaults:
        known, _ = parser.parse_known_args(tokens)
        parser.set_defaults(**task_defaults.get(known.task, {}))
    script, config, model = parser.parse_args_and_config(args=tokens)
    if script.task not in ("math", "code", "medical", "biology"):
        parser.error("Choose math, code, medical, or biology.")
    if not model.model_name_or_path:
        parser.error("Supply --model_name_or_path (or --model).")
    if isinstance(script, ExternalArguments) and script.task not in ("math", "code"):
        parser.error("External-teacher training supports math and code.")
    return script, config, model


def model_kwargs(model, gradient_checkpointing):
    from trl import get_kbit_device_map, get_quantization_config

    kwargs = {
        "revision": model.model_revision,
        "trust_remote_code": model.trust_remote_code,
        "dtype": getattr(model, "dtype", None)
        or getattr(model, "torch_dtype", None)
        or "bfloat16",
        "attn_implementation": model.attn_implementation or "flash_attention_2",
        "use_cache": not gradient_checkpointing,
    }
    quantization = get_quantization_config(model)
    if quantization is not None:
        kwargs.update(
            quantization_config=quantization, device_map=get_kbit_device_map()
        )
    return kwargs


def validate_self_options(script, model, buffered=False):
    import math

    if script.fixed_teacher and not model.use_peft:
        raise ValueError(
            "A fixed self-teacher requires PEFT, as in the original trainer."
        )
    if script.fixed_teacher and script.use_ema_teacher:
        raise ValueError("Choose fixed_teacher or use_ema_teacher, not both.")
    if script.reason_first and script.task != "math":
        raise ValueError("The original reason_first collator is available for math.")
    if script.top_k_loss < 0 or script.jsd_token_clip < 0:
        raise ValueError("Loss top-k and token clipping must be nonnegative.")
    if (
        not math.isfinite(script.atd_w_clip)
        or script.atd_w_clip <= 0
        or not math.isfinite(script.atd_mu)
        or script.atd_mu < 0
    ):
        raise ValueError(
            "w_clip must be positive and mu nonnegative; both must be finite."
        )
    if buffered and (
        script.reason_first or script.use_tinker_loss or script.jsd_token_clip
    ):
        raise ValueError(
            "The original buffered AOPD path requires reason_first=False, use_tinker_loss=False, and jsd_token_clip=0."
        )
