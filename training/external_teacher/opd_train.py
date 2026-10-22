# Plain external-teacher OPD with forward or reverse KL.
import os
from training.arguments import (
    EXTERNAL_ZERO2,
    OPDArguments,
    model_kwargs,
    parse_arguments,
)
from .tasks import load_training_data


def parse_args(argv=None):
    from trl.experimental.distillation import DistillationConfig

    script, config, model = parse_arguments(
        OPDArguments,
        DistillationConfig,
        argv,
        output_dir="outputs/opd",
        deepspeed=EXTERNAL_ZERO2,
        learning_rate=5e-6,
        max_steps=400,
        max_completion_length=4096,
        max_length=20000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        lmbda=1.0,
        loss_top_k=0,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
        bf16=True,
        gradient_checkpointing=True,
        lora_dropout=0.0,
        save_steps=25,
        save_only_model=True,
        eval_strategy="no",
        num_generations=1,
        warmup_ratio=0.05,
        max_grad_norm=0.1,
        weight_decay=0.0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        task_defaults={
            "math": {
                "max_steps": 450,
                "max_completion_length": 8192,
                "max_length": 9500,
            },
            "code": {
                "max_steps": 350,
                "max_completion_length": 4096,
                "max_length": 6144,
                "max_prompt_length": 2048,
                "vllm_max_model_length": 6144,
                "teacher_model": "Qwen/Qwen3-8B",
            },
        },
    )
    if script.divergence not in ("fkl", "rkl"):
        raise ValueError("Choose --divergence fkl or rkl.")
    config.beta = float(script.divergence == "rkl")
    return script, config, model


def main():
    script, config, model = parse_args()
    import torch
    from transformers import AutoTokenizer
    from trl import get_peft_config
    from trl.experimental.distillation import DistillationTrainer

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    config.model_init_kwargs = {
        **model_kwargs(model, config.gradient_checkpointing),
        **(config.model_init_kwargs or {}),
    }
    config.teacher_model_init_kwargs = {
        "dtype": "bfloat16",
        "attn_implementation": model.attn_implementation or "flash_attention_2",
        "device_map": {"": rank},
        "revision": script.teacher_revision,
        **(config.teacher_model_init_kwargs or {}),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_name_or_path,
        revision=model.model_revision,
        trust_remote_code=model.trust_remote_code,
    )
    original_template = tokenizer.apply_chat_template

    def chat_template(*values, **kwargs):
        kwargs["enable_thinking"] = script.student_thinking
        return original_template(*values, **kwargs)

    tokenizer.apply_chat_template = chat_template
    dataset, _ = load_training_data(script, tokenizer, seed=config.seed)
    trainer = DistillationTrainer(
        model=model.model_name_or_path,
        teacher_model=script.teacher_model,
        args=config,
        processing_class=tokenizer,
        peft_config=get_peft_config(model),
        train_dataset=dataset,
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=script.resume_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
