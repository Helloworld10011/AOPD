# Plain OPSD with the original teacher, thinking, and loss controls.
from training.arguments import (
    ZERO2,
    SelfDistillationArguments,
    model_kwargs,
    parse_arguments,
    validate_self_options,
)


def parse_args(argv=None):
    from trl.experimental.gold import GOLDConfig

    return parse_arguments(
        SelfDistillationArguments,
        GOLDConfig,
        argv,
        output_dir="outputs/opsd",
        learning_rate=5e-6,
        max_steps=400,
        deepspeed=ZERO2,
        max_length=20000,
        max_completion_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        temperature=1.1,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        lmbda=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        bf16=True,
        gradient_checkpointing=True,
        lora_dropout=0.05,
        save_steps=25,
        save_only_model=True,
        eval_strategy="no",
        steps_per_generation=1,
        max_grad_norm=0.1,
        task_defaults={
            "math": {
                "max_length": 35000,
                "max_completion_length": 8192,
                "lr_scheduler_type": "linear",
                "warmup_ratio": 0.0,
                "vllm_gpu_memory_utilization": 0.3,
            },
            "medical": {
                "max_steps": 300,
                "max_completion_length": 2048,
                "lr_scheduler_type": "linear",
                "warmup_ratio": 0.0,
            },
            "biology": {
                "learning_rate": 2e-5,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": -1,
                "lr_scheduler_type": "constant",
                "warmup_ratio": 0.0,
            },
            "code": {
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": -1,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
            },
        },
    )


def main():
    script, config, model = parse_args()
    from transformers import AutoTokenizer
    from trl import get_peft_config
    from .opsd_base import OPSDTrainer
    from .tasks import AOPDDataCollator, load_training_data

    validate_self_options(script, model)
    config.model_init_kwargs = {
        **model_kwargs(model, config.gradient_checkpointing),
        **(config.model_init_kwargs or {}),
    }
    config.dataset_kwargs = {
        **(config.dataset_kwargs or {}),
        "skip_prepare_dataset": True,
    }
    config.remove_unused_columns = False
    config.student_model_revision = model.model_revision
    config.presence_penalty = script.presence_penalty
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_name_or_path,
        revision=model.model_revision,
        trust_remote_code=model.trust_remote_code,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    prompts = AOPDDataCollator(
        tokenizer,
        script.task,
        config.max_length,
        config.max_completion_length,
        student_thinking=script.student_thinking,
        teacher_thinking=script.teacher_thinking,
        reason_first=script.reason_first,
    )
    trainer = OPSDTrainer(
        model=model.model_name_or_path,
        args=config,
        train_dataset=load_training_data(script.task, script.data_dir),
        processing_class=tokenizer,
        peft_config=get_peft_config(model),
        data_collator=prompts.collate_microbatch,
        fixed_teacher=script.fixed_teacher,
        reason_first=script.reason_first,
        use_ema_teacher=script.use_ema_teacher,
        ema_decay=script.ema_decay,
        student_thinking=script.student_thinking,
        teacher_thinking=script.teacher_thinking,
        use_thinking_machines_loss=script.use_tinker_loss,
        top_k_loss=script.top_k_loss,
        jsd_token_clip=script.jsd_token_clip or None,
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=script.resume_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
