# Buffered self-distillation AOPD. Experiment settings are in scripts/run_self_*.sh.
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
        output_dir="outputs/aopd_self",
        deepspeed=ZERO2,
    )


def main():
    script, config, model = parse_args()
    from transformers import AutoTokenizer
    from trl import get_peft_config
    from .aopd_trainer import AOPDTrainer
    from .tasks import AOPDDataCollator, get_reward, load_training_data

    validate_self_options(script, model, buffered=True)
    config.model_init_kwargs = {
        **model_kwargs(model, config.gradient_checkpointing),
        **(config.model_init_kwargs or {}),
    }
    config.dataset_kwargs = {
        **(config.dataset_kwargs or {}),
        "skip_prepare_dataset": True,
    }
    config.student_model_revision = model.model_revision
    config.presence_penalty = script.presence_penalty
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_name_or_path,
        revision=model.model_revision,
        trust_remote_code=model.trust_remote_code,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    collator = AOPDDataCollator(
        tokenizer,
        script.task,
        config.max_length,
        config.max_completion_length,
        student_thinking=script.student_thinking,
        teacher_thinking=script.teacher_thinking,
    )
    trainer = AOPDTrainer(
        model=model.model_name_or_path,
        args=config,
        train_dataset=load_training_data(script.task, script.data_dir),
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=get_peft_config(model),
        fixed_teacher=script.fixed_teacher,
        reason_first=script.reason_first,
        use_ema_teacher=script.use_ema_teacher,
        ema_decay=script.ema_decay,
        student_thinking=script.student_thinking,
        teacher_thinking=script.teacher_thinking,
        top_k_loss=script.top_k_loss,
        jsd_token_clip=None,
        reward_func=get_reward(script.task),
        atd_num_generations=script.atd_num_generations,
        w_clip=script.atd_w_clip,
        mu=script.atd_mu,
    )
    trainer.train(resume_from_checkpoint=script.resume_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
