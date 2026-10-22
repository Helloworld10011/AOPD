# External-teacher AOPD. Experiment settings are listed in scripts/run_external_*.sh.
import os
from training.arguments import (
    EXTERNAL_ZERO2,
    ExternalAOPDArguments,
    model_kwargs,
    parse_arguments,
)
from .tasks import get_reward, load_training_data


def parse_args(argv=None):
    from trl.experimental.distillation import DistillationConfig

    return parse_arguments(
        ExternalAOPDArguments,
        DistillationConfig,
        argv,
        output_dir="outputs/aopd_external",
        loss_top_k=0,
        deepspeed=EXTERNAL_ZERO2,
    )


def main():
    script, config, model = parse_args()
    import torch
    from transformers import AutoTokenizer
    from trl import get_peft_config
    from .aopd_trainer import AOPDTrainer

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
    dataset, lookup = load_training_data(script, tokenizer, seed=config.seed)
    trainer = AOPDTrainer(
        model=model.model_name_or_path,
        teacher_model=script.teacher_model,
        args=config,
        processing_class=tokenizer,
        peft_config=get_peft_config(model),
        train_dataset=dataset,
        reward_func=get_reward(script.task),
        answer_lookup=lookup,
        w_clip=script.atd_w_clip,
        mu=script.atd_mu,
    )
    trainer.train(resume_from_checkpoint=script.resume_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
