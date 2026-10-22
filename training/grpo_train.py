# Standard GRPO with task data and rewards; no teacher or setting selector.
from pathlib import Path
from training.arguments import ZERO2, TaskArguments, model_kwargs, parse_arguments


def parse_args(argv=None):
    from trl import GRPOConfig

    return parse_arguments(
        TaskArguments,
        GRPOConfig,
        argv,
        output_dir="outputs/grpo",
        deepspeed=ZERO2,
        learning_rate=1e-5,
        max_steps=400,
        max_completion_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="none",
        mask_truncated_completions=True,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        vllm_max_model_length=20000,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        bf16=True,
        gradient_checkpointing=True,
        lora_dropout=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_steps=25,
        save_only_model=True,
        eval_strategy="no",
    )


def prepare_dataset(dataset, tokenizer, task, answer_lookup=None, thinking=False):
    def prepare(row):
        if "prompt" in row and "ground_truth" in row:
            return {"prompt": row["prompt"], "ground_truth": row["ground_truth"]}
        if answer_lookup is not None:
            messages = row["messages"]
        elif task == "math":
            messages = [
                {
                    "role": "user",
                    "content": f"Problem: {row['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                }
            ]
        else:
            messages = row.get("problem", row.get("messages"))
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        answer = (
            answer_lookup[prompt]
            if answer_lookup is not None
            else row.get("Answer", row.get("answer"))
        )
        return {"prompt": prompt, "ground_truth": answer}

    return dataset.map(prepare, remove_columns=dataset.column_names)


def load_task(script, config, tokenizer):
    # Original external math/TACO data is the default; prepared DeepCoder data also works.
    code_root = Path(script.data_dir or "data/external_code")
    if script.task in ("math", "code") and not (code_root / "train_data").is_dir():
        from training.external_teacher.tasks import load_training_data, get_reward

        dataset, lookup = load_training_data(script, tokenizer, seed=config.seed)
    else:
        from training.self_distillation.tasks import load_training_data, get_reward

        dataset, lookup = load_training_data(script.task, script.data_dir), None
    return prepare_dataset(
        dataset, tokenizer, script.task, lookup, script.student_thinking
    ), get_reward(script.task)


def main():
    script, config, model = parse_args()
    from transformers import AutoTokenizer
    from trl import GRPOTrainer, get_peft_config

    config.model_init_kwargs = {
        **model_kwargs(model, config.gradient_checkpointing),
        **(config.model_init_kwargs or {}),
    }
    config.chat_template_kwargs = {"enable_thinking": script.student_thinking}
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_name_or_path,
        revision=model.model_revision,
        trust_remote_code=model.trust_remote_code,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    original_template = tokenizer.apply_chat_template

    def chat_template(*values, **kwargs):
        kwargs["enable_thinking"] = script.student_thinking
        return original_template(*values, **kwargs)

    tokenizer.apply_chat_template = chat_template
    dataset, reward = load_task(script, config, tokenizer)

    def correct_reward(prompts, completions, ground_truth, **kwargs):
        return reward(prompts, completions, ground_truth)

    trainer = GRPOTrainer(
        model=model.model_name_or_path,
        args=config,
        processing_class=tokenizer,
        peft_config=get_peft_config(model),
        train_dataset=dataset,
        reward_funcs=[correct_reward],
    )
    trainer.train(resume_from_checkpoint=script.resume_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
