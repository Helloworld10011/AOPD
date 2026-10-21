# Original OPSD medical baseline with reference-conditioned teacher.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.gold import GOLDConfig
from vllm import SamplingParams

from self_distillation.opsd_trainer import OPSDTrainer

TRAIN_DATASET = "ytz20/sys_medmcqa_train"
EVAL_DATASET = "ytz20/sys_medmcqa_test"


def get_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return ""


def extract_answer(completion):
    # Same extraction as LMOps/opcd/verl/verl/trainer/ppo/reward.py:get_medmcqa_acc(version="v1").
    text = get_completion_text(completion)
    matches = re.findall(r"<answer>([\s\S]*?)<\/answer>", text, re.IGNORECASE)
    if not matches:
        return None
    answer = re.search(r"\(([ABCDabcd]?)\)", matches[-1])
    if answer and answer.group(1):
        return answer.group(1).upper()
    answer = re.search(r"[ABCDabcd]", matches[-1])
    return answer.group(0).upper() if answer else None


def correct_reward(prompts, completions, ground_truth, **kwargs):
    if len(completions) != len(ground_truth):
        raise ValueError("Completion and answer counts differ.")
    return [float(extract_answer(completion) == str(answer).upper()) for completion, answer in zip(completions, ground_truth)]


def prepare_example(example, index, require_reference=True):
    messages = example["content"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"MedMCQA row {index}: content must be a list of messages.")
    if any(not isinstance(m, dict) or m.get("role") != "user" or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        raise ValueError(f"MedMCQA row {index}: expected prompt-only user messages.")
    answer = str(example["answer"]).strip().upper()
    if answer not in ("A", "B", "C", "D"):
        raise ValueError(f"MedMCQA row {index}: invalid answer key {answer!r}.")
    reference = example.get("teacher_response") or ""
    if require_reference and (not isinstance(reference, str) or not reference.strip()):
        raise ValueError(f"MedMCQA row {index}: teacher_response is missing.")
    return {"problem": messages, "solution": reference, "Answer": answer, "atd_group_id": index}


def load_medmcqa(args, seed):
    # Both separately released artifacts have a Hugging Face split named "train".
    train = load_dataset(TRAIN_DATASET, split="train", revision=args.data_revision)
    evaluation = load_dataset(EVAL_DATASET, split="train", revision=args.data_revision)
    if args.max_train_samples:
        if args.max_train_samples > len(train):
            raise ValueError("max_train_samples exceeds the released training set size.")
        train = train.shuffle(seed=seed).select(range(args.max_train_samples))
    if args.eval_size:
        if args.eval_size > len(evaluation):
            raise ValueError("eval_size exceeds the released evaluation set size.")
        evaluation = evaluation.select(range(args.eval_size))
    train = train.map(prepare_example, with_indices=True, remove_columns=train.column_names, desc="MedMCQA train")
    evaluation = evaluation.map(prepare_example, with_indices=True, fn_kwargs={"require_reference": False}, remove_columns=evaluation.column_names, desc="MedMCQA eval")
    if not len(train) or not len(evaluation):
        raise ValueError("Training and evaluation datasets must both be nonempty.")
    return train, evaluation


def render_prompt_pairs(features, tokenizer):
    students, teachers = [], []
    for feature in features:
        student = feature["problem"]
        reference = feature["solution"]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Each training record needs its released teacher_response.")
        teacher = [dict(message) for message in student]
        teacher[-1]["content"] = f"\n{student[-1]['content']}\n\nThis is an example for a response to the question:\n{reference}\n\nNow answer with a response of your own, including the thinking process.\n"
        students.append(tokenizer.apply_chat_template(student, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        teachers.append(tokenizer.apply_chat_template(teacher, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return students, teachers


def collate_prompt_pairs(features, tokenizer, prompt_budget):
    students, teachers = render_prompt_pairs(features, tokenizer)
    batch = {}
    for role, prompts in (("student", students), ("teacher", teachers)):
        encoded = tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
        if encoded.input_ids.size(1) > prompt_budget:
            raise ValueError(f"{role} prompt exceeds max_length - max_completion_length; increase max_length rather than truncate the question/reference.")
        batch[f"{role}_prompts"] = encoded.input_ids
        batch[f"{role}_prompt_attention_mask"] = encoded.attention_mask
        batch[f"{role}_prompt_length"] = encoded.input_ids.size(1)
        batch[f"{role}_prompt_lengths_per_example"] = encoded.attention_mask.sum(-1)
    return batch


class MedMCQADataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, features):
        return collate_prompt_pairs(features, self.tokenizer, self.max_length)


class VLLMRewardEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, args, max_length):
        self.args, self.trainer = args, None
        self.answers = list(eval_dataset["Answer"])
        self.prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) for messages in eval_dataset["problem"]]
        lengths = [len(ids) for ids in tokenizer(self.prompts, add_special_tokens=False, truncation=False)["input_ids"]]
        if max(lengths) + args.eval_max_new_tokens > max_length:
            raise ValueError("Evaluation prompt plus eval_max_new_tokens exceeds max_length.")
        self._last_eval_step = None

    def _run_eval(self, state):
        if self._last_eval_step == state.global_step:
            return
        trainer = self.trainer
        if getattr(trainer, "_last_vllm_sync_step", -1) != state.global_step:
            trainer._move_model_to_vllm()
            trainer._last_vllm_sync_step = state.global_step
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        indices = list(range(rank, len(self.prompts), world))
        packet = {"error": None, "records": []}
        started = time.perf_counter()
        if rank == 0:
            print(f"[eval:medmcqa] step={state.global_step}, questions={len(self.prompts)}, n=1, max_tokens={self.args.eval_max_new_tokens}", flush=True)
        try:
            params = [SamplingParams(n=1, temperature=self.args.eval_temperature, top_p=0.95, top_k=20, min_p=0.0, max_tokens=self.args.eval_max_new_tokens, seed=self.args.eval_seed + i) for i in indices]
            outputs = trainer.vllm_engine.generate([self.prompts[i] for i in indices], params, use_tqdm=(rank == 0)) if indices else []
            if len(outputs) != len(indices) or any(len(output.outputs) != 1 for output in outputs):
                raise RuntimeError("Evaluation output count does not match the requested questions.")
            for i, output in zip(indices, outputs):
                sample = output.outputs[0]
                prediction = extract_answer(sample.text)
                score = correct_reward(None, [sample.text], [self.answers[i]])[0]
                packet["records"].append({"index": i, "answer": self.answers[i], "prediction": prediction, "response": sample.text, "score": score, "tokens": len(sample.token_ids), "truncated": sample.finish_reason == "length"})
        except Exception as exc:
            packet["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"
        packets = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(packets, packet)
        else:
            packets[0] = packet
        errors = [item["error"] for item in packets if item["error"]]
        if errors:
            raise RuntimeError("MedMCQA evaluation failed: " + "; ".join(errors))
        records = sorted([record for item in packets for record in item["records"]], key=lambda item: item["index"])
        if [record["index"] for record in records] != list(range(len(self.prompts))):
            raise RuntimeError("Incomplete or duplicated distributed evaluation.")
        count = len(records)
        accuracy = sum(record["score"] for record in records) / count
        metrics = {
            "train/global_step": state.global_step,
            "eval_correct_reward": accuracy,
            "eval_medmcqa_accuracy": accuracy,
            "eval_medmcqa_truncated": sum(record["truncated"] for record in records) / count,
            "eval_medmcqa_invalid_answer": sum(record["prediction"] is None for record in records) / count,
            "eval_medmcqa_mean_tokens": sum(record["tokens"] for record in records) / count,
        }
        if rank == 0:
            output_dir = Path(trainer.args.output_dir) / "eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {"dataset": EVAL_DATASET, "metrics": metrics, "results": records}
            (output_dir / f"step_{state.global_step:05d}_medmcqa.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"[eval:medmcqa] accuracy={accuracy:.4f}, time={(time.perf_counter() - started):.1f}s; {metrics}", flush=True)
            if wandb.run is not None:
                wandb.log(metrics)
        self._last_eval_step = state.global_step

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_eval(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.args.reward_eval_steps == 0:
            self._run_eval(state)


@dataclass
class CustomScriptArguments(ScriptArguments):
    data_revision: str = field(default="main")
    max_train_samples: int = field(default=0)
    eval_size: int = field(default=0)
    reward_eval_steps: int = field(default=25)
    eval_max_new_tokens: int = field(default=8192)
    eval_temperature: float = field(default=0.6)
    eval_seed: int = field(default=42)
    preflight_only: bool = field(default=False)
    use_tinker_loss: bool = field(default=False)
    fixed_teacher: bool = field(default=True)
    run_config: str = field(default=None)
    presence_penalty: float = field(default=0.0)
    reason_first: bool = field(default=False)
    top_k_loss: int = field(default=0)
    jsd_token_clip: float = field(default=0.0)
    use_ema_teacher: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    student_thinking: bool = field(default=False)
    teacher_thinking: bool = field(default=False)


def main():
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.model_name_or_path = model_args.model_name_or_path or "Qwen/Qwen3-8B"
    if script_args.student_thinking or script_args.teacher_thinking:
        raise ValueError("Use student_thinking=False and teacher_thinking=False for this run.")
    if not script_args.fixed_teacher or not model_args.use_peft or script_args.use_ema_teacher or script_args.reason_first:
        raise ValueError("Use fixed_teacher=True, use_peft=True, use_ema_teacher=False and reason_first=False.")
    if script_args.use_tinker_loss or training_args.beta != 0 or training_args.lmbda != 1:
        raise ValueError("Keep beta=0, lmbda=1 and use_tinker_loss=False for the existing objective.")
    if training_args.eval_strategy != "no" or not training_args.use_vllm or training_args.vllm_mode != "colocate" or training_args.vllm_tensor_parallel_size != 1 or training_args.vllm_enable_sleep_mode:
        raise ValueError("Use eval_strategy=no, colocated vLLM, tensor parallel size 1 and sleep mode disabled.")
    if min(script_args.max_train_samples, script_args.eval_size, script_args.top_k_loss) < 0 or min(script_args.reward_eval_steps, script_args.eval_max_new_tokens) < 1:
        raise ValueError("Invalid data, evaluation or top-k settings.")
    if not math.isfinite(script_args.eval_temperature) or script_args.eval_temperature < 0 or not math.isfinite(script_args.jsd_token_clip) or script_args.jsd_token_clip < 0:
        raise ValueError("Temperature and token clipping must be finite and nonnegative.")
    if training_args.max_length is None or training_args.max_length <= max(training_args.max_completion_length, script_args.eval_max_new_tokens):
        raise ValueError("max_length must leave room for both the prompt and completion.")
    world = int(os.environ.get("WORLD_SIZE", 1))
    batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world

    dtype = getattr(model_args, "torch_dtype", None) or getattr(model_args, "dtype", None) or "bfloat16"
    if isinstance(dtype, str) and dtype != "auto":
        dtype_map = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
        if dtype.lower() not in dtype_map:
            raise ValueError(f"Unsupported model dtype: {dtype}")
        dtype = dtype_map[dtype.lower()]
    model_kwargs = dict(revision=model_args.model_revision, trust_remote_code=model_args.trust_remote_code, attn_implementation=model_args.attn_implementation or "flash_attention_2", torch_dtype=dtype, use_cache=not training_args.gradient_checkpointing)
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs
    training_args.dataset_kwargs = {**(training_args.dataset_kwargs or {}), "skip_prepare_dataset": True}
    training_args.presence_penalty = script_args.presence_penalty
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision, trust_remote_code=model_args.trust_remote_code, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with training_args.main_process_first(desc="Load MedMCQA"):
        train_dataset, eval_dataset = load_medmcqa(script_args, training_args.seed)
    prompt_budget = training_args.max_length - training_args.max_completion_length
    data_collator = MedMCQADataCollator(tokenizer, prompt_budget)
    eval_callback = VLLMRewardEvalCallback(tokenizer, eval_dataset, script_args, training_args.max_length)
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"MedMCQA: {len(train_dataset)} training questions; {len(eval_dataset)} evaluation questions; non-thinking; reference-conditioned teacher.", flush=True)
    if script_args.preflight_only:
        for start in range(0, len(train_dataset), 64):
            features = [train_dataset[i] for i in range(start, min(start + 64, len(train_dataset)))]
            students, teachers = render_prompt_pairs(features, tokenizer)
            for role, prompts in (("student", students), ("teacher", teachers)):
                lengths = [len(ids) for ids in tokenizer(prompts, truncation=False, add_special_tokens=False)["input_ids"]]
                if max(lengths) > prompt_budget:
                    raise ValueError(f"{role} prompt near training row {start} exceeds the prompt budget.")
        print("Preflight passed: data, answer keys and all training/evaluation prompt lengths checked. No model weights loaded.", flush=True)
        return
    run_config = script_args.run_config or "opsd_medmcqa"
    if Path(training_args.output_dir).name != run_config:
        training_args.output_dir = str(Path(training_args.output_dir) / run_config)
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    run_name = f"{run_config}_lr{lr_str}_bs{batch_size}"
    training_args.run_name = run_name
    if int(os.environ.get("RANK", 0)) == 0:
        config = {**asdict(script_args), "model_name": model_args.model_name_or_path, "learning_rate": training_args.learning_rate, "effective_batch_size": batch_size, "max_steps": training_args.max_steps, "max_completion_length": training_args.max_completion_length, "temperature": training_args.temperature, "max_length": training_args.max_length, "lr_scheduler_type": str(training_args.lr_scheduler_type), "lora_r": model_args.lora_r, "lora_alpha": model_args.lora_alpha, "beta": training_args.beta, "max_grad_norm": training_args.max_grad_norm, "train_dataset": TRAIN_DATASET, "eval_dataset": EVAL_DATASET, "train_size": len(train_dataset), "eval_count": len(eval_dataset), "train_fingerprint": train_dataset._fingerprint, "eval_fingerprint": eval_dataset._fingerprint, "teacher_context": "released teacher_response in the existing reference-conditioned prompt"}
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        wandb.init(entity=training_args.wandb_entity, project=training_args.wandb_project, name=run_name, config=config)
        wandb.define_metric("train/global_step")
        wandb.define_metric("eval_*", step_metric="train/global_step")

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=False,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=False,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=False,
        student_thinking=False,
        teacher_thinking=False,

    )
    trainer.model_accepts_loss_kwargs = False
    eval_callback.trainer = trainer
    trainer.add_callback(eval_callback)
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
