# Single-token external/self-teacher intervention study; separate from training and benchmark scores.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
#!/usr/bin/env python3
"""Frozen Qwen single-token intervention study. Run from the repository root."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

RULES = ("random_disagreement", "highest_kl")
CATEGORIES = ("helpful", "unchanged", "harmful")
ANSWER_BOUNDARY = re.compile(r"\\(?:boxed|fbox)\s*\{|<answer\b|\bfinal\s+answer\b", re.I)
PROMPT = "Problem: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def append_json(f, value):
    f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    f.flush()


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode(tokenizer, ids, special=False):
    return tokenizer.decode(ids, skip_special_tokens=not special, clean_up_tokenization_spaces=False)


def stage_path(root, stage, rank):
    return root / "shards" / f"{stage}_{rank:02d}.jsonl"


def load_stage(root, stage, workers):
    return [row for rank in range(workers) for row in read_jsonl(stage_path(root, stage, rank))]


def tokenizers_for(config):
    from transformers import AutoTokenizer
    student = AutoTokenizer.from_pretrained(config["student"], revision=config["student_commit"], use_fast=True)
    teacher = AutoTokenizer.from_pretrained(config["teacher"], revision=config["teacher_commit"], use_fast=True)
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Teacher/student token-to-ID maps differ. Token replacement and full-vocabulary KL would be invalid.")
    return student, teacher


def render(tokenizer, question):
    messages = [{"role": "user", "content": PROMPT.format(question=question)}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)


def load_grader(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("intervention_math_grader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "boxed_reward_fn", None)) or not callable(getattr(module, "timeout", None)):
        raise ValueError("The grader must export boxed_reward_fn and timeout, as in your existing math runs.")
    return module


def initialize(args, root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoConfig, GenerationConfig
    grader_path = Path(args.grader).resolve()
    if not grader_path.is_file():
        raise FileNotFoundError(f"Missing {grader_path}. Pass --grader /path/to/your/existing/math_grader.py.")
    load_grader(grader_path)
    requested = {k: v for k, v in vars(args).items() if k not in ("stage", "worker_stage", "rank", "output")}
    requested["grader"] = str(grader_path)
    config_path = root / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config["requested"] != requested:
            raise ValueError("Arguments differ from this output directory's saved configuration. Use a new --output.")
        if config["grader_sha256"] != hashlib.sha256(grader_path.read_bytes()).hexdigest():
            raise ValueError("The reward checker changed. Use the original checker or a new output directory.")
        if config["code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("The experiment script changed. Use a new output directory rather than mixing implementations.")
        if config["questions_sha256"] != hashlib.sha256((root / "questions.jsonl").read_bytes()).hexdigest():
            raise ValueError("Saved questions were modified.")
        (root / "shards").mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        return config
    if any(root.glob("shards/*.jsonl")):
        raise ValueError("Found stage outputs without a manifest; use a clean output directory.")
    api = HfApi()
    config = {**requested, "requested": requested, "workers": len(args.gpus.split(",")), "schema_version": 1}
    config["student_commit"] = api.model_info(args.student, revision=args.student_revision).sha
    config["teacher_commit"] = api.model_info(args.teacher, revision=args.teacher_revision).sha
    config["dataset_commit"] = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    student_tok, teacher_tok = tokenizers_for(config)
    mapped_ids = set(student_tok.get_vocab().values())
    sc = AutoConfig.from_pretrained(args.student, revision=config["student_commit"])
    tc = AutoConfig.from_pretrained(args.teacher, revision=config["teacher_commit"])
    if sc.model_type != "qwen3" or tc.model_type != "qwen3" or sc.vocab_size != tc.vocab_size:
        raise ValueError("This script expects two dense Qwen3 causal LMs with the same output vocabulary.")
    if max(student_tok.get_vocab().values()) >= sc.vocab_size:
        raise ValueError("Tokenizer IDs exceed the model vocabulary.")
    if args.max_model_len > min(sc.max_position_embeddings, tc.max_position_embeddings):
        raise ValueError("--max-model-len exceeds one of the model context windows.")
    gc = GenerationConfig.from_pretrained(args.student, revision=config["student_commit"])
    eos = gc.eos_token_id
    if eos is None:
        raise ValueError("The student generation configuration has no EOS token.")
    config["eos_token_ids"] = [eos] if isinstance(eos, int) else list(eos)
    config["vocab_size"] = sc.vocab_size
    config["token_map_sha256"] = digest(student_tok.get_vocab())
    config["special_token_ids"] = sorted(set(student_tok.all_special_ids + teacher_tok.all_special_ids + config["eos_token_ids"]))
    ds = load_dataset(args.dataset, split="train", revision=config["dataset_commit"])
    ds = ds.add_column("ablation_source_index", list(range(len(ds))))
    heldout = ds.train_test_split(test_size=512, seed=42)["test"]
    questions = []
    for index, row in enumerate(heldout.select(range(args.num_questions))):
        question = row["problem"]
        if not isinstance(question, str) or not question.strip() or row["Answer"] is None:
            raise ValueError(f"Invalid question/answer at holdout index {index}.")
        sp, tp = render(student_tok, question), render(teacher_tok, question)
        if len(sp) + args.max_tokens > args.max_model_len or len(tp) + args.max_prefix > args.max_model_len:
            raise ValueError(f"Question {index} exceeds the context budget; increase --max-model-len. No silent prompt truncation.")
        if not set(sp).issubset(mapped_ids) or not set(tp).issubset(mapped_ids):
            raise ValueError("Invalid prompt token ID.")
        questions.append({"id": index, "source_index": row["ablation_source_index"], "question": question,
                          "answer": str(row["Answer"]), "student_prompt_ids": sp, "teacher_prompt_ids": tp})
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "questions.jsonl").open("w", encoding="utf-8") as f:
        for row in questions:
            append_json(f, row)
    config.update({"questions_sha256": hashlib.sha256((root / "questions.jsonl").read_bytes()).hexdigest(),
                   "grader_sha256": hashlib.sha256(grader_path.read_bytes()).hexdigest(),
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "selection_split": "512 held-out rows from train_test_split(test_size=512, seed=42); first num_questions",
                   "teacher_context": "Same question/instructions, teacher's own chat template; no answer/reference supplied",
                   "student_thinking": False, "teacher_thinking": False, "kl_temperature": 1.0,
                   "greedy_temperature": 0.0, "versions": {}})
    for name in ("torch", "transformers", "datasets", "vllm", "numpy"):
        try:
            config["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            config["versions"][name] = None
    write_json(config_path, config)
    return config


def make_engine(config):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if config["batch_invariant"] else "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from vllm import LLM
    if config["batch_invariant"]:
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
        if not vllm_is_batch_invariant():
            raise RuntimeError("vLLM batch-invariant mode did not enable. Check the vLLM installation.")
    return LLM(model=config["student"], revision=config["student_commit"], tokenizer_revision=config["student_commit"],
               dtype="bfloat16", tensor_parallel_size=1, max_model_len=config["max_model_len"],
               gpu_memory_utilization=config["gpu_memory_utilization"], max_num_seqs=config["batch_size"],
               max_num_batched_tokens=8192, enable_prefix_caching=False, enable_chunked_prefill=True,
               enforce_eager=True, seed=config["seed"], generation_config="vllm", disable_log_stats=True)


def sampling(config, remaining):
    from vllm import SamplingParams
    if remaining < 1:
        raise ValueError("No continuation budget remains.")
    return SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0,
                          presence_penalty=0.0, frequency_penalty=0.0, max_tokens=remaining,
                          stop_token_ids=config["eos_token_ids"], seed=config["seed"], skip_special_tokens=False)


def generate(engine, prompts, budgets, config):
    outputs = engine.generate([{"prompt_token_ids": p} for p in prompts], [sampling(config, b) for b in budgets], use_tqdm=False)
    if len(outputs) != len(prompts):
        raise RuntimeError("Generation returned an unexpected response count.")
    result = []
    for output, prompt in zip(outputs, prompts):
        if len(output.outputs) != 1 or list(output.prompt_token_ids) != prompt:
            raise RuntimeError("Generation changed a prompt or response count.")
        sample = output.outputs[0]
        if sample.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected finish reason: {sample.finish_reason}")
        result.append({"token_ids": list(sample.token_ids), "finish_reason": sample.finish_reason,
                       "stop_reason": sample.stop_reason, "truncated": sample.finish_reason == "length"})
    return result


def baseline_stage(config, root, rank):
    path = stage_path(root, "baseline", rank)
    done = {r["id"] for r in read_jsonl(path)}
    rows = [r for r in read_jsonl(root / "questions.jsonl") if r["id"] % config["workers"] == rank and r["id"] not in done]
    if not rows:
        return
    engine = make_engine(config)
    tokenizer = engine.get_tokenizer()
    with path.open("a", encoding="utf-8") as f:
        for start in range(0, len(rows), config["batch_size"]):
            chunk = rows[start:start + config["batch_size"]]
            outputs = generate(engine, [r["student_prompt_ids"] for r in chunk], [config["max_tokens"]] * len(chunk), config)
            for row, output in zip(chunk, outputs):
                output.update({"id": row["id"], "response": decode(tokenizer, output["token_ids"])})
                append_json(f, output)
            print(f"[baseline rank {rank}] {min(start + len(chunk), len(rows))}/{len(rows)} saved", flush=True)


def eligible_end(tokenizer, ids, max_prefix):
    end = min(max_prefix, len(ids) - 1)
    text = decode(tokenizer, ids, special=True)
    boundary = ANSWER_BOUNDARY.search(text)
    if boundary:
        for j in range(end + 1):
            if len(decode(tokenizer, ids[:j + 1], special=True)) > boundary.start():
                end = j - 1
                break
    return end


def choose_positions(positions, seed, question_id):
    eligible = [p for p in positions if p["eligible"]]
    if not eligible:
        return {}
    rng = random.Random(f"{seed}:{question_id}:random_disagreement")
    return {"random_disagreement": rng.choice(eligible), "highest_kl": max(eligible, key=lambda p: (p["kl_ts"], -p["position"]))}


def distribution_stats(s_logits, t_logits):
    import torch
    s = torch.log_softmax(s_logits.float(), dim=-1)
    t = torch.log_softmax(t_logits.float(), dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    if not torch.isfinite(kl).all() or kl.min().item() < -1e-4:
        raise RuntimeError("Invalid KL divergence.")
    sa, ta = s.argmax(-1), t.argmax(-1)
    s2, t2 = s.topk(2, dim=-1).values, t.topk(2, dim=-1).values
    return {"kl_ts": kl.clamp_min(0).tolist(), "student_token_id": sa.tolist(), "teacher_token_id": ta.tolist(),
            "s_student": s.gather(-1, sa[:, None]).exp().flatten().tolist(),
            "s_teacher": s.gather(-1, ta[:, None]).exp().flatten().tolist(),
            "t_student": t.gather(-1, sa[:, None]).exp().flatten().tolist(),
            "t_teacher": t.gather(-1, ta[:, None]).exp().flatten().tolist(),
            "student_logit_margin": (s2[:, 0] - s2[:, 1]).tolist(), "teacher_logit_margin": (t2[:, 0] - t2[:, 1]).tolist()}


def score_question(student, teacher, st, tt, question, baseline, config):
    import torch
    ids = baseline["token_ids"]
    end = eligible_end(st, ids, config["max_prefix"])
    if end < config["min_prefix"]:
        return {"id": question["id"], "positions": [], "choices": {}, "reason": "no_intermediate_window", "scan_end": end}
    js = list(range(config["min_prefix"], end + 1))
    # The state at P+j-1 predicts completion token j. Future completion tokens are never attended to.
    with torch.inference_mode():
        def hidden(model, prompt):
            inputs = torch.tensor([prompt + ids[:end]], device=next(model.parameters()).device, dtype=torch.long)
            h = model.model(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
            offsets = torch.tensor([len(prompt) + j - 1 for j in js], device=h.device)
            return h[0].index_select(0, offsets).contiguous()
        sh, th = hidden(student, question["student_prompt_ids"]), hidden(teacher, question["teacher_prompt_ids"])
        banned = set(config["special_token_ids"])
        mapped = set(st.get_vocab().values())
        positions = []
        for start in range(0, len(js), config["score_chunk_size"]):
            stop = start + config["score_chunk_size"]
            sl, tl = student.get_output_embeddings()(sh[start:stop]), teacher.get_output_embeddings()(th[start:stop])
            if sl.shape != tl.shape:
                raise RuntimeError("Teacher/student logit shapes differ.")
            stats = distribution_stats(sl, tl)
            for k, j in enumerate(js[start:stop]):
                p = {key: value[k] for key, value in stats.items()}
                p.update({"position": j, "baseline_token_id": ids[j]})
                a, b = p["student_token_id"], p["teacher_token_id"]
                p["student_token_text"], p["teacher_token_text"] = decode(st, [a], True), decode(tt, [b], True)
                reason = None
                if a != ids[j]:
                    reason = "student_hf_vllm_argmax_mismatch"
                elif a == b:
                    reason = "same_preferred_token"
                elif a in banned or b in banned or a not in mapped or b not in mapped:
                    reason = "special_or_unmapped_token"
                elif ANSWER_BOUNDARY.search(decode(st, ids[:j] + [b], True)):
                    reason = "teacher_enters_answer_boundary"
                p.update({"eligible": reason is None, "exclusion_reason": reason})
                positions.append(p)
            del sl, tl
    choices = choose_positions(positions, config["seed"], question["id"])
    return {"id": question["id"], "scan_end": end, "positions": positions, "choices": choices,
            "reason": None if choices else "no_eligible_disagreement"}


def select_stage(config, root, rank):
    import torch
    from transformers import AutoModelForCausalLM
    path = stage_path(root, "selection", rank)
    done = {r["id"] for r in read_jsonl(path)}
    rows = [r for r in read_jsonl(root / "questions.jsonl") if r["id"] % config["workers"] == rank and r["id"] not in done]
    if not rows:
        return
    bases = {r["id"]: r for r in load_stage(root, "baseline", config["workers"])}
    st, tt = tokenizers_for(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": config["score_attention"], "low_cpu_mem_usage": True}
    student = AutoModelForCausalLM.from_pretrained(config["student"], revision=config["student_commit"], **kwargs).to("cuda").eval()
    teacher = AutoModelForCausalLM.from_pretrained(config["teacher"], revision=config["teacher_commit"], **kwargs).to("cuda").eval()
    student.requires_grad_(False)
    teacher.requires_grad_(False)
    with path.open("a", encoding="utf-8") as f:
        for number, row in enumerate(rows, 1):
            result = score_question(student, teacher, st, tt, row, bases[row["id"]], config)
            append_json(f, result)
            print(f"[selection rank {rank}] {number}/{len(rows)}; eligible positions={sum(p['eligible'] for p in result['positions'])}", flush=True)


def intervention_stage(config, root, rank):
    path = stage_path(root, "interventions", rank)
    done = {(r["id"], r["position"]) for r in read_jsonl(path)}
    questions = {r["id"]: r for r in read_jsonl(root / "questions.jsonl")}
    bases = {r["id"]: r for r in load_stage(root, "baseline", config["workers"])}
    jobs = []
    for row in read_jsonl(stage_path(root, "selection", rank)):
        for pos in sorted({p["position"] for p in row["choices"].values()}):
            if (row["id"], pos) not in done:
                choice = next(p for p in row["choices"].values() if p["position"] == pos)
                jobs.append((questions[row["id"]], bases[row["id"]], choice))
    if not jobs:
        return
    engine = make_engine(config)
    tokenizer = engine.get_tokenizer()
    pairs_per_batch = max(1, config["batch_size"] // 2)
    with path.open("a", encoding="utf-8") as f:
        for start in range(0, len(jobs), pairs_per_batch):
            chunk = jobs[start:start + pairs_per_batch]
            prompts, budgets = [], []
            for question, baseline, choice in chunk:
                j = choice["position"]
                shared = question["student_prompt_ids"] + baseline["token_ids"][:j]
                prompts.extend([shared + [choice["student_token_id"]], shared + [choice["teacher_token_id"]]])
                budgets.extend([config["max_tokens"] - j - 1] * 2)
            outputs = generate(engine, prompts, budgets, config)
            for k, (question, baseline, choice) in enumerate(chunk):
                j = choice["position"]
                replay, intervention = outputs[2 * k:2 * k + 2]
                prefix = baseline["token_ids"][:j]
                replay["token_ids"] = prefix + [choice["student_token_id"]] + replay["token_ids"]
                intervention["token_ids"] = prefix + [choice["teacher_token_id"]] + intervention["token_ids"]
                for output in (replay, intervention):
                    output["response"] = decode(tokenizer, output["token_ids"])
                match = replay["token_ids"] == baseline["token_ids"]
                append_json(f, {"id": question["id"], "position": j, "prefix_completion_token_ids": prefix,
                                "prefix_text": decode(tokenizer, prefix, True), "choice": choice, "replay_exact_match": match,
                                "baseline_replay": replay, "intervention": intervention})
            print(f"[interventions rank {rank}] {min(start + len(chunk), len(jobs))}/{len(jobs)} token pairs saved", flush=True)


def grade_response(grader, text, answer, seconds):
    try:
        with grader.timeout(seconds):
            _, value = grader.boxed_reward_fn(text, answer, fast=True)
        value = float(value)
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise ValueError(f"Expected binary correctness; got {value}.")
        return {"correct": int(value), "status": "ok", "error": None}
    except Exception as exc:
        return {"correct": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def metric_vectors(b, t):
    import numpy as np
    b, t = np.asarray(b), np.asarray(t)
    helpful, harmful, unchanged = (b == 0) & (t == 1), (b == 1) & (t == 0), b == t
    return {"helpful": helpful, "unchanged": unchanged, "harmful": harmful,
            "still_correct": (b == 1) & (t == 1), "still_wrong": (b == 0) & (t == 0),
            "baseline_accuracy": b, "intervention_accuracy": t, "net_accuracy_change": t.astype(float) - b,
            "rescue_rate": helpful, "damage_rate": harmful}


def statistics_for(b, t, draws=None):
    import numpy as np
    vectors = metric_vectors(b, t)
    b = np.asarray(b)
    if draws is None:
        return {name: (float(v[b == 0].mean()) if np.any(b == 0) else None) if name == "rescue_rate" else
                      (float(v[b == 1].mean()) if np.any(b == 1) else None) if name == "damage_rate" else float(v.mean())
                for name, v in vectors.items()}
    result = {}
    for name, v in vectors.items():
        num = v[draws].sum(-1)
        den = (b[draws] == (0 if name == "rescue_rate" else 1)).sum(-1) if name in ("rescue_rate", "damage_rate") else np.full(len(draws), len(b))
        result[name] = np.divide(num, den, out=np.full(len(draws), np.nan), where=den > 0)
    return result


def bootstrap_summary(pairs, population, config):
    import numpy as np
    by_rule = {rule: {p["id"]: p for p in pairs if p["rule"] == rule and p[population]} for rule in RULES}
    ids = sorted(set(by_rule[RULES[0]]) & set(by_rule[RULES[1]]))
    if not ids:
        return [], [], []
    b = np.array([by_rule[RULES[0]][i]["baseline_correct"] for i in ids], dtype=int)
    t = {rule: np.array([by_rule[rule][i]["intervention_correct"] for i in ids], dtype=int) for rule in RULES}
    point = {rule: statistics_for(b, t[rule]) for rule in RULES}
    boots = {rule: {m: [] for m in point[rule]} for rule in RULES}
    rng = np.random.default_rng(config["seed"] + 8191)
    for start in range(0, config["bootstrap_samples"], 128):
        draws = rng.integers(0, len(ids), size=(min(128, config["bootstrap_samples"] - start), len(ids)))
        for rule in RULES:
            values = statistics_for(b, t[rule], draws)
            for metric, value in values.items():
                boots[rule][metric].append(value)
    boots = {rule: {m: np.concatenate(v) for m, v in d.items()} for rule, d in boots.items()}
    def ci(values):
        finite = values[np.isfinite(values)]
        return [float(x) for x in np.quantile(finite, [0.025, 0.975])] if len(finite) else [None, None]
    rows, differences = [], []
    for rule in RULES:
        for metric, value in point[rule].items():
            den = int((b == 0).sum()) if metric == "rescue_rate" else int((b == 1).sum()) if metric == "damage_rate" else len(ids)
            lo, hi = ci(boots[rule][metric])
            rows.append({"population": population, "rule": rule, "metric": metric, "n_questions": len(ids), "denominator": den,
                         "numerator": None if value is None else round(value * den), "fraction": value,
                         "percent": None if value is None else 100 * value, "ci95_low": lo, "ci95_high": hi,
                         "ci95_low_percent": None if lo is None else 100 * lo, "ci95_high_percent": None if hi is None else 100 * hi})
    for metric in point[RULES[0]]:
        a, z = point[RULES[1]][metric], point[RULES[0]][metric]
        lo, hi = ci(boots[RULES[1]][metric] - boots[RULES[0]][metric])
        differences.append({"population": population, "contrast": "highest_kl - random_disagreement", "metric": metric,
                            "n_questions": len(ids), "difference": None if a is None or z is None else a - z,
                            "ci95_low": lo, "ci95_high": hi})
    return rows, differences, ids


def summarize(config, root):
    questions = {r["id"]: r for r in read_jsonl(root / "questions.jsonl")}
    bases = {r["id"]: r for r in load_stage(root, "baseline", config["workers"])}
    selections = {r["id"]: r for r in load_stage(root, "selection", config["workers"])}
    interventions = {(r["id"], r["position"]): r for r in load_stage(root, "interventions", config["workers"])}
    if set(bases) != set(questions) or set(selections) != set(questions):
        raise RuntimeError("Baseline/selection stages are incomplete; refusing to produce a partial figure.")
    grader = load_grader(Path(config["grader"]))
    base_grades = {i: grade_response(grader, bases[i]["response"], q["answer"], config["grader_timeout"]) for i, q in questions.items()}
    intervention_grades = {key: grade_response(grader, r["intervention"]["response"], questions[key[0]]["answer"], config["grader_timeout"])
                           for key, r in interventions.items()}
    pairs, audit = [], []
    for i, q in questions.items():
        for rule, choice in selections[i]["choices"].items():
            key = (i, choice["position"])
            if key not in interventions:
                raise RuntimeError(f"Missing continuation for {key}; no partial summary is allowed.")
            r = interventions[key]
            bg, tg = base_grades[i], intervention_grades[key]
            b, t = bg["correct"], tg["correct"]
            valid_grade = bg["status"] == tg["status"] == "ok"
            verified = valid_grade and r["replay_exact_match"]
            category = ("helpful" if t > b else "harmful" if t < b else "unchanged") if valid_grade else None
            pair = {"id": i, "source_index": q["source_index"], "rule": rule, "position": choice["position"],
                    "kl_ts": choice["kl_ts"], "student_token_id": choice["student_token_id"], "teacher_token_id": choice["teacher_token_id"],
                    "student_token_text": choice["student_token_text"], "teacher_token_text": choice["teacher_token_text"],
                    "baseline_correct": b, "intervention_correct": t, "category": category,
                    "transition": f"{b}->{t}" if valid_grade else None, "replay_exact_match": r["replay_exact_match"],
                    "baseline_truncated": bases[i]["truncated"], "intervention_truncated": r["intervention"]["truncated"],
                    "verified": verified, "verified_untruncated": verified and not bases[i]["truncated"] and not r["intervention"]["truncated"],
                    "baseline_grade_error": bg["error"], "intervention_grade_error": tg["error"]}
            pairs.append(pair)
            audit.append({**pair, "question": q["question"], "answer_key": q["answer"],
                          "student_prompt_token_ids": q["student_prompt_ids"], "teacher_prompt_token_ids": q["teacher_prompt_ids"],
                          "prefix_completion_token_ids": r["prefix_completion_token_ids"], "prefix_text": r["prefix_text"],
                          "choice": choice, "baseline": bases[i], "intervention": r["intervention"], "baseline_replay": r["baseline_replay"]})
    rows, differences, primary_ids = bootstrap_summary(pairs, "verified", config)
    sensitivity, sensitivity_diff, sensitivity_ids = bootstrap_summary(pairs, "verified_untruncated", config)
    included = set(primary_ids)
    for pair in pairs:
        pair["in_primary_figure"] = pair["id"] in included
    for row in audit:
        row["in_primary_figure"] = row["id"] in included
    write_csv(root / "plot_summary.csv", rows)
    write_csv(root / "plot_summary_untruncated.csv", sensitivity)
    write_csv(root / "paired_rule_differences.csv", differences + sensitivity_diff)
    write_csv(root / "pairs.csv", pairs)
    with (root / "baseline_scores.jsonl").open("w", encoding="utf-8") as f:
        for i in sorted(questions):
            append_json(f, {**bases[i], "answer_key": questions[i]["answer"], "grade": base_grades[i]})
    with (root / "audit_records.jsonl").open("w", encoding="utf-8") as f:
        for row in audit:
            append_json(f, row)
    eligible_ids = [i for i, s in selections.items() if s["choices"]]
    coverage = {"n_requested": len(questions), "n_with_eligible_disagreement": len(eligible_ids),
                "n_primary_common_verified": len(primary_ids), "n_untruncated_common_verified": len(sensitivity_ids),
                "primary_question_ids": primary_ids, "untruncated_question_ids": sensitivity_ids,
                "no_intervention_reasons": dict(Counter(s["reason"] for s in selections.values() if not s["choices"])),
                "position_exclusions": dict(Counter(p["exclusion_reason"] for s in selections.values() for p in s["positions"] if not p["eligible"])),
                "replay_mismatch_pairs": sum(not p["replay_exact_match"] for p in pairs),
                "baseline_grader_failures": sum(g["status"] != "ok" for g in base_grades.values()),
                "intervention_grader_failures": sum(g["status"] != "ok" for g in intervention_grades.values()),
                "baseline_truncated_all": sum(b["truncated"] for b in bases.values()),
                "intervention_truncated_unique": sum(r["intervention"]["truncated"] for r in interventions.values())}
    write_json(root / "summary.json", {"coverage": coverage, "statistics": rows, "untruncated_statistics": sensitivity,
                                       "interpretation": "One-token interventions on frozen greedy student completions; unchanged means unchanged binary correctness.",
                                       "denominator": "Same questions for both rules: eligible, successful grading, exact original-token replay for BOTH interventions.",
                                       "ci": "95% paired question-bootstrap percentile intervals; random location drawn once per question, not resampled inside bootstrap.",
                                       "budget": "Forced token counts toward the same total completion-token budget in both branches."})
    print(json.dumps(coverage, indent=2), flush=True)
    for rule in RULES:
        chosen = {r["metric"]: r for r in rows if r["rule"] == rule}
        if chosen:
            text = ", ".join(f"{k}={chosen[k]['percent']:.2f}%" for k in CATEGORIES)
            print(f"{rule}: n={len(primary_ids)}; {text}", flush=True)
    if len(primary_ids) < len(eligible_ids):
        print("WARNING: Some questions failed replay/grading checks and are excluded from BOTH primary bars. See summary.json.", flush=True)
    if not primary_ids:
        raise RuntimeError("No verified paired cases; raw outputs were saved, but there is no valid plot summary.")


def run_stage(stage, config, root):
    processes = []
    try:
        for rank, gpu in enumerate(config["gpus"].split(",")):
            log = (root / "logs" / f"{stage}_{rank:02d}.log").open("a", buffering=1)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONUNBUFFERED": "1"}
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-stage", stage, "--rank", str(rank), "--output", str(root)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append((rank, process, log))
        while any(p.poll() is None for _, p, _ in processes):
            failed = [(rank, p.returncode) for rank, p, _ in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"{stage} worker failed: {failed}. Inspect {root / 'logs'}. Completed records are preserved.")
            time.sleep(2)
        failed = [(rank, p.returncode) for rank, p, _ in processes if p.returncode]
        if failed:
            raise RuntimeError(f"{stage} worker failed: {failed}. Inspect {root / 'logs'}.")
    finally:
        for _, process, log in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default="Qwen/Qwen3-1.7B")
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--student-revision", default="main")
    p.add_argument("--teacher-revision", default="main")
    p.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--num-questions", type=int, default=512)
    p.add_argument("--min-prefix", type=int, default=32, help="Minimum number of generated tokens BEFORE the replaced token.")
    p.add_argument("--max-prefix", type=int, default=256, help="Maximum number of generated tokens BEFORE the replaced token (inclusive).")
    p.add_argument("--max-tokens", type=int, default=8192, help="Total completion budget including shared prefix and forced token.")
    p.add_argument("--max-model-len", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--score-chunk-size", type=int, default=32)
    p.add_argument("--score-attention", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    p.add_argument("--batch-invariant", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default="0", help="GPU IDs, one independent worker per GPU; e.g. 0,1,2,3.")
    p.add_argument("--grader", default=str(Path(__file__).with_name("math_grader.py")))
    p.add_argument("--grader-timeout", type=int, default=3)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--output", default="teacher_intervention_results")
    p.add_argument("--stage", choices=("all", "prepare", "baseline", "select", "intervene", "summarize"), default="all")
    p.add_argument("--worker-stage", choices=("baseline", "select", "intervene"), help=argparse.SUPPRESS)
    p.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = arguments()
    root = Path(args.output).resolve()
    if args.worker_stage:
        config = json.loads((root / "config.json").read_text())
        {"baseline": baseline_stage, "select": select_stage, "intervene": intervention_stage}[args.worker_stage](config, root, args.rank)
        return
    if args.stage == "summarize":
        config = json.loads((root / "config.json").read_text())
        if hashlib.sha256(Path(config["grader"]).read_bytes()).hexdigest() != config["grader_sha256"]:
            raise ValueError("Grader changed since the experiment was prepared.")
        summarize(config, root)
        return
    if not 1 <= args.num_questions <= 512 or not 1 <= args.min_prefix <= args.max_prefix < args.max_tokens - 1:
        raise ValueError("Require 1 <= num_questions <= 512 and 1 <= min_prefix <= max_prefix < max_tokens-1.")
    if args.batch_size < 2 or args.score_chunk_size < 1 or args.bootstrap_samples < 100 or args.grader_timeout < 1:
        raise ValueError("Invalid batch, chunk, bootstrap, or timeout setting.")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU utilization must be between zero and one.")
    args.gpus = ",".join(x.strip() for x in args.gpus.split(","))
    devices = args.gpus.split(",")
    if any(not x for x in devices) or len(set(devices)) != len(devices) or len(devices) > args.num_questions:
        raise ValueError("GPU IDs must be distinct and nonempty, with at least one question per GPU.")
    config = initialize(args, root)
    print(f"Prepared {config['num_questions']} questions; {config['workers']} GPU worker(s); output={root}", flush=True)
    if args.stage == "prepare":
        return
    for stage in ("baseline", "select", "intervene"):
        if args.stage in ("all", stage):
            print(f"Starting {stage}; progress logs: {root / 'logs'}", flush=True)
            run_stage(stage, config, root)
    if args.stage in ("all", "intervene"):
        summarize(config, root)


if __name__ == "__main__":
    main()
