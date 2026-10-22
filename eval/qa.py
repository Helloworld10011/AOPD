# Original task answer parsers extracted without importing the training entry points.

import re

from typing import Optional

VALID_LETTERS = set("ABCD")


def get_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return ""


def medical_answer(completion):
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


def _normalize_letter(snippet: str, case_sensitive: bool = False) -> Optional[str]:
    if not snippet:
        return None
    cleaned = re.sub("[\\*\\(\\)\\[\\]\\.\\:`\\\"']", " ", snippet).strip()
    if case_sensitive:
        if cleaned in VALID_LETTERS:
            return cleaned
        m = re.match("\\s*([ABCD])\\b", cleaned)
        return m.group(1) if m else None
    else:
        if cleaned.upper() in VALID_LETTERS:
            return cleaned.upper()
        m = re.match("\\s*([ABCD])\\b", cleaned, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None


def biology_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D from a model response with multiple fallbacks."""
    if not text:
        return None
    text = re.sub("<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    matches = re.findall(
        "<answer>\\s*(.*?)\\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if matches:
        cand = _normalize_letter(matches[-1], case_sensitive=False)
        if cand:
            return cand
    m = re.search("<answer>\\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = _normalize_letter(m.group(1), case_sensitive=False)
        if cand:
            return cand
    cue_patterns = [
        "(?:final\\s+answer|the\\s+answer\\s+is|answer\\s*[:\\-])\\s*\\(?\\*?\\*?([ABCDabcd])\\b",
        "\\boption\\s*\\(?\\*?\\*?([ABCDabcd])\\b",
        "\\b([ABCDabcd])\\s*(?:is\\s+(?:the\\s+)?correct|is\\s+right)",
        "\\\\boxed\\{\\s*\\(?([ABCDabcd])\\)?\\s*\\}",
    ]
    for pat in cue_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(
        "^\\s*\\(?\\*?\\*?([ABCD])\\)?\\s*[\\.\\):]", text, flags=re.MULTILINE
    )
    if m:
        return m.group(1)
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("*`()[].:\"' ")
        if stripped in VALID_LETTERS:
            return stripped
    return None


# Official MMLU-Pro formatting/parser functions, revision f418b116db00b065c2aea046518d8fcf74d39872.

import re

choices = list("ABCDEFGHIJKLMNOP")


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def mmlupro_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


# Evaluate frozen prompt/gold JSONL with the original medical, biology, or MMLU-Pro parser.
import argparse, hashlib, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--task", choices=["medical", "biology", "mmlupro"], required=True
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()
    if args.tp < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("--tp and --max_tokens must be positive.")
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError(f"Do not overwrite an evaluation: {dest}")
    from vllm import LLM, SamplingParams

    grader = {
        "medical": medical_answer,
        "biology": biology_answer,
        "mmlupro": mmlupro_answer,
    }[args.task]
    cap = args.max_tokens or (8192 if args.task == "medical" else 4096)
    context = 20000 if args.task == "medical" else 16384
    source = Path(args.data)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert rows and len({x["uid"] for x in rows}) == len(rows)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=context,
        seed=7319,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            x["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for x in rows
    ]
    if any(
        len(tokenizer.encode(p, add_special_tokens=False)) + cap > context
        for p in prompts
    ):
        raise ValueError(
            "A prompt exceeds the fixed context budget; do not silently truncate it."
        )
    parameters = [
        SamplingParams(
            n=1,
            temperature=0.6 if args.task == "medical" else 0.0,
            top_p=0.95 if args.task == "medical" else 1.0,
            top_k=20 if args.task == "medical" else -1,
            min_p=0.0,
            max_tokens=cap,
            seed=42 + i if args.task == "medical" else 7319,
            stop=["Question:"] if args.task == "mmlupro" else [],
        )
        for i in range(len(rows))
    ]
    outputs = llm.generate(prompts, parameters)
    assert len(outputs) == len(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    correct = truncated = invalid = 0
    with dest.open("x") as stream:
        for row, prompt, out in zip(rows, prompts, outputs):
            assert len(out.outputs) == 1
            result = out.outputs[0]
            prediction = grader(result.text)
            ok = prediction == row["gold"]
            record = {
                **row,
                "prediction": prediction,
                "correct": bool(ok),
                "completion": result.text,
                "token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason,
                "tokens": len(result.token_ids),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            correct += ok
            invalid += prediction is None
            truncated += result.finish_reason == "length"
    summary = {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "invalid": invalid,
        "truncated": truncated,
        "task": args.task,
        "model": args.model,
        "revision": args.revision,
        "max_tokens": cap,
        "thinking": False,
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    dest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
