# Shared external-teacher task prompts, dataset splits, and terminal rewards.
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

CODE_INSTRUCTION = "Write Python code to solve the problem. Present the code in a Python code block at the end. Think first, then write the Python code in a ```python code block."


def render_example(example, task):
    if task == "math":
        messages = [
            {"role": "system", "content": MATH_INSTRUCTION},
            {"role": "user", "content": example["problem"]},
        ]
    else:
        problem = example["question"]
        if example.get("starter_code"):
            problem += f"\n\nStarter code:\n```python\n{example['starter_code']}\n```"
        messages = [{"role": "user", "content": problem + "\n\n" + CODE_INSTRUCTION}]
    return {"messages": messages}


def load_training_data(args, tokenizer, seed=None):
    from datasets import load_dataset, load_from_disk

    seed = getattr(args, "seed", 42) if seed is None else seed
    if args.task == "math":
        raw = load_dataset(
            "siyanzhao/Openthoughts_math_30k_opsd", split="train"
        ).shuffle(seed=seed)
        raw = raw.select(range(len(raw) // 2))
        train = raw.select(range(256, len(raw)))
    else:
        data_dir = args.data_dir or "data/external_code"
        if not Path(data_dir).is_dir():
            raise FileNotFoundError(
                "Prepare external coding data first: python -m data.prepare --task external_code"
            )
        raw = load_from_disk(data_dir)

        def fits(example):
            text = tokenizer.apply_chat_template(
                render_example(example, "code")["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return len(tokenizer(text)["input_ids"]) <= 2048

        raw = raw.filter(fits).shuffle(seed=seed)
        train = raw.select(range(256, min(5856, len(raw))))
    lookup = {
        tokenizer.apply_chat_template(
            render_example(row, args.task)["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ): row["Answer" if args.task == "math" else "ground_truth"]
        for row in raw
    }
    return (
        train.map(
            lambda row: render_example(row, args.task),
            remove_columns=train.column_names,
        ),
        lookup,
    )


def get_reward(task):
    if task == "math":
        from eval.math_grader import boxed_reward_fn, timeout

        def reward(prompts, completions, answers):
            scores = []
            for text, answer in zip(completions, answers):
                try:
                    with timeout(3):
                        _, score = boxed_reward_fn(text, answer, fast=True)
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            return scores

        return reward
    from .code_reward import compute_score, check_sandbox

    check_sandbox()
    pool = ThreadPoolExecutor(max_workers=4)
    semaphore = threading.Semaphore(20)

    def score(text, tests):
        if "```" not in text:
            return 0.0
        tests = json.loads(tests) if isinstance(tests, str) else tests
        selected = json.dumps({key: tests[key][:5] for key in ("inputs", "outputs")})
        try:
            result, metadata = compute_score(
                "http://127.0.0.1:8080/run_code",
                semaphore,
                1024,
                text,
                selected,
                continuous=False,
                timeout=10,
            )
            if any(item.get("api_request_error") for item in metadata):
                raise RuntimeError(
                    "SandboxFusion failed while scoring a completion. Check the sandbox server."
                )
            return float(result)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("Invalid coding reward tests.") from error

    return lambda prompts, completions, answers: list(
        pool.map(score, completions, answers)
    )
