# Prepare verified TACO stdin problems using the original SandboxFusion filtering.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import json
import logging
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
from external_teacher.sandbox_fusion import compute_score as code_compute_score

SANDBOX_URL = "http://127.0.0.1:8080/run_code"
OUT_DIR = BASE_DIR / "taco_stdin_clean"

if OUT_DIR.exists():
    raise FileExistsError(f"Do not overwrite a prepared split: {OUT_DIR}")

MAX_TESTS = 10
MIN_TESTS = 5
MAX_REFERENCE_SOLUTIONS = 5
VERIFY_WORKERS = 8
VERIFY_TIMEOUT = 10

_SEM = threading.Semaphore(32)

logging.getLogger("sandbox_fusion").setLevel(logging.WARNING)
logging.getLogger("sandbox_fusion.utils").setLevel(logging.WARNING)


def parse_json(x):
    if isinstance(x, str):
        return json.loads(x)
    return x


def to_str(x):
    if isinstance(x, (list, tuple)):
        return "\n".join(map(str, x))
    return str(x)


def is_usable_stdin_problem(ex):
    try:
        io = parse_json(ex["input_output"])
        if io.get("fn_name"):
            return False
        if io.get("assert_case"):
            return False

        inputs = io.get("inputs")
        outputs = io.get("outputs")

        return (
            isinstance(inputs, list)
            and isinstance(outputs, list)
            and len(inputs) == len(outputs)
            and len(inputs) >= MIN_TESTS
        )
    except Exception:
        return False


def make_tests(ex):
    io = parse_json(ex["input_output"])
    n = min(MAX_TESTS, len(io["inputs"]), len(io["outputs"]))

    tests = {
        "inputs": [to_str(x) for x in io["inputs"][:n]],
        "outputs": [to_str(x) for x in io["outputs"][:n]],
    }

    return {"ground_truth": json.dumps(tests, ensure_ascii=False)}


def py3_solutions(solutions, limit=MAX_REFERENCE_SOLUTIONS):
    if isinstance(solutions, str):
        try:
            solutions = json.loads(solutions)
        except json.JSONDecodeError:
            solutions = [solutions]

    good = []

    for solution in solutions:
        if not isinstance(solution, str):
            continue

        try:
            compile(solution, "<solution>", "exec")
        except SyntaxError:
            continue

        good.append(solution)

        if len(good) >= limit:
            break

    return good


def verify(ex):
    for solution in py3_solutions(ex["solutions"]):
        try:
            score, _ = code_compute_score(
                sandbox_fusion_url=SANDBOX_URL,
                concurrent_semaphore=_SEM,
                memory_limit_mb=1024,
                completion=f"```python\n{solution}\n```",
                test_cases=ex["ground_truth"],
                continuous=False,
                timeout=VERIFY_TIMEOUT,
            )

            if score == 1.0:
                return True

        except Exception:
            pass

    return False


raw = load_dataset("likaixin/TACO-verified", split="train")

print(f"Original TACO-Verified: {len(raw)}")

raw = raw.filter(is_usable_stdin_problem)

print(f"stdin-only with >= {MIN_TESTS} tests: {len(raw)}")

raw = raw.map(make_tests)

with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as pool:
    keep = list(pool.map(verify, raw))

clean = raw.add_column("_keep", keep)
clean = clean.filter(lambda ex: ex["_keep"])

print(f"Verified under SandboxFusion: {len(clean)} / {len(raw)}")
print("Difficulty distribution:", Counter(clean["difficulty"]))

keep_columns = [
    "id",
    "question",
    "starter_code",
    "difficulty",
    "source",
    "ground_truth",
]

clean = clean.remove_columns([
    c for c in clean.column_names if c not in keep_columns
])

clean.save_to_disk(str(OUT_DIR))

print(f"Saved to: {OUT_DIR}")
