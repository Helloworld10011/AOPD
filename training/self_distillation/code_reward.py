# Score TACO candidates in isolated Docker containers using five-test fractional rewards.
"""Bounded stdin-only coding verifier; candidate programs execute only in Docker.

Five-test fractional reward and newline-only output comparison match the existing
opsd_train_taco_lcb.py / VERL SandboxFusion path. Full-suite pass rate is separate.
"""

import ast
import json
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor

IMAGE = os.environ.get(
    "AOPD_CODE_IMAGE",
    "ghcr.io/nuprl/multipl-e-evaluation@sha256:2cb4fcd74d89d149f7a3feb177f7d8169cef390a7ea7951c9b84e7ff59ea26cc",
)
TIMEOUT = 10
N_REWARD_TESTS = 5
POOL = ThreadPoolExecutor(max_workers=4)


def check_docker():
    """Ensure the original grading image is available before loading a training model."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            print(
                "Downloading the pinned coding runtime (first run only)...", flush=True
            )
            subprocess.run(["docker", "pull", IMAGE], check=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Self-coding needs access to a running Docker daemon and its pinned grading image."
        ) from error


# Expected outputs never enter the container. Each candidate runs as a child
# process, with fresh stdin and captured output, not in the grading interpreter.
DRIVER = r"""
import json,os,resource,signal,subprocess,sys,tempfile
request=json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
with tempfile.TemporaryDirectory() as d:
    program=os.path.join(d,"program.py")
    with open(program,"w") as f:f.write(request["code"])
    results=[]
    for index,stdin in enumerate(request["inputs"]):
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            p=subprocess.Popen([sys.executable,"-I",program],stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=d,start_new_session=True)
            status="finished"
            try:p.communicate(stdin.encode(),timeout=request["timeout"])
            except subprocess.TimeoutExpired:
                status="timeout"
                os.killpg(p.pid,signal.SIGKILL);p.communicate()
            # Remove any surviving descendants in this candidate's group.
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            out.seek(0);err.seek(0)
            results.append({"status":status,"returncode":p.returncode,"stdout":out.read(1048576).decode(errors="replace"),"stderr":err.read(2048).decode(errors="replace")})
    print(json.dumps(results))
"""


def extract_code(completion):
    if isinstance(completion, list):
        completion = completion[0].get("content", "") if completion else ""
    # Preserve the existing VERL extraction convention, including an unclosed
    # final code fence; syntax/runtime checks still apply to the extracted code.
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    if "```" in completion:
        code = completion.split("```")[1]
        if "\n" in code:
            first, rest = code.split("\n", 1)
            if first.strip().isalpha():
                code = rest
        return code
    # Accept a complete raw Python response when no Markdown fence is present.
    # Never extract prose heuristically; every program still runs in Docker.
    raw = completion.strip()
    if raw:
        try:
            tree = ast.parse(raw)
        except (SyntaxError, ValueError, RecursionError):
            return None
        if tree.body:
            return raw
    return None


def run_code(code, inputs, timeout=TIMEOUT):
    name = "aopd-code-" + uuid.uuid4().hex
    argv = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "1g",
        "--memory-swap",
        "1g",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,nosuid,size=64m",
        "--user",
        "65534:65534",
        "--entrypoint",
        "python3",
        IMAGE,
        "-I",
        "-c",
        DRIVER,
    ]
    try:
        result = subprocess.run(
            argv,
            input=json.dumps({"code": code, "inputs": inputs, "timeout": timeout}),
            text=True,
            capture_output=True,
            timeout=(timeout + 2) * len(inputs) + 30,
        )
        if result.returncode:
            raise RuntimeError(
                f"Docker verifier failed ({result.returncode}): {result.stderr[-2000:]}"
            )
        rows = json.loads(result.stdout)
        if len(rows) != len(inputs):
            raise RuntimeError("Verifier result count mismatch")
        return rows
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        # Infrastructure failures must not silently become negative rewards.
        raise RuntimeError("Docker verifier timed out or returned malformed results")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


def score_details(completion, ground_truth, test_limit=N_REWARD_TESTS):
    tests = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if tests.get("fn_name"):
        raise ValueError("This frozen task supports stdin problems only")
    inputs, outputs = tests["inputs"], tests["outputs"]
    if (
        not inputs
        or len(inputs) != len(outputs)
        or not all(isinstance(x, str) for x in inputs + outputs)
    ):
        raise ValueError("Invalid stdin test cases")
    if test_limit:
        inputs, outputs = inputs[:test_limit], outputs[:test_limit]
    code = extract_code(completion)
    if code is None:
        return {
            "fraction": 0.0,
            "all_pass": 0.0,
            "passed": [False] * len(inputs),
            "missing_code": True,
        }
    rows = run_code(code, inputs)
    passed = [
        r["status"] == "finished"
        and r["returncode"] == 0
        and r["stdout"].rstrip("\n") == target.rstrip("\n")
        for r, target in zip(rows, outputs)
    ]
    return {
        "fraction": sum(passed) / len(passed),
        "all_pass": float(all(passed)),
        "passed": passed,
        "missing_code": False,
        "execution": [
            {k: r[k] for k in ("status", "returncode", "stderr")} for r in rows
        ],
    }


def correct_reward(prompts, completions, ground_truth, **kwargs):
    if len(completions) != len(ground_truth):
        raise ValueError("Completion and test counts differ")
    return [x["fraction"] for x in POOL.map(score_details, completions, ground_truth)]
