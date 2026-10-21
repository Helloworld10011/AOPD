# AOPD: Advantage-Based On-Policy Distillation

**When to Follow the Teacher: Advantage-Based On-Policy Distillation**

AOPD learns from teacher suggestions according to whether they help the current student succeed. It uses the student's own sampled responses, their final rewards, and teacher probabilities along those responses. It needs no additional teacher rollouts or learned value model.

## Start here

There are two implementations because their teachers and training dependencies differ:

| Folder | Teacher | Tasks |
| --- | --- | --- |
| [`external_teacher/`](external_teacher/) | A separate frozen model | Math, coding |
| [`self_distillation/`](self_distillation/) | The frozen base student given a reference solution | Math, medical QA, biology, coding |
| [`evaluation/`](evaluation/) | Standalone benchmarks, graders, and diagnostic callbacks | Math, HumanEval+, MultiPL-E, QA, token interventions |
| [`data/`](data/) | Dataset preparation | Biology, coding, QA evaluation inputs |

Each source file starts with a short description. Each training folder has its own file guide and requirements. Run commands from the repository root.

## Installation

Use **Python 3.11** and a separate environment for each training folder. These versions follow the source environments; the external trainer uses TRL's experimental `DistillationTrainer`, while self-distillation uses the older GOLD/OPSD API.

```bash
# External teacher
python3.11 -m venv .venv-external
source .venv-external/bin/activate
pip install -r external_teacher/requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation

# Self-distillation: create this in a separate shell/environment
python3.11 -m venv .venv-self
source .venv-self/bin/activate
pip install -r self_distillation/requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

The launchers target eight GPUs with BF16, LoRA, and colocated vLLM. Match the PyTorch/CUDA build to your machine. Coding additionally needs Docker. Set `WANDB_MODE=disabled` to disable training uploads, or log in to your own W&B account.

## Training

The short `run_*.sh` launchers use the **paper settings**: batch 32 on eight workers, non-thinking prompts, and the schedules below. They are the public starting points. The actual archived experiment commands are retained separately in [`self_distillation/recipes/`](self_distillation/recipes/); source settings and differences are explained in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). These two sets are labelled because several source launchers differ from the paper.

| Setting | Command | Student / teacher | Updates | Training tokens |
| --- | --- | --- | ---: | ---: |
| External math | `bash external_teacher/run_math.sh` | Qwen3-1.7B / Qwen3-4B-Instruct-2507 | 450 | 8192 |
| External coding | `bash external_teacher/run_code.sh` | Qwen3-1.7B / Qwen3-8B | 350 | 4096 |
| Self math | `bash self_distillation/run_math.sh` | Qwen3-1.7B; reference-conditioned self-teacher | 400 | 8192 |
| Self medical | `bash self_distillation/run_medical.sh` | Qwen3-4B-Instruct-2507 | 300 | 2048 |
| Self biology | `bash self_distillation/run_biology.sh` | Qwen3-4B-Instruct-2507 | 400 | 4096 |
| Self coding | `bash self_distillation/run_code.sh` | Qwen3-4B-Instruct-2507 | 400 | 4096 |

Choose a different student with `MODEL=Qwen/Qwen3-8B bash self_distillation/run_math.sh` or `MODEL=Qwen/Qwen2.5-7B-Instruct bash self_distillation/run_medical.sh`. `OUTPUT_DIR` changes the output root. Remaining command-line options can be appended to a launcher. Task preparation and the original baseline files are documented inside each folder.

### The loss, in a few steps

1. Sample a group of responses from the student and score their final answers.
2. Give each response a leave-one-out advantage: its reward minus the mean reward of the other responses in its group.
3. Combine the sampled teacher/student ratio with a dense teacher correction, clipping the ratio at `c = 4`. The original top-k/tail handling and residual term are preserved.

**All-wrong groups.** The optional extra forward KL is applied only when **every response in a group has zero reward**. A wrong response in a mixed group does not receive this extra term. It supplies an imitation signal where all leave-one-out advantages would otherwise be zero. `mu=0` disables it; the external math source used `mu=0.08`, external coding used `0`, and the retained self-math all-wrong entry point defaults to `0`. Medical, biology, and self-coding recipes retain `mu=0`.

**Why keep the residual?** Clipping caps teacher mass at `c` times the detached student probability. The leftover mass is the residual. The existing code gives that leftover mass a bounded, advantage-dependent correction: discourage it on positive-advantage responses and encourage it on negative-advantage responses. This is an implementation addition to the clipped objective described in the paper, and is retained unchanged. See the [exact expression](REPRODUCIBILITY.md#residual-term).

## Evaluation

All standalone evaluations live in [`evaluation/`](evaluation/README.md). Merge a LoRA adapter first if the evaluator expects a complete model:

```bash
python -m evaluation.merge_adapter --base_model Qwen/Qwen3-1.7B \
  --adapter_path outputs/RUN/checkpoint-400 --save_path outputs/RUN/merged

python -m evaluation.math --model_path outputs/RUN/merged --exp_name RUN \
  --dataset aime --max_tokens 16192 --prompt_style self

bash evaluation/humaneval.sh outputs/RUN/merged
```

The math prompt style is explicit: use `external` for external-teacher math and `self` for reference-conditioned math. HumanEval's short wrapper is greedy pass@1; pass@10 requires at least ten sampled completions. Training callbacks are development diagnostics and do not replace benchmark evaluation.

## Reproducibility and credits

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records paper/source differences, the residual formula, data splits, checkpoint selection, and release checks. [SOURCE_MAP.json](SOURCE_MAP.json) maps retained code to original source hashes. The cleanup excludes checkpoints, run logs, generated answers, caches, and unused experimental variants.

This implementation builds on Hugging Face TRL, OPSD (*Self-Distilled Reasoner*, [paper](https://arxiv.org/abs/2601.18734)), and VERL's SandboxFusion client. Upstream copyright and license notices are retained; see [NOTICE](NOTICE) and [licenses/](licenses/).
