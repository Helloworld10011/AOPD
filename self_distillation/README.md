# AOPD with a reference-conditioned self-teacher

The student sees the problem. The teacher sees the same problem plus a reference solution, using the frozen base model with LoRA adapters disabled. Task-specific reference templates and reward parsers are preserved. This implementation uses TRL GOLD/OPSD and has its **own requirements**, separate from external-teacher training.

The short launchers use the paper's non-thinking setting and batch 32 on eight workers. Math/code use learning rate `2e-5`; biology keeps its source model-specific setting (the Qwen3 launcher uses `1e-5`). Recorded actual commands, including their different batches, are in `recipes/`.

```bash
bash self_distillation/run_math.sh
MODEL=Qwen/Qwen3-8B bash self_distillation/run_math.sh
bash self_distillation/run_medical.sh
MODEL=Qwen/Qwen2.5-7B-Instruct bash self_distillation/run_medical.sh

python -m data.prepare_biology
bash self_distillation/run_biology.sh

python -m data.prepare_code --output data/code --direct_code
docker pull ghcr.io/nuprl/multipl-e-evaluation@sha256:2cb4fcd74d89d149f7a3feb177f7d8169cef390a7ea7951c9b84e7ff59ea26cc
bash self_distillation/run_code.sh
```

Medical data is loaded from the released `ytz20/sys_medmcqa_train` and `ytz20/sys_medmcqa_test` artifacts. Biology's teacher reference is the answer label only. Self-coding uses the pinned MultiPL-E Docker image for isolated Python execution; it does not use the external-teacher SandboxFusion service. `AOPD_CODE_IMAGE` can override the image, but changes the execution environment and must be reported.

The math entry point is the retained **all-wrong** version. `MU=0.2 bash self_distillation/run_math.sh` adds KL only to all-wrong groups; its default `MU=0` retains the source entry point's coefficient. Medical, biology, coding, and supplementary ProofWriter use `mu=0`.

## File guide

| File | Purpose |
| --- | --- |
| `train_math.py` | OpenThoughts AOPD using the maintained all-wrong math variant |
| `train_medical.py` | Medical QA training with released teacher responses |
| `train_biology.py` | Biology training with the source answer parser and privileged gold label |
| `train_code.py` | TACO direct-code training with five-test fractional reward |
| `aopd_loss.py` | Original self-distillation clipped loss, residual, and kappa diagnostics |
| `buffered_trainer.py` | Keep whole reward groups aligned across accumulated microbatches |
| `distributed_trainer.py` | Cross-rank prompt IDs, grouping checks, and leave-one-out advantages |
| `all_wrong_trainer.py` | Optional forward KL only when every response in a group fails |
| `opsd_trainer.py` | Retained upstream OPSD/GOLD training and generation infrastructure |
| `data_collator.py` | Original student/teacher reference prompts and token alignment |
| `code_reward.py` | Execute candidate Python in Docker; return fractional reward on five tests |
| `baseline_{math,medical,biology,code}.py` | Original OPSD baselines |
| `train_proofwriter.py`, `baseline_proofwriter.py` | Supplementary ProofWriter entry points |
| `run_{math,medical,biology,code}.sh` | Short launchers using paper dimensions |
| `accelerate.yaml` | Original eight-GPU ZeRO-2 configuration |
| `requirements.txt` | Separate self-distillation environment |
| `recipes/` | Actual archived main instruct commands and selected-checkpoint metadata |

Use `python -m self_distillation.train_medical --help` (or another entry point) for its arguments. The retained class and metric names sometimes say `ATD`; these are the historical code names for AOPD.

## Recorded main instruct recipes

`recipes/recorded_recipes.json` preserves arguments, source-config hashes, model revisions, and data hashes. Its eight shell scripts cover AOPD/OPSD coding and biology for Qwen2.5-7B-Instruct and Qwen3-4B-Instruct-2507. Unlike the short launchers, these retain the actual source batches and method-specific decoding/schedules.

`recipes/selected_checkpoints.json` distinguishes the reported selected checkpoint from the trained endpoint. In particular, Qwen3 biology trained to 400 but its selected original-objective view uses saved step 50 with an 8192-token evaluation cap. This is post-hoc selection; retain endpoint/negative results and disclose the extra inference budget.

The original Qwen2.5 medical result protocol remains author-owned. The runnable medical source is retained; no missing historical scores or exact original command are invented. Supplementary ProofWriter accepts a prepared `train_data/` and `eval_data/` directory via `--science_data_dir`; its schema is the same as biology, with the original task prompts and gold answers. It is not a main-table task in the supplied paper.
