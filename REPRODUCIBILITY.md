# Reproducing the experiments

## Paper defaults and recorded settings

The author requested paper settings for the main launchers, with source settings retained separately. The `run_*.sh` files follow the paper's batch 32, non-thinking setting, update counts, and training caps. This does **not** establish that these reconstructed defaults reproduce every published number. No new training was performed for this release.

The retained mathematical loss, reward functions, reference templates, split logic, and task-specific parsers stay close to their source versions. Recorded main instruct commands live in `self_distillation/recipes/`, with source configuration hashes and data hashes in `recorded_recipes.json`. `selected_checkpoints.json` records checkpoint, seed, learning rate, inference cap, and selection status separately from the training endpoint.

| Setting | Source setting | Public starting point |
| --- | --- | --- |
| External math learning rate | `1e-5` | `2e-5`, as stated in the paper |
| External math/coding duration | 1 / 2 epochs | 450 / 350 optimizer updates |
| External batch on 8 ranks | 2 per rank × accumulation 8 = 128 rows | 1 × 4 × 8 = 32 rows |
| Self-math thinking | Both flags `True` in the source launcher | Both flags `False`, as stated in the paper |
| Self-math training cap | 16192 in the source launcher | 8192, as stated in the paper |
| Self-math batch on 8 ranks | 2 × 8 × 8 = 128 rows | 1 × 4 × 8 = 32 rows |
| Instruct AOPD source commands | Commonly batch 128; model/task-specific LR and schedules | Paper batch 32; retained task-specific LR/schedule |
| External extra KL | Per wrong response in math; unconditional path disabled by `mu=0` in coding | Entire all-wrong group only, per author request |

The external source configuration is in `external_teacher/source_settings.json`. Instruct recipes keep the actual different AOPD/OPSD batches, learning rates, schedulers, and generation caps. GPU count changes the batch size; do not change it silently when comparing methods. In the external implementation, a full reward group must fit in a rank's accumulated rollout buffer; incomplete groups raise an error.

`MODEL` overrides in the short launchers use the model's resolved current revision. Recorded instruct recipes pin the original revision. Pin model/data revisions before starting a new scientific comparison. Not every historical external-teacher revision or original medical protocol is recoverable from these three source trees; none has been invented.

## Residual term

At a student-generated history, write `S` for the current student distribution, `T` for the frozen teacher, `c=4`, and `A` for the response's leave-one-out advantage. With stop-gradient denoted by `sg`:

```text
T_tilde = sg(min(T, c * sg(S)))
K       = sg(max(T - T_tilde, 0))
q       = sigmoid(log(c) + log(S) - sg(log(T)))
```

`T_tilde` is not renormalized. The residual loss at that history is:

```text
A > 0:  |A| * sum_a K(a) * 4*q(a)
A < 0:  |A| * sum_a K(a) * 4*(1-q(a))
A = 0:  0
```

The code averages this over valid completion tokens and adds it to the clipped AOPD core. Minimization discourages leftover teacher mass for positive advantages and encourages it for negative advantages. The factors are bounded between 0 and 4; `K` is detached. The logged `atd/kappa_mass` is the amount of teacher mass removed by clipping, `sum_a K(a)`.

This is the existing bounded sigmoid correction, **not** a literal `-K log S` or `-K log(1-S)` penalty. Old comments describing those penalties were inaccurate and are corrected in the external trainer. The residual is an additional implementation choice; it is not an unbiased recovery of the clipped-away part of the paper's objective. On a top-k loss, these expressions use the retained teacher-selected support and its tail bucket.

## Extra KL on all-wrong groups

For reward group `g`, the gate is `all(R <= 0)`; task rewards here are nonnegative. The extra term is `mu * mean_response[gate(g) * mean_valid_token KL(T || S)]`. The mean includes ungated responses as zeros. Clipping/residual calculations are separate from this forward KL.

Mixed groups keep the original advantage-weighted loss, including their wrong responses. All-correct groups receive no extra KL. In an all-zero group, the AOPD core and residual vanish because `A=0`; only the optional KL remains. `mu=0` recovers the retained original-objective loss. Nonzero unconditional KL is not offered by the released task recipes.

## Data and evaluation

External math preserves its source split: shuffle seed 42, first half of OpenThoughts, first 256 examples for development. Self math preserves the maintained all-wrong entry point's `train_test_split(test_size=512, seed=42)`. These are different splits. The external OpenThoughts evaluator's first 512 shuffled examples are a diagnostic mixture relative to external training, not a clean held-out external test set.

External coding uses the original verified `likaixin/TACO-verified` stdin preparation. Self coding uses the pinned DeepCoder TACO source, fixed shared-context filtering, 256 development examples, 512 reserved examples, and 5409 training examples. The direct-code condition changes the final instruction after selection. The five-test reward and development callback are not full-suite benchmark pass@1. The paper's prose describes partial coding reward, while the retained external coding source calls SandboxFusion with `continuous=False` (all-five-pass); this difference is explicit and unchanged.

Biology preserves the released 450/50 SciKnowEval source split: 128 hash-selected development examples and 322 training examples. Its self-teacher sees only the gold answer as a privileged reference, not a generated biological rationale. Benchmark evaluation additionally uses 717 MMLU-Pro biology questions with the official five-shot content/parser inside a native non-thinking chat. The selected Qwen3 checkpoint uses 8192 inference tokens; its original 4096 results and the 4096 OPSD condition must remain distinct.

Medical training uses `ytz20/sys_medmcqa_train`, with the released `teacher_response` reference. Its separately released callback evaluation artifact has a Hugging Face split also called `train`. Full MedQA and MedMCQA benchmark protocols are separate. Original author-owned Qwen2.5 medical scores/protocol were pending in the archived comparison; this release does not claim those were independently recovered or verified.

Archived selected instruct checkpoints are descriptive; several were selected after test exposure. Do not present them as an unbiased method comparison, combine per-column maxima, attach old endpoint intervals to selected checkpoints, or replace benchmark scores with callback metrics. The code release makes no new score or completion claim.

## Scope of validation

Release validation checks Python/shell syntax, portable imports and launch arguments, source-equivalent loss values and gradients on small CPU tensors, mixed/all-wrong group gating, and local data reconstruction where source inputs are available. No GPU training or benchmark generation is started during packaging. Exact receipts are in `VALIDATION.json`.
