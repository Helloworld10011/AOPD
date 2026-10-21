# Data preparation

Run these commands from the repository root in the self-distillation environment. Prepared datasets stay outside Git. The scripts fail if their output already exists, preserving a frozen split.

| Command | Output | Source / split |
| --- | --- | --- |
| `python -m data.prepare_biology` | `data/biology/` | Pinned SciKnowEval biology: 322 train, 128 development, 50 test |
| `python -m data.prepare_code --output data/code --direct_code` | `data/code/` | Pinned DeepCoder TACO: 5409 train, 256 development, 512 reserved |
| `python -m data.prepare_medical` | `data/medical/` | Pinned MedQA English four-option test (1273) and MedMCQA validation (4183) |
| `python -m data.prepare_mmlupro` | `data/mmlupro717.jsonl` | Pinned MMLU-Pro biology with official five-shot content |

Self math and medical training load their released Hugging Face datasets in the entry points. External coding uses its distinct original preparation: `python -m external_teacher.prepare_code`, with SandboxFusion running first. See the training folder READMEs.

Biology verifies original source hashes and the exact archived training/development JSONL hashes. TACO keeps the historical Qwen3/OLMo tokenizer context filter; OLMo here defines data eligibility only. `--direct_code` changes the instruction after filtering, preserving the source row membership/order and tests. Arrow container hashes can vary with the datasets version; compare rows, prompts, tests, and pinned source hashes as well.

Medical evaluation prompts preserve the released wrapper and A–D choice order. Prepared benchmark inputs contain no teacher reference. The archived Qwen3 evaluation found no exact normalized-question overlap with training/callback data; this preparation script does not repeat that costly full-training overlap audit and makes no new contamination claim.
