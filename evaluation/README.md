# Evaluation

Run from the repository root. Install `evaluation/requirements.txt` into the appropriate inference environment. Keep the same tokenizer, prompt, sampling settings, and token budget across methods. Training-time callbacks are diagnostics; they do not replace these benchmark evaluations.

| File | Purpose |
| --- | --- |
| `merge_adapter.py` | Merge a LoRA adapter and preserve its tokenizer/chat template |
| `math.py` | Math benchmarks: four sampled completions for pass@1/2/4 plus an independent temperature-0.6 accuracy sample |
| `math_grader.py` | Original boxed-answer and symbolic equivalence grader |
| `humaneval.sh` | Original EvalPlus/vLLM greedy HumanEval and HumanEval+ route |
| `multiple.py` | Generate MultiPL-E completion files for its separate evaluator |
| `qa.py` | Medical/biology/MMLU-Pro inference with retained parsers and raw JSONL |
| `qa_graders.py` | Original medical and biology answer extraction |
| `mmlupro_helpers.py` | Pinned official MMLU-Pro five-shot formatting and answer extraction |
| `code_callback.py` | Training-time development callback for self-coding |
| `token_intervention.py` | Separate single-token causal intervention study |
| `requirements.txt` | Evaluation-specific packages |

## Math

```bash
python -m evaluation.math --model_path outputs/RUN/merged --exp_name RUN \
  --dataset aime --max_tokens 16192 --prompt_style external
```

Choose `self` for reference-conditioned math prompts. This replaces the old filename-based prompt guess. Datasets: `aime`, `amc`, `hmmt`, `math500`, `minerva`, `openthoughts`. OpenThoughts is a diagnostic split: the first 512 shuffled examples overlap the external source training split. It must not be described as clean held-out external test accuracy.

The retained source generates at temperature 0.6, top-p 0.95, top-k 20. Old source labels incorrectly called its independent accuracy sample temperature zero; the release corrects those labels without changing generation. Results go to `results/math/` unless `--output_dir` is specified.

## Coding

```bash
bash evaluation/humaneval.sh outputs/RUN/merged
python -m evaluation.multiple --model outputs/RUN/merged --lang java \
  --out_dir results/multiple/java --max_tokens 512
```

The short HumanEval wrapper is **greedy pass@1**. For sampled pass@10, use at least ten non-greedy completions in the EvalPlus version's sampled-generation mode; a greedy run cannot provide pass@10. Its command-line options are available through `evalplus.evaluate --help`.

`multiple.py` retains the source greedy default. To generate ten sampled candidates, append `--n 10 --temperature 0.8 --seed 7319`. Execute and score those JSON files with the official [MultiPL-E evaluator](https://github.com/nuprl/MultiPL-E), using its language runtime/container. Generation alone is not a benchmark score. The archived instruct comparison used HumanEval cap 768, MultiPL-E cap 512, and sampled n=10; preserve its full protocol if comparing those recorded results. LiveCodeBench is an additional benchmark and is not replaced by the five-test training reward.

## Medical and biology

Prepare the inputs using `data/README.md`, then:

```bash
python -m evaluation.qa --model outputs/RUN/merged --task medical \
  --data data/medical/medqa_test.jsonl --output results/medical/medqa.jsonl
python -m evaluation.qa --model outputs/RUN/merged --task biology \
  --data data/biology/source50.jsonl --output results/biology/source50.jsonl
python -m evaluation.qa --model outputs/RUN/merged --task mmlupro \
  --data data/mmlupro717.jsonl --output results/biology/mmlupro.jsonl
```

Medical uses temperature 0.6, top-p 0.95, top-k 20, seed `42 + row index`, and cap 8192, matching the archived Qwen3 common evaluation. Biology/MMLU-Pro use greedy decoding, seed 7319, and cap 4096. MMLU-Pro stops at `Question:`. The selected Qwen3 AOPD biology condition uses `--max_tokens 8192`; keep it distinct from 4096-token conditions. Outputs retain raw text, token IDs, finish reasons, prompt hashes, denominator, invalid count, and truncation count. Invalid answers count wrong.

This compact evaluator preserves prompts, parsers, and decoding rules. It is not a replacement for the archived experiment's model/adapter hash audit, training-overlap checks, uncertainty analysis, or full runtime manifest.

## Single-token interventions

```bash
python -m evaluation.token_intervention --help
```

The retained script exposes fixed baseline/selection/intervention/summary stages and random versus highest-KL token selection. Use a fresh output directory and pinned model/data revisions. The original external-teacher defaults do not by themselves reproduce the separate reference-conditioned self-teacher study; that study's teacher-context adaptation and replay checks must be retained in its own protocol. No intervention was run during code packaging.
