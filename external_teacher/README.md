# AOPD with an external teacher

The student generates responses; a separate frozen teacher scores next-token distributions on those same histories. Math uses Qwen3-4B-Instruct-2507 as teacher; coding uses Qwen3-8B. Both use the **same AOPD trainer**, including the original clipped core and bounded residual.

Use this folder's requirements in a separate Python 3.11 environment; see the root README. The short launchers use paper defaults: learning rate `2e-5`, batch 32 on eight GPUs, group size 4, temperature 0.8, top-p 1, no sampling top-k pruning, and non-thinking prompts. Their loss uses teacher top-64 plus a tail bucket; sampling top-k and loss top-k are different settings.

```bash
bash external_teacher/run_math.sh
# Example override: source math learning rate and source batch dimensions
bash external_teacher/run_math.sh --learning_rate 1e-5 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8
```

For coding, start the original SandboxFusion service in a separate terminal, prepare data once, then train:

```bash
bash external_teacher/sandbox.sh
# In the training environment, from the repository root:
python -m external_teacher.prepare_code
bash external_teacher/run_code.sh
```

Preparation keeps stdin problems with at least five tests, verifies a released reference solution, and saves `taco_stdin_clean/` in this folder. Training filters prompts to 2048 tokens, shuffles with seed 42, reserves 256 development examples, and uses up to 5600 training examples. The source reward requires all five chosen tests to pass (`continuous=False`); the paper's partial-reward wording differs.

`MU` controls the optional all-wrong-group KL: math defaults to 0.08 from its source; coding defaults to 0. Nonzero values apply only when all four responses fail. `MU=0 bash external_teacher/run_math.sh` disables this extra KL. A complete reward group must fit in each rank's accumulated rollout buffer.

## File guide

| File | Purpose |
| --- | --- |
| `aopd_trainer.py` | Rewards, leave-one-out advantages, clipped AOPD loss, residual, and all-wrong gate |
| `train_math.py` | OpenThoughts math data, Qwen teacher, and training entry point |
| `train_code.py` | TACO data, SandboxFusion reward, and training entry point |
| `prepare_code.py` | Original verified stdin-only TACO preparation |
| `run_math.sh`, `run_code.sh` | Short paper-default launchers; append CLI options to override |
| `sandbox.sh` | Start the original SandboxFusion image on localhost:8080 |
| `sandbox_fusion/` | Only the two required VERL reward-client files |
| `zero2.json` | Original ZeRO-2 configuration, included locally instead of an external absolute path |
| `requirements.txt` | Separate external-teacher environment |
| `source_settings.json` | Original source hyperparameters before paper-default changes |
| `baseline_math_opd.py`, `baseline_math_grpo.py` | Retained math OPD and GRPO entry points |
| `baseline_code_opd.py`, `baseline_code_rkl.py`, `baseline_code_grpo.py` | Retained coding FKL, RKL, and GRPO entry points |

Baseline entry points retain their own source hyperparameters. Run them as modules, e.g. `accelerate launch --multi_gpu --num_processes 8 --module external_teacher.baseline_code_opd --model_path Qwen/Qwen3-1.7B --exp opd_code --output_dir_base outputs`. They do not inherit the AOPD paper-default changes. Coding baselines use the same prepared data and SandboxFusion service.
