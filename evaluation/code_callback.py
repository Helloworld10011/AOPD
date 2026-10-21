# Training-time TACO callback; diagnostic accuracy is separate from full benchmark evaluation.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
"""Shared deterministic development decoding for the coding task adapter."""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from transformers import TrainerCallback
from vllm import SamplingParams

from self_distillation.code_reward import POOL, score_details


class VLLMTacoEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, examples, eval_steps=25, max_new_tokens=4096):
        self.answers = list(examples['answer'])
        self.uids = list(examples['uid'])
        self.prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        for x in examples['prompt']]
        assert self.prompts and len(self.prompts) == len(self.answers) == len(self.uids)
        self.budget = max_new_tokens
        self.params = [SamplingParams(temperature=.6, top_p=.95, top_k=20, min_p=0., max_tokens=max_new_tokens,
                                      n=1, seed=int(hashlib.sha256(('7319:' + uid).encode()).hexdigest()[:8], 16) % (2 ** 31))
                       for uid in self.uids]
        self.eval_steps = eval_steps
        self.trainer = None

    def _run_eval(self, state):
        trainer = self.trainer
        if getattr(trainer, '_last_vllm_sync_step', -1) != state.global_step:
            trainer._move_model_to_vllm()
            trainer._last_vllm_sync_step = state.global_step
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        indices = list(range(rank, len(self.prompts), world))
        outs = trainer.vllm_engine.generate([self.prompts[i] for i in indices],
                                           [self.params[i] for i in indices], use_tqdm=False)
        assert len(outs) == len(indices)
        texts = [o.outputs[0].text for o in outs]
        scores = list(POOL.map(score_details, texts, [self.answers[i] for i in indices]))
        directory = Path(trainer.args.output_dir) / 'taco_screen'
        directory.mkdir(exist_ok=True)
        rows = []
        for i, out, score in zip(indices, outs, scores):
            result = out.outputs[0]
            rows.append({'uid': self.uids[i], 'step': state.global_step, 'seed': 7319,
                         'completion': result.text, 'tokens': len(result.token_ids), 'finish_reason': result.finish_reason,
                         'tests_sha256': hashlib.sha256(self.answers[i].encode()).hexdigest(), **score})
        path = directory / f'generations_step_{state.global_step:06d}_rank{rank}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        stats = torch.tensor([sum(x['all_pass'] for x in rows), sum(x['fraction'] for x in rows), len(rows),
                              sum(x['tokens'] for x in rows), sum(x['finish_reason'] == 'length' for x in rows),
                              sum(x['missing_code'] for x in rows)], device=trainer.accelerator.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(stats)
        if state.is_world_process_zero:
            n = stats[2].item()
            metrics = {'train/global_step': state.global_step, 'eval_correct_reward': stats[0].item() / n,
                       'eval_test_fraction': stats[1].item() / n, 'eval_mean_tokens': stats[3].item() / n,
                       'eval_truncation_fraction': stats[4].item() / n, 'eval_missing_code_fraction': stats[5].item() / n,
                       'eval_count': int(n)}
            (directory / f'metrics_step_{state.global_step:06d}.json').write_text(json.dumps(metrics, indent=2) + '\n')
            print(metrics, flush=True)
            if wandb.run is not None:
                wandb.log(metrics)

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_eval(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._run_eval(state)
