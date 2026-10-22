# One self-distillation AOPD trainer: grouped rollouts, clipped loss, residual, and all-wrong KL.
import math
from collections import defaultdict
from functools import partial
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers.trainer_utils import seed_worker
from trl.trainer.utils import RepeatSampler, pad
from .opsd import OPSDTrainer
from .tasks import AOPDDataCollator

def _add_tail_bucket(log_probs):
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True).clamp(max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum))
    return torch.cat([log_probs, tail], dim=-1)

def _truncated_fkl_chunk(student_logits, teacher_logits, temperature, top_k, add_tail, w_clip):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if top_k and top_k > 0 and student_logits.size(-1) > top_k:
        s_full = F.log_softmax(student_logits, dim=-1)
        t_full = F.log_softmax(teacher_logits, dim=-1)
        support = teacher_logits.topk(top_k, dim=-1).indices
        s_logp = s_full.gather(-1, support)
        t_logp = t_full.gather(-1, support)

        if add_tail:
            s_logp = _add_tail_bucket(s_logp)
            t_logp = _add_tail_bucket(t_logp)
        else:
            s_logp = s_logp - torch.logsumexp(s_logp, dim=-1, keepdim=True)
            t_logp = t_logp - torch.logsumexp(t_logp, dim=-1, keepdim=True)
    else:
        s_logp = F.log_softmax(student_logits, dim=-1)
        t_logp = F.log_softmax(teacher_logits, dim=-1)

    trunc_t_logp = torch.minimum(t_logp, math.log(w_clip) + s_logp.detach()).detach()
    trunc_t_prob = trunc_t_logp.exp()
    fkl_elem = torch.nan_to_num(trunc_t_prob * (trunc_t_logp - s_logp), nan=0.0)

    with torch.no_grad():
        t_prob = t_logp.exp()
        residual_prob = (t_prob - trunc_t_prob).clamp_min(0)

    ratio_logit = math.log(w_clip) + s_logp - t_logp.detach()
    ratio_prob = torch.sigmoid(ratio_logit.float()).to(s_logp.dtype)
    residual_pos_tok = (residual_prob * (4.0 * ratio_prob)).sum(-1)
    residual_neg_tok = (residual_prob * (4.0 * (1.0 - ratio_prob))).sum(-1)
    residual_grad_tok = (residual_prob * (4.0 * ratio_prob * (1.0 - ratio_prob))).sum(-1)
    kappa_tok = residual_prob.sum(-1)

    return fkl_elem, residual_pos_tok, residual_neg_tok, residual_grad_tok, kappa_tok

def truncated_fkl_terms(student_logits, teacher_logits, temperature=1.0, top_k=0, add_tail=True, w_clip=4.0, chunk_size=256):
    outputs = [[] for _ in range(5)]
    seq_len = student_logits.size(1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        s_chunk = student_logits[:, start:end, :]
        t_chunk = teacher_logits[:, start:end, :]

        if s_chunk.requires_grad:
            chunk_outputs = checkpoint(lambda s, t: _truncated_fkl_chunk(s, t, temperature, top_k, add_tail, w_clip), s_chunk, t_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            chunk_outputs = _truncated_fkl_chunk(s_chunk, t_chunk, temperature, top_k, add_tail, w_clip)

        for i, out in enumerate(chunk_outputs):
            outputs[i].append(out)

    return tuple(torch.cat(parts, dim=1) for parts in outputs)

def _sampled_logps_chunk(student_logits, teacher_logits, token_ids, temperature):
    s_logps = F.log_softmax(student_logits / temperature, dim=-1)
    with torch.no_grad():
        t_logps = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = s_logps.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    t_logp = t_logps.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return s_logp, t_logp

def sampled_logps_chunked(student_logits, teacher_logits, token_ids, temperature=1.0, chunk_size=256):
    student_parts, teacher_parts = [], []
    seq_len = student_logits.size(1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        s_chunk = student_logits[:, start:end, :]
        t_chunk = teacher_logits[:, start:end, :]
        ids_chunk = token_ids[:, start:end]

        if s_chunk.requires_grad:
            s_logp, t_logp = checkpoint(lambda s, t, ids: _sampled_logps_chunk(s, t, ids, temperature), s_chunk, t_chunk, ids_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            s_logp, t_logp = _sampled_logps_chunk(s_chunk, t_chunk, ids_chunk, temperature)

        student_parts.append(s_logp)
        teacher_parts.append(t_logp)

    return torch.cat(student_parts, dim=1), torch.cat(teacher_parts, dim=1)

def _forward_kl_chunk(student_logits, teacher_logits, temperature, top_k, add_tail):
    student_logits = student_logits / temperature
    s_logp = F.log_softmax(student_logits, dim=-1)
    with torch.no_grad():
        teacher_logits = teacher_logits / temperature
        t_logp = F.log_softmax(teacher_logits, dim=-1)

    if top_k and 0 < top_k < student_logits.size(-1):
        support = teacher_logits.topk(top_k, dim=-1).indices
        s_logp = s_logp.gather(-1, support)
        t_logp = t_logp.gather(-1, support)
        if add_tail:
            s_logp = _add_tail_bucket(s_logp)
            t_logp = _add_tail_bucket(t_logp)
        else:
            s_logp = s_logp - torch.logsumexp(s_logp, dim=-1, keepdim=True)
            t_logp = t_logp - torch.logsumexp(t_logp, dim=-1, keepdim=True)

    return F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(-1)

class AOPDTrainer(OPSDTrainer):

    def __init__(self, *args, reward_func=None, atd_num_generations=4, w_clip=4.0, mu=0.0, add_tail=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.reward_func = reward_func
        self.atd_num_generations = atd_num_generations
        self.w_clip = w_clip
        if not math.isfinite(mu) or mu < 0:
            raise ValueError('mu must be finite and nonnegative.')
        self.mu = mu
        self._current_all_wrong = None
        self.add_tail = add_tail
        self._current_advantages = None
        self._atd_metrics = defaultdict(list)
        if reward_func is None:
            raise ValueError('ATD requires reward_func.')
        if atd_num_generations < 2:
            raise ValueError('ATD requires atd_num_generations >= 2.')
        if not self.use_vllm:
            raise ValueError('This ATD implementation currently requires use_vllm=True.')
        if self.beta != 0:
            raise ValueError('ATD requires beta=0.')
        if self.use_thinking_machines_loss:
            raise ValueError('ATD is incompatible with use_thinking_machines_loss.')
        if self.jsd_token_clip is not None:
            raise ValueError('Set --jsd_token_clip 0 for ATD. ATD uses w_clip instead.')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._current_advantages = inputs.pop('advantages', None)
        self._current_all_wrong = inputs.pop('atd_all_wrong', None)
        if self._current_advantages is None:
            raise RuntimeError('ATD advantages were not attached to the batch.')
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            self._current_advantages = None
            self._current_all_wrong = None

    def generalized_jsd_loss(self, student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction='batchmean', logits_are_probs=False, top_k=None, token_clip=None):
        if self._current_advantages is None:
            return OPSDTrainer.generalized_jsd_loss(student_logits, teacher_logits, labels=labels, beta=beta, temperature=temperature, reduction=reduction, logits_are_probs=logits_are_probs, top_k=top_k, token_clip=token_clip)
        if labels is None:
            raise ValueError('ATD requires completion labels.')
        A = self._current_advantages.to(student_logits.device)
        if A.numel() != student_logits.shape[0]:
            raise ValueError(f'advantages={A.numel()} but batch={student_logits.shape[0]}')
        x_elem, residual_pos_tok, residual_neg_tok, residual_grad_tok, kappa_tok = truncated_fkl_terms(student_logits, teacher_logits, temperature=temperature, top_k=top_k or 0, add_tail=self.add_tail, w_clip=self.w_clip)
        x_tok = x_elem.sum(-1)
        mask = labels != -100
        sampled_ids = labels.clamp_min(0)
        s_logp, t_logp = sampled_logps_chunked(student_logits, teacher_logits, sampled_ids, temperature=temperature, chunk_size=256)
        with torch.no_grad():
            w = torch.exp(t_logp - s_logp).clamp(max=self.w_clip)
        core_tok = w * -s_logp - x_tok
        core_tok = torch.where(mask, core_tok, torch.zeros_like(core_tok))
        denom = mask.sum(-1).clamp_min(1)
        core_seq = core_tok.sum(-1) / denom
        residual_pos_tok = torch.where(mask, residual_pos_tok, torch.zeros_like(residual_pos_tok))
        residual_neg_tok = torch.where(mask, residual_neg_tok, torch.zeros_like(residual_neg_tok))
        residual_pos_seq = residual_pos_tok.sum(-1) / denom
        residual_neg_seq = residual_neg_tok.sum(-1) / denom
        residual_seq = torch.where(A > 0, residual_pos_seq, residual_neg_seq)
        core_loss = A * core_seq
        residual_loss = A.abs() * residual_seq
        loss = (core_loss + residual_loss).mean()
        with torch.no_grad():
            residual_grad_tok = torch.where(mask, residual_grad_tok, torch.zeros_like(residual_grad_tok))
            residual_grad_seq = residual_grad_tok.sum(-1) / denom
            pos = A > 0
            neg = A < 0
            contrib = core_loss + residual_loss
            self._atd_metrics['atd/pos_loss'].append(contrib[pos].mean().item() if pos.any() else 0.0)
            self._atd_metrics['atd/neg_loss'].append(contrib[neg].mean().item() if neg.any() else 0.0)
            self._atd_metrics['atd/residual_pos'].append(residual_loss[pos].mean().item() if pos.any() else 0.0)
            self._atd_metrics['atd/residual_neg'].append(residual_loss[neg].mean().item() if neg.any() else 0.0)
            self._atd_metrics['atd/kappa_mass'].append(((kappa_tok * mask).sum() / mask.sum().clamp_min(1)).item())
            self._atd_metrics['atd/frac_w_clipped'].append((((w >= self.w_clip) & mask).float().sum() / mask.float().sum().clamp_min(1)).item())
            self._atd_metrics['atd/residual_grad_pos'].append((A[pos] * residual_grad_seq[pos]).mean().item() if pos.any() else 0.0)
            self._atd_metrics['atd/residual_grad_neg'].append((-A[neg] * residual_grad_seq[neg]).mean().item() if neg.any() else 0.0)

        if self.mu:
            gate = self._current_all_wrong
            if gate is None or gate.shape != (student_logits.size(0),):
                raise RuntimeError("Missing all-wrong group labels.")
            parts = []
            for start in range(0, student_logits.size(1), 256):
                end = min(start + 256, student_logits.size(1))
                s_chunk, t_chunk = student_logits[:, start:end], teacher_logits[:, start:end]
                if torch.is_grad_enabled() and s_chunk.requires_grad:
                    part = checkpoint(_forward_kl_chunk, s_chunk, t_chunk, temperature, top_k, self.add_tail, use_reentrant=False, preserve_rng_state=False)
                else:
                    part = _forward_kl_chunk(s_chunk, t_chunk, temperature, top_k, self.add_tail)
                parts.append(part)
            kl_tok = torch.cat(parts, dim=1)
            kl_seq = torch.where(mask, kl_tok, torch.zeros_like(kl_tok)).sum(-1) / denom
            loss = loss + self.mu * (gate.to(kl_seq.device, kl_seq.dtype) * kl_seq).mean()

        return loss

    def log(self, logs, start_time=None):
        if self._atd_metrics:
            keys = sorted(self._atd_metrics)
            values = torch.tensor([sum(self._atd_metrics[k]) / len(self._atd_metrics[k]) for k in keys], dtype=torch.float64, device=self.accelerator.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= dist.get_world_size()
            logs = logs.copy()
            logs.update({k: v.item() for k, v in zip(keys, values)})
            self._atd_metrics.clear()
        return super().log(logs, start_time)

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        steps = self.args.steps_per_generation
        if self.vllm_mode != 'colocate' or self.vllm_tensor_parallel_size != 1:
            raise ValueError('This adapter requires colocated vLLM with tensor parallel size 1.')
        if self.vllm_enable_sleep_mode or self.reason_first:
            raise ValueError('Keep vLLM sleep mode and reason_first disabled.')
        if steps != self.args.gradient_accumulation_steps or steps < 1:
            raise ValueError('Set steps_per_generation equal to gradient_accumulation_steps.')
        if not self.args.dataloader_drop_last:
            raise ValueError('Set dataloader_drop_last=True.')
        if self.accelerator.split_batches or self.accelerator.dispatch_batches:
            raise ValueError('Use split_batches=False and dispatch_batches=False (or None).')
        if self.args.packing or self.args.group_by_length:
            raise ValueError('Keep packing=False and group_by_length=False.')
        if not isinstance(self.data_collator, AOPDDataCollator):
            raise ValueError('Use AOPDDataCollator with this trainer.')
        global_rollouts = self.args.per_device_train_batch_size * steps * self.accelerator.num_processes
        if global_rollouts % self.atd_num_generations:
            raise ValueError('The global generation batch must contain complete advantage groups.')
        prompts_per_batch = global_rollouts // self.atd_num_generations
        if len(train_dataset) < prompts_per_batch:
            raise ValueError('The dataset is smaller than one global generation batch.')
        return RepeatSampler(data_source=train_dataset, mini_repeat_count=self.atd_num_generations, batch_size=prompts_per_batch, repeat_count=steps, shuffle=True, seed=self.args.seed)

    def get_train_dataloader(self):
        if not isinstance(self.train_dataset, Dataset):
            raise ValueError('This adapter expects a map-style Hugging Face Dataset.')
        dataset = self._remove_unused_columns(self.train_dataset, description='training')
        kwargs = dict(batch_size=self._train_batch_size * self.args.steps_per_generation, collate_fn=self.data_collator, sampler=self._get_train_sampler(dataset), drop_last=True, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)
        if self.args.dataloader_num_workers > 0:
            kwargs.update(persistent_workers=self.args.dataloader_persistent_workers, prefetch_factor=self.args.dataloader_prefetch_factor, worker_init_fn=partial(seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index))
        return self.accelerator.prepare(DataLoader(dataset, **kwargs))

    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        if pad_token_id is None:
            raise ValueError('A padding token is required.')
        features = inputs.get('atd_features')
        micro_size = self.args.per_device_train_batch_size
        steps = self.args.steps_per_generation
        local_size = micro_size * steps
        if features is None or len(features) != local_size:
            raise RuntimeError(f'Expected {local_size} local generation rows from the buffered collator.')
        row_ids = tuple((feature['atd_group_id'] for feature in features))
        if not getattr(self, '_atd_rollout_buffer', None):
            self._atd_buffer_ids = row_ids
            self._atd_buffer_step = self.state.global_step
            self._atd_rollout_buffer = self._build_atd_rollout_buffer(features, generation_config, pad_token_id)
        elif row_ids != self._atd_buffer_ids or self.state.global_step != self._atd_buffer_step:
            raise RuntimeError('Generation-buffer alignment changed before the accumulated update completed.')
        micro_inputs, result = self._atd_rollout_buffer.pop(0)
        inputs.clear()
        inputs.update(micro_inputs)
        return result

    def _build_atd_rollout_buffer(self, features, generation_config, pad_token_id):
        device = self.accelerator.device
        micro_size = self.args.per_device_train_batch_size
        local_size = len(features)
        group_size = self.atd_num_generations
        local_count = torch.tensor([local_size], device=device, dtype=torch.long)
        counts = self.accelerator.gather(local_count)
        if not (counts == local_size).all().item():
            raise RuntimeError(f'Unequal generation batch sizes across ranks: {counts.tolist()}')
        local_ids = torch.tensor([feature['atd_group_id'] for feature in features], device=device, dtype=torch.long)
        all_ids = self.accelerator.gather(local_ids)
        grouped_ids = all_ids.view(-1, group_size)
        if not (grouped_ids == grouped_ids[:, :1]).all().item():
            raise RuntimeError('Gathered rewards would combine different dataset rows in one group.')
        if grouped_ids[:, 0].unique().numel() != grouped_ids.size(0):
            raise RuntimeError('A dataset row appears in multiple groups in one generation batch.')
        microbatches = [self.data_collator.collate_microbatch(features[start:start + micro_size]) for start in range(0, local_size, micro_size)]
        prompt_rows = []
        for batch in microbatches:
            for ids, mask in zip(batch['student_prompts'], batch['student_prompt_attention_mask']):
                prompt_rows.append(ids[mask.bool()].to(device))
        prompt_width = max((row.numel() for row in prompt_rows))
        if self.args.max_length and prompt_width > self.args.max_length - generation_config.max_new_tokens:
            raise ValueError('A student prompt exceeds the reserved prompt budget; do not silently change its truncation.')
        generation_inputs = {'student_prompts': pad(prompt_rows, padding_value=pad_token_id, padding_side=self.processing_class.padding_side), 'student_prompt_length': prompt_width}
        result = OPSDTrainer._generate_on_policy_outputs_vllm(self, generation_inputs, generation_config, pad_token_id)
        generated_ids, generated_mask = result[:2]
        completion_ids = generated_ids[:, prompt_width:]
        completion_mask = generated_mask[:, prompt_width:]
        if completion_ids.size(1) == 0:
            raise RuntimeError('vLLM returned an empty completion batch.')
        texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        answers = [feature['Answer'] for feature in features]
        rewards = torch.as_tensor(self.reward_func(None, texts, answers), dtype=torch.float32, device=device)
        if rewards.shape != (local_size,):
            raise RuntimeError('The reward function must return one scalar per rollout.')
        all_rewards = self.accelerator.gather(rewards)
        if not torch.isfinite(all_rewards).all().item():
            raise RuntimeError('Non-finite reward encountered.')
        grouped_rewards = all_rewards.view(-1, group_size)
        grouped_advantages = grouped_rewards - (grouped_rewards.sum(dim=1, keepdim=True) - grouped_rewards) / (group_size - 1)
        all_advantages = grouped_advantages.reshape(-1)
        start = self.accelerator.process_index * local_size
        advantages = all_advantages[start:start + local_size]
        group_all_wrong = ~(grouped_rewards > 0).any(dim=1)
        all_wrong = group_all_wrong.repeat_interleave(group_size)[start:start + local_size]
        buffer = []
        for index, batch in enumerate(microbatches):
            start = index * micro_size
            stop = start + micro_size
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch.pop('ground_truth')
            batch.pop('atd_group_ids')
            batch['advantages'] = advantages[start:stop].contiguous()
            batch['atd_all_wrong'] = all_wrong[start:stop].contiguous()
            mask = completion_mask[start:stop]
            valid_columns = mask.any(dim=0).nonzero(as_tuple=False).flatten()
            width = int(valid_columns[-1].item()) + 1 if valid_columns.numel() else 1
            ids = torch.cat([batch['student_prompts'], completion_ids[start:stop, :width]], dim=1)
            attention_mask = torch.cat([batch['student_prompt_attention_mask'], mask[:, :width]], dim=1)
            labels = ids.masked_fill(attention_mask == 0, -100)
            prompt_texts = self.processing_class.batch_decode(batch['student_prompts'], skip_special_tokens=False)
            micro_result = (ids, attention_mask, labels, prompt_texts, result[-1][start:stop])
            buffer.append((batch, micro_result))
        self._atd_metrics['atd/reward'].append(all_rewards.mean().item())
        self._atd_metrics['atd/adv_abs'].append(all_advantages.abs().mean().item())
        zero_std = grouped_rewards.max(dim=1).values == grouped_rewards.min(dim=1).values
        self._atd_metrics['atd/frac_reward_zero_std'].append(zero_std.float().mean().item())
        self._atd_metrics['atd/group_check_passed'].append(1.0)
        self._atd_metrics['atd/local_rollouts'].append(float(micro_size))
        self._atd_metrics['atd/local_generation_rollouts'].append(float(local_size))
        return buffer

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for column in ('Answer', 'atd_group_id'):
            if column not in self._signature_columns:
                self._signature_columns.append(column)
