# Self-distillation AOPD loss with original clipped importance weights and bounded residual.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from collections import defaultdict

from self_distillation.opsd_trainer import OPSDTrainer
from self_distillation.data_collator import SelfDistillationDataCollator
from torch.utils.checkpoint import checkpoint


class ATDSelfDistillationDataCollator(SelfDistillationDataCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["ground_truth"] = [f["Answer"] for f in features]
        return batch


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


class ATDOPSDTrainer(OPSDTrainer):
    def __init__(self, *args, reward_func=None, atd_num_generations=4, w_clip=4.0, mu=0.0, add_tail=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_accepts_loss_kwargs = False
        self.reward_func = reward_func
        self.atd_num_generations = atd_num_generations
        self.w_clip = w_clip
        if mu != 0:
            raise ValueError("Use all_wrong_trainer for nonzero group-gated KL.")
        self.mu = mu
        self.add_tail = add_tail
        self._current_advantages = None
        self._atd_metrics = defaultdict(list)

        if reward_func is None:
            raise ValueError("ATD requires reward_func.")
        if atd_num_generations < 2:
            raise ValueError("ATD requires atd_num_generations >= 2.")
        if not self.use_vllm:
            raise ValueError("This ATD implementation currently requires use_vllm=True.")
        if self.beta != 0:
            raise ValueError("ATD requires beta=0.")
        if self.use_thinking_machines_loss:
            raise ValueError("ATD is incompatible with use_thinking_machines_loss.")
        if self.jsd_token_clip is not None:
            raise ValueError("Set --jsd_token_clip 0 for ATD. ATD uses w_clip instead.")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "Answer" not in self._signature_columns:
            self._signature_columns.append("Answer")

    def _repeat_inputs(self, inputs, batch_size):
        G = self.atd_num_generations

        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                inputs[key] = value.repeat_interleave(G, dim=0)
            elif isinstance(value, list) and len(value) == batch_size:
                inputs[key] = [x for x in value for _ in range(G)]

    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        batch_size = inputs["student_prompts"].shape[0]

        self._repeat_inputs(inputs, batch_size)

        result = super()._generate_on_policy_outputs_vllm(
            inputs, generation_config, pad_token_id
        )

        generated_ids = result[0]
        student_prompt_len = inputs["student_prompt_length"]
        completion_texts = self.processing_class.batch_decode(
            generated_ids[:, student_prompt_len:],
            skip_special_tokens=True,
        )
        ground_truth = inputs.pop("ground_truth")

        rewards = self.reward_func(None, completion_texts, ground_truth)
        rewards = torch.tensor(
            [float(r) for r in rewards],
            dtype=torch.float32,
            device=self.accelerator.device,
        ).view(batch_size, self.atd_num_generations)

        G = self.atd_num_generations
        advantages = rewards - (rewards.sum(dim=1, keepdim=True) - rewards) / (G - 1)

        inputs["advantages"] = advantages.reshape(-1)

        with torch.no_grad():
            self._atd_metrics["atd/reward"].append(rewards.mean().item())
            self._atd_metrics["atd/adv_abs"].append(advantages.abs().mean().item())
            zero_std = (rewards.max(dim=1).values == rewards.min(dim=1).values).float().mean()
            self._atd_metrics["atd/frac_reward_zero_std"].append(zero_std.item())

        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._current_advantages = inputs.pop("advantages", None)

        if self._current_advantages is None:
            raise RuntimeError("ATD advantages were not attached to the batch.")

        try:
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        finally:
            self._current_advantages = None

    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        if self._current_advantages is None:
            return OPSDTrainer.generalized_jsd_loss(
                student_logits,
                teacher_logits,
                labels=labels,
                beta=beta,
                temperature=temperature,
                reduction=reduction,
                logits_are_probs=logits_are_probs,
                top_k=top_k,
                token_clip=token_clip,
            )

        if labels is None:
            raise ValueError("ATD requires completion labels.")

        A = self._current_advantages.to(student_logits.device)

        if A.numel() != student_logits.shape[0]:
            raise ValueError(
                f"advantages={A.numel()} but batch={student_logits.shape[0]}"
            )

        x_elem, residual_pos_tok, residual_neg_tok, residual_grad_tok, kappa_tok = truncated_fkl_terms(
            student_logits,
            teacher_logits,
            temperature=temperature,
            top_k=top_k or 0,
            add_tail=self.add_tail,
            w_clip=self.w_clip,
        )

        x_tok = x_elem.sum(-1)

        mask = labels != -100
        sampled_ids = labels.clamp_min(0)

        s_logp, t_logp = sampled_logps_chunked(
            student_logits,
            teacher_logits,
            sampled_ids,
            temperature=temperature,
            chunk_size=256,
        )

        with torch.no_grad():
            w = torch.exp(t_logp - s_logp).clamp(max=self.w_clip)

        core_tok = w * (-s_logp) - x_tok
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

            self._atd_metrics["atd/pos_loss"].append(contrib[pos].mean().item() if pos.any() else 0.0)
            self._atd_metrics["atd/neg_loss"].append(contrib[neg].mean().item() if neg.any() else 0.0)
            self._atd_metrics["atd/residual_pos"].append(residual_loss[pos].mean().item() if pos.any() else 0.0)
            self._atd_metrics["atd/residual_neg"].append(residual_loss[neg].mean().item() if neg.any() else 0.0)
            self._atd_metrics["atd/kappa_mass"].append(((kappa_tok * mask).sum() / mask.sum().clamp_min(1)).item())
            self._atd_metrics["atd/frac_w_clipped"].append(
                (((w >= self.w_clip) & mask).float().sum() / mask.float().sum().clamp_min(1)).item()
            )
            self._atd_metrics["atd/residual_grad_pos"].append(
                (A[pos] * residual_grad_seq[pos]).mean().item() if pos.any() else 0.0
            )
            self._atd_metrics["atd/residual_grad_neg"].append(
                ((-A[neg]) * residual_grad_seq[neg]).mean().item() if neg.any() else 0.0
            )

        return loss

    def log(self, logs, start_time=None):
        if self._atd_metrics:
            keys = sorted(self._atd_metrics)
            values = torch.tensor(
                [sum(self._atd_metrics[k]) / len(self._atd_metrics[k]) for k in keys],
                dtype=torch.float64,
                device=self.accelerator.device,
            )

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= dist.get_world_size()

            logs = logs.copy()
            logs.update({k: v.item() for k, v in zip(keys, values)})
            self._atd_metrics.clear()

        return super().log(logs, start_time)
