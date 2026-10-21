# Add forward KL only to groups where every rollout failed; retain the original math all-wrong implementation.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from trl.trainer.utils import pad

from self_distillation.opsd_trainer import OPSDTrainer
from self_distillation.buffered_trainer import BufferedATDOPSDTrainer
from self_distillation.aopd_loss import _add_tail_bucket


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


class AllWrongKLATDOPSDTrainer(BufferedATDOPSDTrainer):
    """Existing buffered ATD plus forward KL only on all-wrong reward groups."""

    def __init__(self, *args, mu=0.2, **kwargs):
        if not math.isfinite(mu) or mu < 0:
            raise ValueError("mu must be finite and nonnegative.")
        # Keep the parent trainer's unconditional KL disabled.
        super().__init__(*args, mu=0.0, **kwargs)
        self.all_wrong_kl_mu = float(mu)
        self._current_all_wrong = None

    def _build_atd_rollout_buffer(self, features, generation_config, pad_token_id):
        device = self.accelerator.device
        micro_size = self.args.per_device_train_batch_size
        local_size = len(features)
        group_size = self.atd_num_generations
        local_count = torch.tensor([local_size], device=device, dtype=torch.long)
        counts = self.accelerator.gather(local_count)
        if not (counts == local_size).all().item():
            raise RuntimeError(f"Unequal generation batch sizes across ranks: {counts.tolist()}")

        local_ids = torch.tensor([feature["atd_group_id"] for feature in features], device=device, dtype=torch.long)
        all_ids = self.accelerator.gather(local_ids)
        grouped_ids = all_ids.view(-1, group_size)
        if not (grouped_ids == grouped_ids[:, :1]).all().item():
            raise RuntimeError("Gathered rewards would combine different dataset rows in one group.")
        if grouped_ids[:, 0].unique().numel() != grouped_ids.size(0):
            raise RuntimeError("A dataset row appears in multiple groups in one generation batch.")

        # Collate each training microbatch separately, preserving its original prompt padding.
        microbatches = [self.data_collator.collate_microbatch(features[start:start + micro_size]) for start in range(0, local_size, micro_size)]
        prompt_rows = []
        for batch in microbatches:
            for ids, mask in zip(batch["student_prompts"], batch["student_prompt_attention_mask"]):
                prompt_rows.append(ids[mask.bool()].to(device))
        prompt_width = max(row.numel() for row in prompt_rows)
        if self.args.max_length and prompt_width > self.args.max_length - generation_config.max_new_tokens:
            raise ValueError("A student prompt exceeds the reserved prompt budget; do not silently change its truncation.")
        generation_inputs = {"student_prompts": pad(prompt_rows, padding_value=pad_token_id, padding_side="right"), "student_prompt_length": prompt_width}

        # Generate once; do not invoke either earlier ATD override or expand each row again.
        result = OPSDTrainer._generate_on_policy_outputs_vllm(self, generation_inputs, generation_config, pad_token_id)
        generated_ids, generated_mask = result[:2]
        if generated_ids.size(0) != local_size or not torch.equal(generated_ids[:, :prompt_width], generation_inputs["student_prompts"]):
            raise RuntimeError("The OPSD generator changed the prompt token IDs or local row count.")
        completion_ids = generated_ids[:, prompt_width:]
        completion_mask = generated_mask[:, prompt_width:]
        if completion_ids.size(1) == 0:
            raise RuntimeError("vLLM returned an empty completion batch.")
        texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        answers = [feature["Answer"] for feature in features]
        rewards = torch.as_tensor(self.reward_func(None, texts, answers), dtype=torch.float32, device=device)
        if rewards.shape != (local_size,):
            raise RuntimeError("The reward function must return one scalar per rollout.")
        all_rewards = self.accelerator.gather(rewards)
        if not torch.isfinite(all_rewards).all().item():
            raise RuntimeError("Non-finite reward encountered.")
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
            batch.pop("ground_truth")
            batch.pop("atd_group_ids")
            batch["advantages"] = advantages[start:stop].contiguous()
            batch["atd_all_wrong"] = all_wrong[start:stop].contiguous()
            mask = completion_mask[start:stop]
            valid_columns = mask.any(dim=0).nonzero(as_tuple=False).flatten()
            width = int(valid_columns[-1].item()) + 1 if valid_columns.numel() else 1
            ids = torch.cat([batch["student_prompts"], completion_ids[start:stop, :width]], dim=1)
            attention_mask = torch.cat([batch["student_prompt_attention_mask"], mask[:, :width]], dim=1)
            labels = ids.masked_fill(attention_mask == 0, -100)
            prompt_texts = self.processing_class.batch_decode(batch["student_prompts"], skip_special_tokens=False)
            micro_result = (ids, attention_mask, labels, prompt_texts, result[-1][start:stop])
            buffer.append((batch, micro_result))

        self._atd_metrics["atd/frac_all_wrong_groups"].append(group_all_wrong.float().mean().item())
        self._atd_metrics["atd/reward"].append(all_rewards.mean().item())
        self._atd_metrics["atd/adv_abs"].append(all_advantages.abs().mean().item())
        zero_std = grouped_rewards.max(dim=1).values == grouped_rewards.min(dim=1).values
        self._atd_metrics["atd/frac_reward_zero_std"].append(zero_std.float().mean().item())
        self._atd_metrics["atd/group_check_passed"].append(1.0)
        self._atd_metrics["atd/local_rollouts"].append(float(micro_size))
        self._atd_metrics["atd/local_generation_rollouts"].append(float(local_size))
        return buffer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._current_all_wrong = inputs.pop("atd_all_wrong", None)
        if self.all_wrong_kl_mu != 0 and self._current_all_wrong is None:
            raise RuntimeError("The all-wrong group mask was not attached to the rollout batch.")
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            self._current_all_wrong = None

    def generalized_jsd_loss(self, student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean", logits_are_probs=False, top_k=None, token_clip=None):
        loss = super().generalized_jsd_loss(student_logits, teacher_logits, labels=labels, beta=beta, temperature=temperature, reduction=reduction, logits_are_probs=logits_are_probs, top_k=top_k, token_clip=token_clip)
        if self._current_advantages is None:
            return loss
        if self.all_wrong_kl_mu == 0:
            self._atd_metrics["atd/extra_wrong_kl"].append(0.0)
            return loss

        gate = self._current_all_wrong
        if gate is None or gate.shape != (student_logits.size(0),):
            raise RuntimeError("Missing or misaligned all-wrong group mask.")
        gate = gate.to(device=student_logits.device, dtype=torch.bool)
        if not gate.any().item():
            self._atd_metrics["atd/extra_wrong_kl"].append(0.0)
            return loss

        parts = []
        for start in range(0, student_logits.size(1), 256):
            end = min(start + 256, student_logits.size(1))
            s_chunk = student_logits[:, start:end, :]
            t_chunk = teacher_logits[:, start:end, :]
            if torch.is_grad_enabled() and s_chunk.requires_grad:
                part = checkpoint(_forward_kl_chunk, s_chunk, t_chunk, temperature, top_k, self.add_tail, use_reentrant=False, preserve_rng_state=False)
            else:
                part = _forward_kl_chunk(s_chunk, t_chunk, temperature, top_k, self.add_tail)
            parts.append(part)

        kl_tok = torch.cat(parts, dim=1)
        mask = labels != -100
        kl_seq = torch.where(mask, kl_tok, torch.zeros_like(kl_tok)).sum(-1) / mask.sum(-1).clamp_min(1)
        extra_kl = self.all_wrong_kl_mu * (gate.to(kl_seq.dtype) * kl_seq).mean()
        self._atd_metrics["atd/extra_wrong_kl"].append(extra_kl.detach().item())
        return loss + extra_kl
