# External-teacher AOPD: leave-one-out advantages, clipped core, residual, and all-wrong group KL.
from trl.experimental.distillation import DistillationTrainer
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
import math

def _add_tail_bucket(log_probs, valid_mask):
    """Append a (K+1)-th tail element: log(1 - sum(exp(top_k_logps)))."""
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum))
    tail_mask = torch.ones_like(valid_mask[..., :1], dtype=torch.bool)
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def truncated_fkl_loss(student_logits, teacher_logits, labels=None, temperature=1.0,
                    reduction="batchmean", top_k=0, add_tail=True, w_clip=4.0,
                    return_residual=False):
    """Forward KL with T replaced by T~ = min(T, w_clip * S_detached).

    If return_residual=True, also returns likelihood/unlikelihood losses for
    the residual K = T - T~.
    """
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    support_mask = None
    if top_k > 0 and student_logits.size(-1) > top_k:
        neg_inf = torch.full((), float("-inf"), dtype=student_logits.dtype, device=student_logits.device)
        student_log_probs_full = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs_full = F.log_softmax(teacher_logits, dim=-1)

        # beta == 0: teacher-selected support
        _, support = teacher_logits.topk(top_k, dim=-1)
        support_mask = torch.ones_like(support, dtype=torch.bool)

        student_support_logps = student_log_probs_full.gather(-1, support)
        teacher_support_logps = teacher_log_probs_full.gather(-1, support)
        student_topk_logps = torch.where(support_mask, student_support_logps, neg_inf)
        teacher_topk_logps = torch.where(support_mask, teacher_support_logps, neg_inf)

        if add_tail:
            base_support_mask = support_mask
            student_log_probs, support_mask = _add_tail_bucket(student_topk_logps, base_support_mask)
            teacher_log_probs, _ = _add_tail_bucket(teacher_topk_logps, base_support_mask)
        else:
            student_log_probs = student_topk_logps - torch.logsumexp(student_topk_logps, dim=-1, keepdim=True)
            teacher_log_probs = teacher_topk_logps - torch.logsumexp(teacher_topk_logps, dim=-1, keepdim=True)
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    trunc_teacher_log_probs = torch.minimum(
        teacher_log_probs, math.log(w_clip) + student_log_probs.detach()).detach()

    if support_mask is not None:
        safe_student = torch.where(support_mask, student_log_probs, torch.zeros_like(student_log_probs))
        safe_teacher = torch.where(support_mask, trunc_teacher_log_probs, torch.zeros_like(trunc_teacher_log_probs))
        trunc_teacher_probs = torch.where(
            support_mask, trunc_teacher_log_probs.exp(), torch.zeros_like(trunc_teacher_log_probs))
        jsd = torch.nan_to_num(trunc_teacher_probs * (safe_teacher - safe_student), nan=0.0)
    else:
        safe_student = student_log_probs
        trunc_teacher_probs = trunc_teacher_log_probs.exp()
        jsd = torch.nan_to_num(
            trunc_teacher_probs * (trunc_teacher_log_probs - student_log_probs), nan=0.0)

    x_elem = DistillationTrainer._reduce_divergence_loss(
        jsd, labels=labels, reduction=reduction)

    if not return_residual:
        return x_elem

    with torch.no_grad():
        if support_mask is not None:
            teacher_probs = torch.where(
                support_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))
            safe_full_teacher = torch.where(
                support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
        else:
            teacher_probs = teacher_log_probs.exp()
            safe_full_teacher = teacher_log_probs

        residual_probs = (teacher_probs - trunc_teacher_probs).clamp_min(0)

    ratio_logit = math.log(w_clip) + safe_student - safe_full_teacher.detach()
    ratio_prob = torch.sigmoid(ratio_logit.float()).to(student_log_probs.dtype)

    residual_pos_tok = (residual_probs * (4.0 * ratio_prob)).sum(-1)
    residual_neg_tok = (residual_probs * (4.0 * (1.0 - ratio_prob))).sum(-1)
    residual_grad_tok = (residual_probs * (4.0 * ratio_prob * (1.0 - ratio_prob))).sum(-1)

    kappa_tok = residual_probs.sum(-1)

    return x_elem, residual_pos_tok, residual_neg_tok, residual_grad_tok, kappa_tok


class AOPDTrainer(DistillationTrainer):
    def __init__(self, *args, reward_func=None, answer_lookup=None, w_clip=4.0, mu=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.reward_func = reward_func
        self.answer_lookup = answer_lookup
        if not math.isfinite(w_clip) or w_clip <= 0 or not math.isfinite(mu) or mu < 0:
            raise ValueError("w_clip must be positive and mu must be nonnegative; both must be finite.")
        self.w_clip = w_clip
        self._current_advantages = None
        self._current_all_wrong = None
        self.mu = mu
        self._atd_metrics = defaultdict(list)

    # ---------- 1. rewards + advantages, right after generation ----------
    def _fill_buffer(self, generation_batch, buffer_steps):
        super()._fill_buffer(generation_batch, buffer_steps)

        rows = []
        for i, on_policy in enumerate(self._buffered_on_policy_flags):
            if not on_policy:
                continue
            texts = self._buffered_text_logs[i]
            if texts is None:
                texts = self._decode_slice(i)
            for j, (p, c) in enumerate(zip(*texts)):
                rows.append((i, j, p, c))
        if not rows:
            return

        R = {}
        for i, j, p, c in rows:
            gt = self.answer_lookup.get(p)
            R[(i, j)] = float(self.reward_func(None, [c], [gt])[0]) if gt is not None else 0.0

        groups = defaultdict(list)
        for i, j, p, _ in rows:
            groups[p].append((i, j))

        A = {}
        all_wrong = {}
        for p, members in groups.items():
            if len(members) != self.num_generations:
                raise RuntimeError("Incomplete local reward group: set local batch*accumulation to a multiple of num_generations and use unique prompts.")
            rs = [R[m] for m in members]
            if not all(math.isfinite(r) for r in rs):
                raise ValueError("Rewards must be finite.")
            group_failed = all(r <= 0 for r in rs)
            for m, r in zip(members, rs):
                all_wrong[m] = group_failed
                b = (sum(rs) - r) / (len(rs) - 1) if len(rs) > 1 else 0.0
                A[m] = r - b

        for i, on_policy in enumerate(self._buffered_on_policy_flags):
            if not on_policy or self._buffered_inputs[i] is None:
                continue
            B = self._buffered_inputs[i]["input_ids"].shape[0]
            self._buffered_inputs[i]["advantages"] = torch.tensor(
                [A.get((i, j), 0.0) for j in range(B)], dtype=torch.float32
            )
            self._buffered_inputs[i]["atd_all_wrong"] = torch.tensor([all_wrong[(i, j)] for j in range(B)], dtype=torch.bool)

    def _decode_slice(self, i):
        sl = self._buffered_inputs[i]
        pl = self._compute_prompt_length(sl)
        comps = self.processing_class.batch_decode(sl["input_ids"][:, pl:], skip_special_tokens=True)
        pad = self.processing_class.pad_token_id
        prompts = [
            self.processing_class.decode(ids[ids != pad], skip_special_tokens=False)
            for ids in sl["input_ids"][:, :pl]
        ]
        return prompts, comps

    # ---------- 2. hand advantages to the loss ----------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._current_advantages = inputs.pop("advantages", None)
        self._current_all_wrong = inputs.pop("atd_all_wrong", None)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    # ---------- 3. the ATD loss ----------
    def generalized_jsd_loss(self, student_logits, teacher_logits, labels=None,
                            beta=0.0, temperature=1.0, reduction="batchmean",
                            top_k=0, add_tail=True):
        if self._current_advantages is None:
            return DistillationTrainer.generalized_jsd_loss(
                student_logits, teacher_logits, labels=labels, beta=beta,
                temperature=temperature, reduction=reduction, top_k=top_k, add_tail=add_tail)

        A = self._current_advantages.to(student_logits.device)
        if self._current_all_wrong is None or self._current_all_wrong.shape != A.shape:
            raise RuntimeError("Missing or misaligned all-wrong group labels.")
        all_wrong = self._current_all_wrong.to(device=A.device, dtype=A.dtype)

        # corrected core + explicit residual
        x_elem, residual_pos_tok, residual_neg_tok, residual_grad_tok, kappa_tok = checkpoint(
            truncated_fkl_loss, student_logits, teacher_logits, labels=None,
            temperature=temperature, reduction="none",
            top_k=top_k, add_tail=add_tail, w_clip=self.w_clip,
            return_residual=True, use_reentrant=False)
        x_tok = x_elem.sum(-1)

        # sampled term
        mask = labels != -100
        tok = labels.clamp_min(0).unsqueeze(-1)
        s_logps = F.log_softmax(student_logits / temperature, -1)
        with torch.no_grad():
            t_logps = F.log_softmax(teacher_logits / temperature, -1)
        s_logp = s_logps.gather(-1, tok).squeeze(-1)
        t_logp = t_logps.gather(-1, tok).squeeze(-1)
        with torch.no_grad():
            w = torch.exp(t_logp - s_logp).clamp(max=self.w_clip)

        core_tok = w * (-s_logp) - x_tok
        core_tok = torch.where(mask, core_tok, torch.zeros_like(core_tok))
        core_seq = core_tok.sum(-1) / mask.sum(-1).clamp_min(1)

        # Clipping leaves K = T - min(T, c*S_detached). Keep the original
        # bounded sigmoid residual: positive A discourages this leftover mass;
        # negative A encourages it. The bounded sigmoid is preserved.
        residual_pos_tok = torch.where(mask, residual_pos_tok, torch.zeros_like(residual_pos_tok))
        residual_neg_tok = torch.where(mask, residual_neg_tok, torch.zeros_like(residual_neg_tok))
        residual_pos_seq = residual_pos_tok.sum(-1) / mask.sum(-1).clamp_min(1)
        residual_neg_seq = residual_neg_tok.sum(-1) / mask.sum(-1).clamp_min(1)
        residual_seq = torch.where(A > 0, residual_pos_seq, residual_neg_seq)

        core_loss = A * core_seq
        residual_loss = A.abs() * residual_seq
        loss = (core_loss + residual_loss).mean()

        with torch.no_grad():
            residual_grad_tok = torch.where(mask, residual_grad_tok, torch.zeros_like(residual_grad_tok))
            residual_grad_seq = residual_grad_tok.sum(-1) / mask.sum(-1).clamp_min(1)
            pos, neg = (A > 0), (A < 0)
            contrib = core_loss + residual_loss

            self._atd_metrics["atd/pos_loss"].append(contrib[pos].mean().item() if pos.any() else 0.0)
            self._atd_metrics["atd/neg_loss"].append(contrib[neg].mean().item() if neg.any() else 0.0)
            self._atd_metrics["atd/residual_pos"].append(residual_loss[pos].mean().item() if pos.any() else 0.0)
            self._atd_metrics["atd/residual_neg"].append(residual_loss[neg].mean().item() if neg.any() else 0.0)
            self._atd_metrics["atd/kappa_mass"].append(((kappa_tok * mask).sum() / mask.sum().clamp_min(1)).item())
            self._atd_metrics["atd/frac_w_clipped"].append(((w >= self.w_clip) & mask).float().sum().div(
                mask.float().sum().clamp_min(1)).item())
            self._atd_metrics["atd/residual_grad_pos"].append(
                (A[pos] * residual_grad_seq[pos]).mean().item() if pos.any() else 0.0)
            self._atd_metrics["atd/residual_grad_neg"].append(
                ((-A[neg]) * residual_grad_seq[neg]).mean().item() if neg.any() else 0.0)

        if self.mu != 0.0:
            kl_elem = DistillationTrainer.generalized_jsd_loss(
                student_logits, teacher_logits, labels=None, beta=0.0,
                temperature=temperature, reduction="none",
                top_k=top_k, add_tail=add_tail)
            kl_tok = kl_elem.sum(-1)
            kl_seq = torch.where(mask, kl_tok, torch.zeros_like(kl_tok)).sum(-1) / mask.sum(-1).clamp_min(1)
            wrong_kl = (all_wrong * kl_seq).mean()
            loss = loss + self.mu * wrong_kl
            self._atd_metrics["atd/extra_wrong_kl"].append((self.mu * wrong_kl).detach().item())

        return loss

    def log(self, logs, start_time=None):
            if self.state.is_world_process_zero and self._atd_metrics:
                logs = logs.copy()
                logs.update({key: sum(values) / len(values) for key, values in self._atd_metrics.items() if values})
                self._atd_metrics.clear()

            return super().log(logs, start_time)
