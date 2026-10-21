# Check prompt identities and compute leave-one-out advantages across ranks.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import torch
from trl.trainer.utils import RepeatSampler
from self_distillation.data_collator import SelfDistillationDataCollator
from self_distillation.opsd_trainer import OPSDTrainer
from self_distillation.aopd_loss import ATDOPSDTrainer


class ATDSelfDistillationDataCollator(SelfDistillationDataCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["ground_truth"] = [feature["Answer"] for feature in features]
        batch["atd_group_ids"] = torch.tensor([feature["atd_group_id"] for feature in features], dtype=torch.long)
        return batch


class DistributedATDOPSDTrainer(ATDOPSDTrainer):
    """Reuse the existing ATD loss with rollout-based, cross-rank grouping."""

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for column in ("Answer", "atd_group_id"):
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if self.vllm_mode != "colocate" or self.vllm_tensor_parallel_size != 1:
            raise ValueError("This adapter requires colocated vLLM with tensor parallel size 1.")
        if self.vllm_enable_sleep_mode:
            raise ValueError("Keep vLLM sleep mode disabled for this adapter.")
        if self.args.steps_per_generation != 1:
            raise ValueError("Set --steps_per_generation 1 for this per-microstep generation adapter.")
        if not self.args.dataloader_drop_last:
            raise ValueError("Set --dataloader_drop_last True to require full local batches.")
        if self.accelerator.split_batches or self.accelerator.dispatch_batches:
            raise ValueError("Use split_batches=False and dispatch_batches=False (or None).")
        if self.args.packing or self.args.group_by_length:
            raise ValueError("Keep packing=False and group_by_length=False with the repeated-prompt sampler.")

        group_size = self.atd_num_generations
        global_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes
        if global_batch_size % group_size:
            raise ValueError("per_device_train_batch_size * world_size must be divisible by atd_num_generations.")
        prompts_per_batch = global_batch_size // group_size
        if len(train_dataset) < prompts_per_batch:
            raise ValueError("The training dataset is smaller than one complete global prompt batch.")

        return RepeatSampler(data_source=train_dataset, mini_repeat_count=group_size, batch_size=prompts_per_batch, repeat_count=1, shuffle=True, seed=self.args.seed)

    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        device = self.accelerator.device
        batch_size = inputs["student_prompts"].size(0)
        group_size = self.atd_num_generations
        rank = self.accelerator.process_index
        world = self.accelerator.num_processes

        local_count = torch.tensor([batch_size], dtype=torch.long, device=device)
        counts = self.accelerator.gather(local_count)
        if not (counts == self.args.per_device_train_batch_size).all().item():
            raise RuntimeError(f"Unexpected per-rank rollout counts: {counts.tolist()}")

        local_group_ids = inputs.pop("atd_group_ids").to(device)
        all_group_ids = self.accelerator.gather(local_group_ids)
        if all_group_ids.numel() != batch_size * world or all_group_ids.numel() % group_size:
            raise RuntimeError("The gathered batch does not contain complete advantage groups.")
        grouped_ids = all_group_ids.view(-1, group_size)
        if not (grouped_ids == grouped_ids[:, :1]).all().item():
            raise RuntimeError(f"Cross-rank prompt grouping is incorrect: {all_group_ids.tolist()}")
        if grouped_ids[:, 0].unique().numel() != grouped_ids.size(0):
            raise RuntimeError("A dataset row appears in more than one group in the same global microbatch.")

        # Bypass the old ATD override: the sampler already repeats prompts globally.
        result = OPSDTrainer._generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id)
        generated_ids = result[0]
        if generated_ids.size(0) != batch_size:
            raise RuntimeError("Generation changed the local rollout batch size.")
        prompt_length = inputs["student_prompt_length"]
        completion_texts = self.processing_class.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True)
        ground_truth = inputs.pop("ground_truth")
        if len(ground_truth) != batch_size:
            raise RuntimeError("Ground-truth answers are not aligned with local rollouts.")

        rewards = torch.as_tensor(self.reward_func(None, completion_texts, ground_truth), dtype=torch.float32, device=device)
        if rewards.shape != (batch_size,):
            raise RuntimeError("The reward function must return exactly one scalar per local rollout.")
        all_rewards = self.accelerator.gather(rewards)
        if not torch.isfinite(all_rewards).all().item():
            raise RuntimeError("The reward function returned non-finite values.")

        grouped_rewards = all_rewards.view(-1, group_size)
        grouped_advantages = grouped_rewards - (grouped_rewards.sum(dim=1, keepdim=True) - grouped_rewards) / (group_size - 1)
        all_advantages = grouped_advantages.reshape(-1)
        inputs["advantages"] = all_advantages[rank * batch_size:(rank + 1) * batch_size].contiguous()

        self._atd_metrics["atd/reward"].append(all_rewards.mean().item())
        self._atd_metrics["atd/adv_abs"].append(all_advantages.abs().mean().item())
        zero_std = grouped_rewards.max(dim=1).values == grouped_rewards.min(dim=1).values
        self._atd_metrics["atd/frac_reward_zero_std"].append(zero_std.float().mean().item())
        self._atd_metrics["atd/group_check_passed"].append(1.0)
        self._atd_metrics["atd/local_rollouts"].append(float(batch_size))
        return result
