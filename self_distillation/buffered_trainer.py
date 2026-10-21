# Buffer a full optimizer update so reward groups remain aligned across accumulated microbatches.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
import torch
from functools import partial
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker
from trl.trainer.utils import RepeatSampler, pad

from self_distillation.opsd_trainer import OPSDTrainer
from self_distillation.distributed_trainer import DistributedATDOPSDTrainer, ATDSelfDistillationDataCollator


class BufferedATDSelfDistillationDataCollator(ATDSelfDistillationDataCollator):
    def __call__(self, features):
        keys = ("problem", "solution", "Answer", "atd_group_id")
        return {"atd_features": [{key: feature[key] for key in keys} for feature in features]}

    def collate_microbatch(self, features):
        return super().__call__(features)


class BufferedATDOPSDTrainer(DistributedATDOPSDTrainer):
    """Buffer one optimizer update of rollouts; retain the existing two-row ATD loss path."""

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        steps = self.args.steps_per_generation
        if self.vllm_mode != "colocate" or self.vllm_tensor_parallel_size != 1:
            raise ValueError("This adapter requires colocated vLLM with tensor parallel size 1.")
        if self.vllm_enable_sleep_mode or self.reason_first:
            raise ValueError("Keep vLLM sleep mode and reason_first disabled.")
        if steps != self.args.gradient_accumulation_steps or steps < 1:
            raise ValueError("Set steps_per_generation equal to gradient_accumulation_steps.")
        if not self.args.dataloader_drop_last:
            raise ValueError("Set dataloader_drop_last=True.")
        if self.accelerator.split_batches or self.accelerator.dispatch_batches:
            raise ValueError("Use split_batches=False and dispatch_batches=False (or None).")
        if self.args.packing or self.args.group_by_length:
            raise ValueError("Keep packing=False and group_by_length=False.")
        if not isinstance(self.data_collator, BufferedATDSelfDistillationDataCollator):
            raise ValueError("Use BufferedATDSelfDistillationDataCollator with this trainer.")

        global_rollouts = self.args.per_device_train_batch_size * steps * self.accelerator.num_processes
        if global_rollouts % self.atd_num_generations:
            raise ValueError("The global generation batch must contain complete advantage groups.")
        prompts_per_batch = global_rollouts // self.atd_num_generations
        if len(train_dataset) < prompts_per_batch:
            raise ValueError("The dataset is smaller than one global generation batch.")
        return RepeatSampler(data_source=train_dataset, mini_repeat_count=self.atd_num_generations, batch_size=prompts_per_batch, repeat_count=steps, shuffle=True, seed=self.args.seed)

    def get_train_dataloader(self):
        if not isinstance(self.train_dataset, Dataset):
            raise ValueError("This adapter expects a map-style Hugging Face Dataset.")
        dataset = self._remove_unused_columns(self.train_dataset, description="training")
        kwargs = dict(batch_size=self._train_batch_size * self.args.steps_per_generation, collate_fn=self.data_collator, sampler=self._get_train_sampler(dataset), drop_last=True, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)
        if self.args.dataloader_num_workers > 0:
            kwargs.update(persistent_workers=self.args.dataloader_persistent_workers, prefetch_factor=self.args.dataloader_prefetch_factor, worker_init_fn=partial(seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index))
        return self.accelerator.prepare(DataLoader(dataset, **kwargs))

    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        if pad_token_id is None:
            raise ValueError("A padding token is required.")
        features = inputs.get("atd_features")
        micro_size = self.args.per_device_train_batch_size
        steps = self.args.steps_per_generation
        local_size = micro_size * steps
        if features is None or len(features) != local_size:
            raise RuntimeError(f"Expected {local_size} local generation rows from the buffered collator.")
        row_ids = tuple(feature["atd_group_id"] for feature in features)

        if not getattr(self, "_atd_rollout_buffer", None):
            self._atd_buffer_ids = row_ids
            self._atd_buffer_step = self.state.global_step
            self._atd_rollout_buffer = self._build_atd_rollout_buffer(features, generation_config, pad_token_id)
        elif row_ids != self._atd_buffer_ids or self.state.global_step != self._atd_buffer_step:
            raise RuntimeError("Generation-buffer alignment changed before the accumulated update completed.")

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
        generation_inputs = {
            "student_prompts": pad(
                prompt_rows,
                padding_value=pad_token_id,
                padding_side=self.processing_class.padding_side,
                # padding_side="right",
            ),
            "student_prompt_length": prompt_width,
        }

        # Generate once; do not invoke either earlier ATD override or expand each row again.
        result = OPSDTrainer._generate_on_policy_outputs_vllm(self, generation_inputs, generation_config, pad_token_id)
        generated_ids, generated_mask = result[:2]
        # if generated_ids.size(0) != local_size or not torch.equal(generated_ids[:, :prompt_width], generation_inputs["student_prompts"]):
        #     raise RuntimeError("The OPSD generator changed the prompt token IDs or local row count.")
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

        buffer = []
        for index, batch in enumerate(microbatches):
            start = index * micro_size
            stop = start + micro_size
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch.pop("ground_truth")
            batch.pop("atd_group_ids")
            batch["advantages"] = advantages[start:stop].contiguous()
            mask = completion_mask[start:stop]
            valid_columns = mask.any(dim=0).nonzero(as_tuple=False).flatten()
            width = int(valid_columns[-1].item()) + 1 if valid_columns.numel() else 1
            ids = torch.cat([batch["student_prompts"], completion_ids[start:stop, :width]], dim=1)
            attention_mask = torch.cat([batch["student_prompt_attention_mask"], mask[:, :width]], dim=1)
            labels = ids.masked_fill(attention_mask == 0, -100)
            prompt_texts = self.processing_class.batch_decode(batch["student_prompts"], skip_special_tokens=False)
            micro_result = (ids, attention_mask, labels, prompt_texts, result[-1][start:stop])
            buffer.append((batch, micro_result))

        self._atd_metrics["atd/reward"].append(all_rewards.mean().item())
        self._atd_metrics["atd/adv_abs"].append(all_advantages.abs().mean().item())
        zero_std = grouped_rewards.max(dim=1).values == grouped_rewards.min(dim=1).values
        self._atd_metrics["atd/frac_reward_zero_std"].append(zero_std.float().mean().item())
        self._atd_metrics["atd/group_check_passed"].append(1.0)
        self._atd_metrics["atd/local_rollouts"].append(float(micro_size))
        self._atd_metrics["atd/local_generation_rollouts"].append(float(local_size))
        return buffer
