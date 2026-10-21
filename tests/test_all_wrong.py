# CPU regression checks for group gating and the extra forward-KL signal.
import unittest
from collections import defaultdict
from unittest.mock import patch

import torch
import torch.nn.functional as F

from external_teacher.aopd_trainer import ATDDistillationTrainer


class AllWrongTests(unittest.TestCase):
    def test_groups_spanning_buffer_slices(self):
        parent = ATDDistillationTrainer.__mro__[1]
        trainer = object.__new__(ATDDistillationTrainer)
        trainer.num_generations = 4
        trainer._buffered_on_policy_flags = [True] * 4
        prompts = ['all_wrong'] * 4 + ['mixed'] * 4
        rewards = [0, 0, 0, 0, 0, 1, 0, 0]
        trainer._buffered_text_logs = [(prompts[i:i+2], [str(x) for x in rewards[i:i+2]]) for i in range(0, 8, 2)]
        trainer._buffered_inputs = [{'input_ids': torch.ones(2, 2, dtype=torch.long)} for _ in range(4)]
        trainer.answer_lookup = {'all_wrong': 'gold', 'mixed': 'gold'}
        trainer.reward_func = lambda prompts, completions, answers: [float(completions[0])]
        with patch.object(parent, '_fill_buffer', return_value=None):
            trainer._fill_buffer(None, 4)
        gate = torch.cat([x['atd_all_wrong'] for x in trainer._buffered_inputs])
        advantages = torch.cat([x['advantages'] for x in trainer._buffered_inputs])
        self.assertEqual(gate.tolist(), [True] * 4 + [False] * 4)
        torch.testing.assert_close(advantages, torch.tensor([0, 0, 0, 0, -1/3, 1, -1/3, -1/3]))

    def test_zero_advantage_kl_is_group_gated_and_masks_padding(self):
        trainer = object.__new__(ATDDistillationTrainer)
        trainer.w_clip = 4.
        trainer.mu = .2
        trainer._atd_metrics = defaultdict(list)
        trainer._current_advantages = torch.zeros(2)
        # The second wrong response belongs to a mixed group, so it is ungated.
        trainer._current_all_wrong = torch.tensor([True, False])
        student = torch.tensor([[[1., 0., -1.], [20., -20., 0.]], [[0., 1., -1.], [0., -1., 1.]]], requires_grad=True)
        teacher = torch.tensor([[[0., 1., -1.], [-20., 20., 0.]], [[1., 0., -1.], [1., -1., 0.]]])
        labels = torch.tensor([[0, -100], [1, 2]])
        loss = trainer.generalized_jsd_loss(student, teacher, labels=labels)
        expected = .2 * F.kl_div(F.log_softmax(student[0, 0], -1), F.softmax(teacher[0, 0], -1), reduction='sum') / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[0, 1].abs().sum().item(), 0.)
        self.assertEqual(student.grad[1].abs().sum().item(), 0.)
        self.assertGreater(student.grad[0, 0].abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
