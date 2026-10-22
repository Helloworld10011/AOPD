# Main-task datasets, original reference prompts, and task rewards.
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk
from .opsd_base import SelfDistillationDataCollator


def load_training_data(task, data_dir=None):
    if task == 'math':
        data = load_dataset('siyanzhao/Openthoughts_math_30k_opsd', split='train')
        data = data.train_test_split(test_size=512, seed=42)['train']
    elif task == 'medical':
        data = load_dataset('ytz20/sys_medmcqa_train', split='train')
        def prepare(row):
            if not row.get('teacher_response'):
                raise ValueError('A medical example is missing its released teacher response.')
            return {'problem': row['content'], 'solution': row['teacher_response'], 'Answer': str(row['answer']).strip().upper()}
        data = data.map(prepare, remove_columns=data.column_names)
    else:
        root = Path(data_dir or f'data/{task}')
        if not (root / 'train_data').is_dir():
            raise FileNotFoundError(f'Prepare {task} data first: python -m data.prepare --task {task}')
        data = load_from_disk(str(root / 'train_data'))
        data = data.rename_columns({'messages': 'problem', 'output_text': 'solution'})
        if 'Answer' not in data.column_names:
            data = data.rename_column('answer', 'Answer')
    return data.add_column('atd_group_id', list(range(len(data))))


def get_reward(task):
    if task == 'code':
        from .code_reward import correct_reward, check_docker
        check_docker()
        return correct_reward
    if task == 'math':
        from eval.math_grader import boxed_reward_fn, timeout
        def reward(prompts, completions, answers, **kwargs):
            rewards = []
            for text, answer in zip(completions, answers):
                try:
                    with timeout(3):
                        _, score = boxed_reward_fn(text, answer, fast=True)
                    rewards.append(score)
                except Exception:
                    rewards.append(0.)
            return rewards
        return reward
    from eval.qa import medical_answer, biology_answer
    parse = medical_answer if task == 'medical' else biology_answer
    def reward(prompts, completions, answers, **kwargs):
        if len(completions) != len(answers):
            raise ValueError('Completion and answer counts differ.')
        return [float(parse(text) == str(answer).strip().upper()) for text, answer in zip(completions, answers)]
    return reward


class AOPDDataCollator:
    """Buffer raw examples, then build the original task-specific teacher/student prompts."""
    def __init__(self, tokenizer, task, max_length, max_completion_length):
        self.tokenizer, self.task = tokenizer, task
        self.max_length = max_length - max_completion_length
        if task == 'math':
            self.math_collator = SelfDistillationDataCollator(
                tokenizer, max_length=max_length, reason_first=False,
                student_thinking=False, teacher_thinking=False)
        elif task in ('biology', 'code'):
            tokenizer.padding_side = 'right'

    def __call__(self, features):
        keys = ('problem', 'solution', 'Answer', 'atd_group_id')
        return {'atd_features': [{key: row[key] for key in keys} for row in features]}

    def collate_microbatch(self, features):
        if self.task == 'math':
            batch = self.math_collator(features)
        else:
            students, teachers = [], []
            for row in features:
                student, reference = row['problem'], row['solution']
                if not student or student[-1]['role'] != 'user' or not isinstance(reference, str) or not reference.strip():
                    raise ValueError('Each example needs a student question and a nonempty reference.')
                teacher = [dict(message) for message in student]
                if self.task == 'code':
                    teacher[-1]['content'] += f'\n\nReference solution:\n{reference}\n\nWrite Python code to solve the original problem in the requested format.'
                else:
                    lead = 'This is an example for a response to the question:' if self.task == 'medical' else 'The verified correct answer to the question is:'
                    teacher[-1]['content'] = f"\n{student[-1]['content']}\n\n{lead}\n{reference}\n\nNow answer with a response of your own, including the thinking process.\n"
                students.append(self.tokenizer.apply_chat_template(student, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                teachers.append(self.tokenizer.apply_chat_template(teacher, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            batch = {}
            for role, prompts in [('student', students), ('teacher', teachers)]:
                encoded = self.tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors='pt')
                if encoded.input_ids.size(1) > self.max_length:
                    raise ValueError('A prompt exceeds the context budget; do not truncate the reference.')
                batch[f'{role}_prompts'] = encoded.input_ids
                batch[f'{role}_prompt_attention_mask'] = encoded.attention_mask
                batch[f'{role}_prompt_length'] = encoded.input_ids.size(1)
                batch[f'{role}_prompt_lengths_per_example'] = encoded.attention_mask.sum(-1)
        batch['ground_truth'] = [row['Answer'] for row in features]
        batch['atd_group_ids'] = torch.tensor([row['atd_group_id'] for row in features], dtype=torch.long)
        return batch
