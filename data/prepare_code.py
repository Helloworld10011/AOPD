# Prepare the original TACO stdin split; optional direct-code instruction preserves source row order/tests.
# Modified for the public release; original source hashes and changes are recorded in SOURCE_MAP.json.
"""Freeze a source-defined coding screen before inspecting model performance."""
import ast
import collections
import hashlib
import json
import random
from pathlib import Path

import argparse
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description="Rebuild the original shared-context TACO split.")
parser.add_argument("--output", default="data/code_original")
parser.add_argument("--direct_code", action="store_true")
args = parser.parse_args()
DEST = Path(args.output)
REPO = 'agentica-org/DeepCoder-Preview-Dataset'
REV = '177913a7bd43791646ef6a43645caa3c871ab3db'
MODELS = ['Qwen/Qwen3-4B-Instruct-2507', 'allenai/Olmo-3-7B-Instruct']
SYSTEM = 'Write Python code to solve the problem. Present the code in a Python code block at the end. Think first, then write the Python code in a ```python code block.'

assert not DEST.exists(), 'Frozen dataset already exists; do not overwrite'
# These models define the historical shared context-length eligibility filter.
revisions = {'Qwen/Qwen3-4B-Instruct-2507': 'cdbee75f17c01a7cc42f958dc650907174af0554', 'allenai/Olmo-3-7B-Instruct': '6e5971d9eba42665f5bd5a0fcf047f299ce1dccc'}
tokenizers = [AutoTokenizer.from_pretrained(m, revision=revisions[m]) for m in MODELS]
counts = collections.Counter()
seen = set()
rows = []
source_hashes = {}
for shard in range(4):
    name = f'taco/train-{shard:05d}-of-00004.parquet'
    path = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=REV))
    source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for ex in batch.to_pylist():
            counts['source_rows'] += 1
            tests = json.loads(ex['tests'])
            if tests.get('fn_name'):
                counts['function_call_excluded'] += 1
                continue
            if (len(tests['inputs']) != len(tests['outputs']) or len(tests['inputs']) < 5
                    or not all(isinstance(x, str) for x in tests['inputs'] + tests['outputs'])):
                counts['incompatible_tests'] += 1
                continue
            uid = hashlib.sha256(ex['problem'].strip().encode()).hexdigest()
            if uid in seen:
                counts['duplicate_problem'] += 1
                continue
            seen.add(uid)
            reference = None
            for solution in ex['solutions']:
                if not solution.strip():
                    continue
                try:
                    ast.parse(solution)
                except (SyntaxError, ValueError):
                    continue
                reference = solution
                break
            if reference is None:
                counts['no_python3_reference'] += 1
                continue
            messages = [{'role': 'user', 'content': ex['problem'] + '\n\n' + SYSTEM}]
            solution = '```python\n' + reference + '\n```'
            teacher = [{'role': 'user', 'content': messages[0]['content'] + '\n\nReference solution:\n' + solution + '\n\nWrite Python code to solve the original problem in the requested format.'}]
            lengths = []
            for tok in tokenizers:
                s = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                t = tok.apply_chat_template(teacher, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                lengths.append((len(s), len(t)))
            if any(s > 2048 or t > 8192 for s, t in lengths):
                counts['fixed_context_limit'] += 1
                continue
            if sum(len(x) for x in tests['inputs'][:5] + tests['outputs'][:5]) > 1048576:
                counts['five_test_payload_over_1MiB'] += 1
                continue
            reward_tests = json.dumps({k: tests[k][:5] for k in ('inputs', 'outputs')})
            rows.append({'uid': uid, 'messages': messages, 'output_text': solution, 'answer': reward_tests,
                         'full_tests': json.dumps(tests), 'source_shard': name})
    print(json.dumps({'shard': shard, 'counts': dict(counts), 'eligible': len(rows)}), flush=True)

assert len(rows) >= 2048, f'Insufficient eligible rows: {len(rows)}'
random.Random(42).shuffle(rows)
splits = {'eval_data': rows[:256], 'heldout_data': rows[256:768], 'train_data': rows[768:6368]}
DEST.mkdir(parents=True)
if args.direct_code:
    direct = 'Write Python code to solve the problem. Return only the complete Python solution in a ```python code block, without an explanation.'
    for row in rows:
        message = row['messages'][0]
        assert message['content'].endswith('\n\n' + SYSTEM)
        message['content'] = message['content'][:-len(SYSTEM)] + direct
for name, selected in splits.items():
    if name == 'train_data':
        records = [{k: r[k] for k in ('messages', 'output_text', 'answer', 'uid')} for r in selected]
    else:
        records = [{'prompt': r['messages'], 'answer': r['answer'], 'full_tests': r['full_tests'], 'uid': r['uid']} for r in selected]
    Dataset.from_list(records).save_to_disk(str(DEST / name))
assert len(rows) == 6177, 'Source/filter drift: do not silently change the experiment split'
assert {k: len(v) for k,v in splits.items()} == {'eval_data':256,'heldout_data':512,'train_data':5409}
manifest = {'source': REPO, 'revision': REV, 'source_sha256': source_hashes, 'eligibility_counts': dict(counts),
            'eligible': len(rows), 'split_sizes': {k: len(v) for k, v in splits.items()}, 'seed': 42,
            'model_revisions_for_shared_context_filter': revisions, 'student_prompt_limit': 2048, 'teacher_prompt_limit': 8192,
            'reference_selection': 'First nonempty syntactically valid Python3 released solution, before model evaluation; no model correctness/difficulty filter.',
            'training_reward': 'Fraction passing first five published tests; same existing code reward convention.',
            'development_metrics': 'First-five test fraction and all-five-pass rate; neither is full-suite pass@1.',
            'heldout': '512 problems reserved; full published tests preserved for later confirmation, never used for selecting runs.',
            'limitation': 'Native-instruct, stdin-only adaptation of literature-used DeepCoder TACO; not a reproduction of LOPD thinking-mode/full-corpus protocol.',
            'files_sha256': {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file()}}
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest), flush=True)
