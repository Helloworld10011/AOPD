# Prepare the main training data and QA evaluation inputs.
import argparse
import ast
import collections
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from eval.qa import preprocess, select_by_category, format_cot_example

SOURCE = "https://raw.githubusercontent.com/Moenupa/SDPO-CL/d277934412bd703838f9225086abe4b1795103c8/datasets/sciknoweval/biology/"

PINS = {'train.json': '7ffbfe4a85c7386e2ca78abd3de82c84d339ecbf0beb9bff93f3e599ac6298f2', 'test.json': '0ffe8df8f33154ca6af90c75b278f8602d75376db79e295e16a29fc9a5da325a'}

def uid(row):
    return hashlib.sha256(' '.join(row['prompt'].split()).encode()).hexdigest()

def prepare_biology(args):
    dest=Path(args.output)
    if dest.exists(): raise FileExistsError(f"Do not overwrite a prepared split: {dest}")
    sources={}
    for name,expected in PINS.items():
        payload=(Path(args.source_dir)/name).read_bytes() if args.source_dir else urllib.request.urlopen(SOURCE+name).read()
        if hashlib.sha256(payload).hexdigest()!=expected: raise ValueError(f"Source hash changed: {name}")
        sources[name]=[json.loads(line) for line in payload.decode().splitlines()]
    train,test=sources['train.json'],sources['test.json']
    assert len(train)==450 and len(test)==50 and len({uid(x) for x in train+test})==500
    ordered=sorted(train,key=lambda x:hashlib.sha256(('biology-dev-42:'+uid(x)).encode()).hexdigest())
    dev,remaining=ordered[:128],ordered[128:]
    tr=[{'uid':uid(x),'messages':[{'role':'system','content':x['system']},{'role':'user','content':x['prompt']}],
         'output_text':'<answer>\n'+x['answer']+'\n</answer>','Answer':x['answer']} for x in remaining]
    dv=[{'uid':uid(x),'context_uid':uid(x),'prompt':[{'role':'system','content':x['system']},{'role':'user','content':x['prompt']}],'answer':x['answer']} for x in dev]
    ev=[{'uid':uid(x),'prompt':[{'role':'system','content':x['system']},{'role':'user','content':x['prompt']}],'gold':x['answer']} for x in test]
    dest.mkdir(parents=True)
    Dataset.from_list(tr).save_to_disk(str(dest/'train_data'))
    Dataset.from_list(dv).save_to_disk(str(dest/'eval_data'))
    for name,rows in [('train_rows.jsonl',tr),('dev_rows.jsonl',dv),('source50.jsonl',ev)]:
        (dest/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    manifest={'source':SOURCE,'source_sha256':PINS,'counts':{'train':322,'development':128,'test':50},
              'row_sha256':{n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in ['train_rows.jsonl','dev_rows.jsonl','source50.jsonl']}}
    assert manifest['row_sha256']['train_rows.jsonl']=='19ae9ff827f1ec58dc6db51288d8057a316482d65cc5cbcd3df3a51434f12134'
    assert manifest['row_sha256']['dev_rows.jsonl']=='016361b46c805b0e3eb83faa3cd624e456d0183d70caff5ab7114620f877d1f0'
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

SPECS = [
 ('medqa_test','bigbio/med_qa','484a6c066fe8e75c83edea0c88b5169316714fcd','med_qa_en_4options_source/test-00000-of-00001.parquet',1273,'b50b49fad2642c10ed343c6174d5ab0cb8efad434e7b66465ac471094eb57deb'),
 ('medmcqa_validation','openlifescienceai/medmcqa','91c6572c454088bf71b679ad90aa8dffcd0d5868','data/validation-00000-of-00001.parquet',4183,'b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021'),
]

def prepare_medical(args):
    dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    for name,repo,rev,filename,count,expected in SPECS:
        source=Path(hf_hub_download(repo,filename,repo_type='dataset',revision=rev))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        raw=pq.read_table(source).to_pylist();assert len(raw)==count;rows=[]
        for index,x in enumerate(raw):
            if name=='medqa_test':
                opts=x['options']
                if isinstance(opts,list):opts={z['key']:z['value'] for z in opts}
                answer=x['answer_idx'];assert opts[answer]==x['answer']
            else:
                opts={k:x['op'+k.lower()] for k in 'ABCD'};answer='ABCD'[x['cop']]
            text=('Given the following question and candidate answers, choose the best answer. <Question>'+x['question']+'\nChoices:\n'+''.join(k+'. '+opts[k]+'\n' for k in 'ABCD')+'</Question>\nAt the end present your answer in <answer> and </answer> using an uppercase letter (A, B, C, or D).')
            rows.append({'uid':name+'_'+str(index),'prompt':[{'role':'user','content':text}],'gold':answer})
        path=dest/(name+'.jsonl');path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        print(json.dumps({'benchmark':name,'questions':len(rows),'data_revision':rev,'raw_sha256':expected}))

REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

INITIAL = 'The following are multiple choice questions (with answers) about biology. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n\n'

def prepare_mmlupro(args):
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    ds=load_dataset('TIGER-Lab/MMLU-Pro',revision=REVISION)
    examples=select_by_category(preprocess(list(ds['validation'])),'biology')[:5]
    test=select_by_category(preprocess(list(ds['test'])),'biology')
    assert len(examples)==5 and len(test)==717
    prefix=INITIAL+''.join(format_cot_example(x,True) for x in examples)
    rows=[{'uid':'mmlupro_biology_'+str(x['question_id']),'prompt':[{'role':'user','content':prefix+format_cot_example(x,False)}],'gold':x['answer'],'source':x['src']} for x in test]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    print(json.dumps({'questions':len(rows),'data_revision':REVISION,'demonstration_question_ids':[x['question_id'] for x in examples],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))

def prepare_code(args):
    DEST = Path(args.output)
    REPO = 'agentica-org/DeepCoder-Preview-Dataset'
    REV = '177913a7bd43791646ef6a43645caa3c871ab3db'
    MODELS = ['Qwen/Qwen3-4B-Instruct-2507', 'allenai/Olmo-3-7B-Instruct']
    SYSTEM = 'Write Python code to solve the problem. Present the code in a Python code block at the end. Think first, then write the Python code in a ```python code block.'
    assert not DEST.exists(), 'Frozen dataset already exists; do not overwrite'
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
    print(f"Prepared {len(splits['train_data'])} training and 256 development examples.")

def prepare_external_code(args):
    import json
    import logging
    import shutil
    import sys
    import threading
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from datasets import load_dataset
    from training.external_teacher.code_reward import compute_score as code_compute_score
    SANDBOX_URL = "http://127.0.0.1:8080/run_code"
    OUT_DIR = Path(args.output)
    if OUT_DIR.exists():
        raise FileExistsError(f"Do not overwrite a prepared split: {OUT_DIR}")
    MAX_TESTS = 10
    MIN_TESTS = 5
    MAX_REFERENCE_SOLUTIONS = 5
    VERIFY_WORKERS = 8
    VERIFY_TIMEOUT = 10
    _SEM = threading.Semaphore(32)
    logging.getLogger("sandbox_fusion").setLevel(logging.WARNING)
    logging.getLogger("sandbox_fusion.utils").setLevel(logging.WARNING)
    def parse_json(x):
        if isinstance(x, str):
            return json.loads(x)
        return x
    def to_str(x):
        if isinstance(x, (list, tuple)):
            return "\n".join(map(str, x))
        return str(x)
    def is_usable_stdin_problem(ex):
        try:
            io = parse_json(ex["input_output"])
            if io.get("fn_name"):
                return False
            if io.get("assert_case"):
                return False

            inputs = io.get("inputs")
            outputs = io.get("outputs")

            return (
                isinstance(inputs, list)
                and isinstance(outputs, list)
                and len(inputs) == len(outputs)
                and len(inputs) >= MIN_TESTS
            )
        except Exception:
            return False
    def make_tests(ex):
        io = parse_json(ex["input_output"])
        n = min(MAX_TESTS, len(io["inputs"]), len(io["outputs"]))

        tests = {
            "inputs": [to_str(x) for x in io["inputs"][:n]],
            "outputs": [to_str(x) for x in io["outputs"][:n]],
        }

        return {"ground_truth": json.dumps(tests, ensure_ascii=False)}
    def py3_solutions(solutions, limit=MAX_REFERENCE_SOLUTIONS):
        if isinstance(solutions, str):
            try:
                solutions = json.loads(solutions)
            except json.JSONDecodeError:
                solutions = [solutions]

        good = []

        for solution in solutions:
            if not isinstance(solution, str):
                continue

            try:
                compile(solution, "<solution>", "exec")
            except SyntaxError:
                continue

            good.append(solution)

            if len(good) >= limit:
                break

        return good
    def verify(ex):
        for solution in py3_solutions(ex["solutions"]):
            try:
                score, _ = code_compute_score(
                    sandbox_fusion_url=SANDBOX_URL,
                    concurrent_semaphore=_SEM,
                    memory_limit_mb=1024,
                    completion=f"```python\n{solution}\n```",
                    test_cases=ex["ground_truth"],
                    continuous=False,
                    timeout=VERIFY_TIMEOUT,
                )

                if score == 1.0:
                    return True

            except Exception:
                pass

        return False
    raw = load_dataset("likaixin/TACO-verified", split="train")
    print(f"Original TACO-Verified: {len(raw)}")
    raw = raw.filter(is_usable_stdin_problem)
    print(f"stdin-only with >= {MIN_TESTS} tests: {len(raw)}")
    raw = raw.map(make_tests)
    with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as pool:
        keep = list(pool.map(verify, raw))
    clean = raw.add_column("_keep", keep)
    clean = clean.filter(lambda ex: ex["_keep"])
    print(f"Verified under SandboxFusion: {len(clean)} / {len(raw)}")
    print("Difficulty distribution:", Counter(clean["difficulty"]))
    keep_columns = [
        "id",
        "question",
        "starter_code",
        "difficulty",
        "source",
        "ground_truth",
    ]
    clean = clean.remove_columns([
        c for c in clean.column_names if c not in keep_columns
    ])
    clean.save_to_disk(str(OUT_DIR))
    print(f"Saved to: {OUT_DIR}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['biology','code','external_code','medical','mmlupro'], required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--source_dir', default=None, help='Optional local pinned biology source directory')
    parser.add_argument('--direct_code', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output = args.output or ('data/mmlupro717.jsonl' if args.task=='mmlupro' else f'data/{args.task}')
    globals()['prepare_' + args.task](args)

if __name__ == '__main__':
    main()
