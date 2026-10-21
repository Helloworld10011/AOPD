# Build the original official five-shot biology prompts for 717 MMLU-Pro test questions.
import argparse, hashlib, json
from pathlib import Path
from datasets import load_dataset
from evaluation.mmlupro_helpers import preprocess, select_by_category, format_cot_example

REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
INITIAL = 'The following are multiple choice questions (with answers) about biology. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='data/mmlupro717.jsonl');args=parser.parse_args()
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

if __name__=='__main__':main()
