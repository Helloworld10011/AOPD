# Evaluate frozen prompt/gold JSONL with the original medical, biology, or MMLU-Pro parser.
import argparse, hashlib, json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--task',choices=['medical','biology','mmlupro'],required=True)
    parser.add_argument('--revision',default=None)
    parser.add_argument('--tp',type=int,default=1)
    parser.add_argument('--max_tokens',type=int,default=None)
    args=parser.parse_args()
    dest=Path(args.output)
    if dest.exists():raise FileExistsError(f'Do not overwrite an evaluation: {dest}')
    from vllm import LLM, SamplingParams
    from evaluation.qa_graders import medical_answer, biology_answer
    from evaluation.mmlupro_helpers import extract_answer as mmlupro_answer
    grader={'medical':medical_answer,'biology':biology_answer,'mmlupro':mmlupro_answer}[args.task]
    cap=args.max_tokens or (8192 if args.task=='medical' else 4096)
    context=20000 if args.task=='medical' else 16384
    source=Path(args.data);rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert rows and len({x['uid'] for x in rows})==len(rows)
    llm=LLM(model=args.model,revision=args.revision,dtype='bfloat16',tensor_parallel_size=args.tp,max_model_len=context,seed=7319)
    tokenizer=llm.get_tokenizer()
    prompts=[tokenizer.apply_chat_template(x['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in rows]
    if any(len(tokenizer.encode(p,add_special_tokens=False))+cap>context for p in prompts):
        raise ValueError('A prompt exceeds the fixed context budget; do not silently truncate it.')
    parameters=[SamplingParams(n=1,temperature=.6 if args.task=='medical' else 0.,top_p=.95 if args.task=='medical' else 1.,top_k=20 if args.task=='medical' else -1,min_p=0.,max_tokens=cap,seed=42+i if args.task=='medical' else 7319,stop=['Question:'] if args.task=='mmlupro' else []) for i in range(len(rows))]
    outputs=llm.generate(prompts,parameters)
    assert len(outputs)==len(rows)
    dest.parent.mkdir(parents=True,exist_ok=True)
    correct=truncated=invalid=0
    with dest.open('x') as stream:
        for row,prompt,out in zip(rows,prompts,outputs):
            assert len(out.outputs)==1
            result=out.outputs[0];prediction=grader(result.text);ok=prediction==row['gold']
            record={**row,'prediction':prediction,'correct':bool(ok),'completion':result.text,'token_ids':list(result.token_ids),'finish_reason':result.finish_reason,'tokens':len(result.token_ids),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            correct+=ok;invalid+=prediction is None;truncated+=result.finish_reason=='length'
    summary={'questions':len(rows),'correct':correct,'accuracy':correct/len(rows),'invalid':invalid,'truncated':truncated,'task':args.task,'model':args.model,'revision':args.revision,'max_tokens':cap,'thinking':False,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    dest.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
