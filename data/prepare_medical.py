# Build MedQA test and MedMCQA validation prompts using the original released medical wrapper.
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

SPECS = [
 ('medqa_test','bigbio/med_qa','484a6c066fe8e75c83edea0c88b5169316714fcd','med_qa_en_4options_source/test-00000-of-00001.parquet',1273,'b50b49fad2642c10ed343c6174d5ab0cb8efad434e7b66465ac471094eb57deb'),
 ('medmcqa_validation','openlifescienceai/medmcqa','91c6572c454088bf71b679ad90aa8dffcd0d5868','data/validation-00000-of-00001.parquet',4183,'b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021'),
]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='data/medical');args=parser.parse_args()
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

if __name__=='__main__':main()
