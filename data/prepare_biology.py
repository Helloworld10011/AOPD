# Rebuild the fixed 322/128/50 biology split from pinned public source files.
import argparse, hashlib, json, urllib.request
from pathlib import Path
from datasets import Dataset

SOURCE = "https://raw.githubusercontent.com/Moenupa/SDPO-CL/d277934412bd703838f9225086abe4b1795103c8/datasets/sciknoweval/biology/"
PINS = {'train.json': '7ffbfe4a85c7386e2ca78abd3de82c84d339ecbf0beb9bff93f3e599ac6298f2', 'test.json': '0ffe8df8f33154ca6af90c75b278f8602d75376db79e295e16a29fc9a5da325a'}

def uid(row):
    return hashlib.sha256(' '.join(row['prompt'].split()).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/biology')
    parser.add_argument('--source_dir',help='Optional existing directory containing train.json and test.json')
    args=parser.parse_args(); dest=Path(args.output)
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

if __name__=='__main__': main()
