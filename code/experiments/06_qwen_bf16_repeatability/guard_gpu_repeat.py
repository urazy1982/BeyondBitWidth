#!/usr/bin/env python3
"""Non-destructive GPU repeatability check for frozen Guard evidence.

Runs fresh BF16 A and B inference with IDENTICAL settings on one stratified,
fixed subset per dataset, and compares them to each other and the archived BF16.
Existing dataset predictions and selections are read-only; outputs go elsewhere.
"""
import argparse, collections, hashlib, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

DATASETS=('qwen_thinking','toxicchat_prompt','xstest_prompt','beavertails_response')

def canonical(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str)

def sha(x): return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    tmp.replace(path)

def select(rows,per_class,seed):
    rng=random.Random(seed)
    picked=[]
    for label in ('safe','unsafe'):
        eligible=[r for r in rows if r['gold']==label]
        chosen=rng.sample(eligible,min(per_class,len(eligible)))
        picked+=chosen
    # Source order avoids accidental differences in execution context.
    return sorted(picked,key=lambda r:r['id'])

def compare(records_a,records_b):
    a={str(r['id']):r for r in records_a};b={str(r['id']):r for r in records_b}
    if set(a)!=set(b):raise ValueError('Mismatched record ID sets')
    changes=[]; invalid=0
    for ident in sorted(a):
        x,y=a[ident],b[ident]
        if x['gold']!=y['gold']:raise ValueError('Gold label mismatch')
        if x['binary']=='invalid' or y['binary']=='invalid':invalid+=1
        if x['binary']!=y['binary']:
            changes.append({'id':ident,'gold':x['gold'],'from':x['binary'],'to':y['binary']})
    return {'paired':len(a),'changed':len(changes),'changed_rate':len(changes)/len(a),
            'invalid_pairs':invalid,'changed_ids':changes}

def run(args):
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('No GPU detected; GPU-only control cannot run on CPU')
    project=Path(args.project).resolve(); root=project/'five_results';out=Path(args.out).resolve()
    if out==root or root in out.parents:raise ValueError('Output must be outside five_results; never alter archived evidence')
    if out.exists() and any(out.glob('*/selection.json')):pass
    token=os.environ.get('HF_TOKEN') or None
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3Guard-Gen-4B',token=token)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3Guard-Gen-4B',token=token,
      device_map={'':'cuda:0'},dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
    model_rev=getattr(model.config,'_commit_hash',None)
    tokenizer_rev=getattr(tok,'_commit_hash',None) or (tok.init_kwargs or {}).get('_commit_hash')
    from run_five import LABEL_RE
    env={'utc':datetime.now(timezone.utc).isoformat(),'model':'Qwen/Qwen3Guard-Gen-4B',
      'model_commit':model_rev,'tokenizer_commit':tokenizer_rev,'torch':torch.__version__,
      'transformers':transformers.__version__,'cuda':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),'max_new_tokens':'from frozen manifest',
      'do_sample':False,'padding':'same single-item batch in every run',
      'cudnn_benchmark':torch.backends.cudnn.benchmark,
      'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
      'per_class':args.per_class,'selection_seed':args.seed}
    dump(out/'environment.json',env)
    all_results=[]
    for name in DATASETS:
        frozen=root/name;man=load(frozen/'manifest.json');rows=load(frozen/'selection.json')
        if sha(rows)!=man['selection_sha256']:raise RuntimeError(name+': frozen selection SHA mismatch')
        if man['model']!='Qwen/Qwen3Guard-Gen-4B':raise RuntimeError('Unexpected model in manifest')
        subset=select(rows,args.per_class,args.seed)
        d=out/name;d.mkdir(parents=True,exist_ok=True)
        meta={'dataset':name,'frozen_sha256':man['selection_sha256'],
          'selected_ids':[r['id'] for r in subset], 'selected_sha256':sha(subset),
          'n':len(subset),'source_split':man['source'].get('split'),
          'safe':sum(r['gold']=='safe' for r in subset),'unsafe':sum(r['gold']=='unsafe' for r in subset)}
        pmeta=d/'subset_manifest.json'
        if pmeta.exists() and load(pmeta)!=meta:raise RuntimeError('Prior repeatability subset mismatch')
        dump(pmeta,meta)
        orig=load(frozen/'bf16.json')
        if not orig.get('finished'):raise RuntimeError(name+': baseline BF16 not complete')
        mapped={str(r['id']):r for r in orig['records']}
        if len(mapped)!=len(rows):raise RuntimeError(name+': BF16 baseline length differs')
        ref=[mapped[str(r['id'])] for r in subset]
        results={}
        for rep in ('A','B'):
            dest=d/f'bf16_repeat_{rep}.json'
            if dest.exists():
                existing=load(dest)
                if existing.get('selected_sha256')!=meta['selected_sha256']:raise RuntimeError('Run selection mismatch')
                records=existing.get('records',[])
                if len(records)>len(subset) or any(records[i]['id']!=subset[i]['id'] or records[i]['gold']!=subset[i]['gold'] for i in range(len(records))):
                    raise RuntimeError('Saved repeat ID mismatch')
            else: records=[]
            if len(records)==len(subset) and dest.exists() and load(dest).get('finished'):
                print(name,rep,'complete; reusing saved repeat',flush=True)
                results[rep]=records;continue
            start=time.monotonic()
            for i in range(len(records),len(subset)):
                row=subset[i]
                prompt=tok.apply_chat_template(row['messages'],tokenize=False)
                if hashlib.sha256(prompt.encode('utf-8')).hexdigest()!=row['prompt_hash']:
                    raise RuntimeError(name+'/'+str(row['id'])+': tokenizer/template drift')
                encoded=tok(prompt,add_special_tokens=False,return_tensors='pt',truncation=False).to('cuda:0')
                if encoded.input_ids.shape[-1]>man['max_input_tokens']:
                    raise RuntimeError('Exceeded frozen token budget')
                with torch.inference_mode():
                    generated=model.generate(**encoded,max_new_tokens=man['max_new_tokens'],
                      do_sample=False,pad_token_id=tok.eos_token_id)
                raw=tok.decode(generated[0,encoded.input_ids.shape[1]:],skip_special_tokens=True)
                m=LABEL_RE.search(raw);pred=m.group(1).lower() if m else 'invalid'
                binary='safe' if pred=='safe' else 'unsafe' if pred in ('unsafe','controversial') else 'invalid'
                records.append({'id':row['id'],'gold':row['gold'],'binary':binary,'prediction':pred,
                  'output':raw,'category':row['category']})
                if (i+1)%20==0 or i+1==len(subset):
                    dump(dest,{'dataset':name,'repeat':rep,'selected_sha256':meta['selected_sha256'],
                      'finished':False,'records':records,'model_commit':model_rev})
                    print(name,rep,f'{i+1}/{len(subset)}',f'{time.monotonic()-start:.0f}s',flush=True)
            doc=load(dest);doc['finished']=True;doc['elapsed_seconds_this_session']=time.monotonic()-start
            dump(dest,doc)
            results[rep]=records
        stats={'dataset':name,'n':len(subset),'bf16_A_vs_B':compare(results['A'],results['B']),
          'original_vs_repeat_A':compare(ref,results['A']),
          'original_vs_repeat_B':compare(ref,results['B']),
          'same_loaded_model_for_A_B':True,
          'caveat':'Original BF16 model commit was not frozen; difference from historical BF16 may include model version drift.'}
        dump(d/'repeatability_summary.json',stats)
        all_results.append(stats)
    dump(out/'repeatability_cross_dataset.json',{'environment':env,'datasets':all_results})
    print('COMPLETE: '+str(out/'repeatability_cross_dataset.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',default=str(Path.home()/'guard-study/guard_five_datasets'))
    p.add_argument('--out',default=str(Path.home()/'guard-study/guard_repeatability_results'))
    p.add_argument('--per-class',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261108)
    a=p.parse_args()
    if a.per_class<1: p.error('--per-class must be positive')
    run(a)
