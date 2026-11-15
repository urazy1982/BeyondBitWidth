#!/usr/bin/env python3
"""Independent-reload NF4/FP4 repeatability on immutable selections."""
import argparse, gc, hashlib, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

DATASETS=('qwen_thinking','toxicchat_prompt','xstest_prompt','beavertails_response')
MODEL='Qwen/Qwen3Guard-Gen-4B'

def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str)
def sha(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def subset_select(rows,per_class,seed):
    rng=random.Random(seed);selected=[]
    for label in ('safe','unsafe'):
        matches=[r for r in rows if r['gold']==label]
        selected+=rng.sample(matches,min(per_class,len(matches)))
    return sorted(selected,key=lambda r:str(r['id']))
def compare(A,B):
    aa={str(x['id']):x for x in A};bb={str(x['id']):x for x in B}
    if len(aa)!=len(A) or len(bb)!=len(B) or set(aa)!=set(bb):raise ValueError('ID set or duplicates mismatch')
    changes=[]
    for key in sorted(aa):
        x,y=aa[key],bb[key]
        if x['gold']!=y['gold']:raise ValueError('Gold mismatch')
        if x['binary']!=y['binary']:
            changes.append({'id':key,'gold':x['gold'],'from':x['binary'],'to':y['binary']})
    return {'paired':len(aa),'changed':len(changes),'changed_rate':len(changes)/len(aa),
            'invalid_pairs':sum(aa[k]['binary']=='invalid' or bb[k]['binary']=='invalid' for k in aa),
            'changed_ids':changes}
def validate(project,out,per_class,seed):
    frozen=project/'five_results';sets={}; manifests={}
    if out==frozen or frozen in out.parents:raise RuntimeError('Output cannot be within original five_results')
    for name in DATASETS:
        folder=frozen/name;rows=read(folder/'selection.json');man=read(folder/'manifest.json')
        if man['selection_sha256']!=sha(rows) or len(rows)!=man['n'] or man['model']!=MODEL:raise RuntimeError(f'{name}: invalid frozen manifest')
        if len({str(x['id']) for x in rows})!=len(rows):raise RuntimeError('Duplicate frozen IDs')
        for fmt in ('nf4','fp4'):
            d=read(folder/f'{fmt}.json')
            if not d.get('finished') or d.get('selection_sha256')!=man['selection_sha256'] or len(d['records'])!=len(rows):
                raise RuntimeError(f'{name}: reference {fmt} incomplete or mismatch')
            if any(str(r['id'])!=str(d['records'][i]['id']) or r['gold']!=d['records'][i]['gold'] for i,r in enumerate(rows)):
                raise RuntimeError(f'{name}: {fmt} reference alignment mismatch')
        sub=subset_select(rows,per_class,seed); meta={'dataset':name,'selected_ids':[str(r['id']) for r in sub],
          'n':len(sub),'selected_sha256':sha(sub),'frozen_sha256':man['selection_sha256'],
          'safe':sum(r['gold']=='safe' for r in sub),'unsafe':sum(r['gold']=='unsafe' for r in sub),
          'per_class':per_class,'seed':seed}
        saved=out/name/'subset_manifest.json'
        if saved.exists() and read(saved)!=meta:raise RuntimeError(f'{name}: repeat subset mismatch')
        write(saved,meta);sets[name]=(sub,meta);manifests[name]=man
    return sets,manifests

def main(args):
    import torch,transformers,bitsandbytes
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from run_five import LABEL_RE
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available: refusing CPU fallback')
    project=Path(args.project).resolve();out=Path(args.out).resolve();sets,mans=validate(project,out,args.per_class,args.seed)
    token=os.environ.get('HF_TOKEN') or None
    tok=AutoTokenizer.from_pretrained(MODEL,token=token)
    tokenizer_commit=getattr(tok,'_commit_hash',None) or (tok.init_kwargs or {}).get('_commit_hash')
    env={'model':MODEL,'tokenizer_commit':tokenizer_commit,'gpu':torch.cuda.get_device_name(0),
         'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,
         'bitsandbytes':getattr(bitsandbytes,'__version__',None),'seed':args.seed,'per_class':args.per_class,
         'do_sample':False,'batch_size':1,'cudnn_benchmark':torch.backends.cudnn.benchmark,
         'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
         'run_started_utc':datetime.now(timezone.utc).isoformat(),
         'design':'Independent model reload per (format,repeat)'}
    write(out/'environment.json',env)
    # Separate loads; never reuse an existing finished repeat as a new independent trial.
    for fmt in ('nf4','fp4'):
      for rep in ('A','B'):
        targets={}
        for name,(rows,meta) in sets.items():
            path=out/name/f'{fmt}_repeat_{rep}.json'
            if path.exists():
                data=read(path)
                if data.get('selected_sha256')!=meta['selected_sha256'] or data.get('format')!=fmt or data.get('repeat')!=rep:
                    raise RuntimeError('Incompatible repeat checkpoint '+str(path))
                recs=data.get('records',[])
                if len(recs)>len(rows) or any(str(recs[i]['id'])!=str(rows[i]['id']) or recs[i]['gold']!=rows[i]['gold'] for i in range(len(recs))):
                    raise RuntimeError('Saved record alignment failure '+str(path))
                if data.get('finished') and len(recs)==len(rows):
                    print(name,fmt,rep,'already completed; preserving',flush=True)
                    continue
            else: recs=[]
            targets[name]=(rows,meta,path,recs)
        if not targets:continue
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type=fmt,
                                  bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,token=token,device_map={'':'cuda:0'},
                       dtype=torch.bfloat16,low_cpu_mem_usage=True,quantization_config=config).eval()
        layers=[m for m in model.modules() if type(m).__name__=='Linear4bit']
        if len(layers)<1:raise RuntimeError(f'{fmt}: zero 4-bit linear layers')
        actual={str(getattr(getattr(m,'weight',None),'quant_type','')).lower() for m in layers}
        if actual and all(x not in ('',fmt) for x in actual):raise RuntimeError(f'{fmt}: layer format mismatch {actual}')
        revision=getattr(model.config,'_commit_hash',None)
        details={'format':fmt,'revision':revision,'tokenizer_revision':tokenizer_commit,
                 'linear4bit_count':len(layers),'quant_type_seen':sorted(actual),'config':str(config),
                 'load_time_utc':datetime.now(timezone.utc).isoformat()}
        write(out/f'{fmt}_{rep}_load_provenance.json',details)
        for name,(rows,meta,path,recs) in targets.items():
            start=time.monotonic();man=mans[name]
            for i in range(len(recs),len(rows)):
                row=rows[i]
                prompt=tok.apply_chat_template(row['messages'],tokenize=False)
                if hashlib.sha256(prompt.encode()).hexdigest()!=row['prompt_hash']:raise RuntimeError(f'{name}/{row["id"]}: prompt hash mismatch')
                inp=tok(prompt,add_special_tokens=False,return_tensors='pt',truncation=False).to('cuda:0')
                if inp.input_ids.shape[-1]>man['max_input_tokens']:raise RuntimeError('Token limit mismatch')
                with torch.inference_mode():
                    generated=model.generate(**inp,max_new_tokens=man['max_new_tokens'],do_sample=False,pad_token_id=tok.eos_token_id)
                raw=tok.decode(generated[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
                m=LABEL_RE.search(raw);label=m.group(1).lower() if m else 'invalid'
                binary='safe' if label=='safe' else 'unsafe' if label in ('unsafe','controversial') else 'invalid'
                recs.append({'id':row['id'],'gold':row['gold'],'binary':binary,'prediction':label,
                             'output':raw,'category':row.get('category',''),'subgroup':row.get('subgroup','')})
                if (i+1)%20==0 or i+1==len(rows):
                    write(path,{'dataset':name,'format':fmt,'repeat':rep,'selected_sha256':meta['selected_sha256'],
                      'finished':False,'records':recs,'model_revision':revision,'updated_utc':datetime.now(timezone.utc).isoformat()})
                    print(f'{name}/{fmt}/{rep} {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
            data=read(path);data['finished']=True;data['elapsed_seconds_this_session']=time.monotonic()-start;write(path,data)
        del model;gc.collect();torch.cuda.empty_cache()
    reports=[]
    for name,(rows,meta) in sets.items():
        original={fmt:read(project/'five_results'/name/f'{fmt}.json') for fmt in ('nf4','fp4')}
        index={fmt:{str(r['id']):r for r in original[fmt]['records']} for fmt in ('nf4','fp4')}
        baseline={fmt:[index[fmt][str(r['id'])] for r in rows] for fmt in ('nf4','fp4')}
        again={fmt:{rep:read(out/name/f'{fmt}_repeat_{rep}.json')['records'] for rep in ('A','B')} for fmt in ('nf4','fp4')}
        r={'dataset':name,'n':len(rows),'selection_sha256':meta['selected_sha256'],
          'nf4_A_vs_B':compare(again['nf4']['A'],again['nf4']['B']),
          'fp4_A_vs_B':compare(again['fp4']['A'],again['fp4']['B']),
          'original_nf4_vs_A':compare(baseline['nf4'],again['nf4']['A']),
          'original_nf4_vs_B':compare(baseline['nf4'],again['nf4']['B']),
          'original_fp4_vs_A':compare(baseline['fp4'],again['fp4']['A']),
          'original_fp4_vs_B':compare(baseline['fp4'],again['fp4']['B']),
          'fresh_nf4_A_vs_fp4_A':compare(again['nf4']['A'],again['fp4']['A']),
          'fresh_nf4_B_vs_fp4_B':compare(again['nf4']['B'],again['fp4']['B']),
          'original_nf4_vs_fp4':compare(baseline['nf4'],baseline['fp4'])}
        write(out/name/'quantized_repeatability_summary.json',r);reports.append(r)
    write(out/'quantized_repeatability_cross_dataset.json',{'environment':env,'datasets':reports})
    print('COMPLETE',str(out/'quantized_repeatability_cross_dataset.json'),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',default=str(Path.home()/'guard-study/guard_five_datasets'))
    ap.add_argument('--out',default=str(Path.home()/'guard-study/guard_quantized_repeatability_results'))
    ap.add_argument('--per-class',type=int,default=128);ap.add_argument('--seed',type=int,default=20261108)
    args=ap.parse_args();assert args.per_class>0
    main(args)
