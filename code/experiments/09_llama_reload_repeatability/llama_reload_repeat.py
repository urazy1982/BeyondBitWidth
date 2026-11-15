#!/usr/bin/env python3
"""Llama Guard 3 independent-reload NF4/FP4 confirmation; no original files overwritten.

Requires the prior replicate.py script next to this file. The experiment pins the
original model revision and frozen per-dataset selections, and records two fresh
independent model loads PER precision arm.
"""
import argparse
import collections
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import replicate as r

DATASETS = ('xstest_prompt', 'beavertails_response')
ARMS = ('nf4', 'fp4')
REPS = ('A', 'B')

def file_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def checked_records(path, rows, frozen_sha, arm, revision):
    obj=r.load(path)
    if not obj.get('finished') or obj.get('arm')!=arm:
        raise RuntimeError(f'Original prediction file unfinished/wrong arm: {path}')
    if obj.get('model_revision')!=revision or obj.get('selection_sha256')!=frozen_sha:
        raise RuntimeError(f'Original prediction metadata differs: {path}')
    records=obj['records']
    if len(records)!=len(rows):raise RuntimeError(f'Record count mismatch: {path}')
    for x,y in zip(records,rows):
        if str(x['id'])!=str(y['id']) or x['gold']!=y['gold']:
            raise RuntimeError(f'Original record misalignment: {path}')
        if x['binary'] not in ('safe','unsafe'):
            raise RuntimeError(f'Invalid original binary prediction: {path} / {x["id"]}')
    return records

def preflight(src):
    env=r.load(src/'environment.json')
    revision=env.get('hf_revision')
    if not revision or len(revision)!=40:raise RuntimeError('Original fixed model revision absent')
    if env.get('model')!=r.MODEL:raise RuntimeError('Original model differs')
    max_input=env['max_input_tokens'];max_new=env['max_new_tokens']
    frozen={}; originals={}; evidence={}
    evidence['environment.json']=file_digest(src/'environment.json')
    for ds in DATASETS:
        dp=src/ds
        select_path=dp/'llama_selection.json'; manifest_path=dp/'llama_manifest.json'
        rows=r.load(select_path);man=r.load(manifest_path)
        if r.sha(rows)!=man['selection_sha256'] or len(rows)!=man['n']:
            raise RuntimeError(f'{ds}: frozen selection integrity mismatch')
        if len({str(x['id']) for x in rows})!=len(rows):
            raise RuntimeError(f'{ds}: duplicate source IDs')
        evidence[str(select_path.relative_to(src))]=file_digest(select_path)
        evidence[str(manifest_path.relative_to(src))]=file_digest(manifest_path)
        originals[ds]={}
        for arm in ARMS:
            p=dp/f'{arm}.json'
            originals[ds][arm]=checked_records(p,rows,man['selection_sha256'],arm,revision)
            evidence[str(p.relative_to(src))]=file_digest(p)
        frozen[ds]=(rows,man)
    return env,revision,max_input,max_new,frozen,originals,evidence

def compare(a,b):
    p=r.paired(a,b)
    return {'paired':p['paired'],'changed':p['flips'],
            'changed_rate':p['flip_rate'],'improvements':p['improvements'],
            'regressions':p['regressions'],
            'unsafe_regressions':p['unsafe_regressions'],
            'unsafe_recoveries':p['unsafe_recoveries'],
            'changed_ids':p['flipped_ids']}

def exact_binomial_two_sided(k,n):
    import math
    if n==0:return 1.0
    lo=min(k,n-k)
    return min(1.,2*sum(math.comb(n,i) for i in range(lo+1))/2**n)

def summarize(src,out,originals,frozen,env,revision,evidence):
    summary={'experiment':'Llama Guard 3 independent model reload repeatability',
             'original_model_revision':revision,
             'original_environment':env,
             'original_input_sha256':evidence,
             'design':{'arms':list(ARMS),'repeats':list(REPS),
                       'independent_model_reload_per_arm_repeat':True,
                       'datasets':list(DATASETS),'selection':'entire original frozen selection',
                       'limitations':['Validation on exactly these frozen samples; not a new holdout.',
                                      'Independent reloads performed on same GPU/software environment.',
                                      'Format effects remain implementation-specific; no logit-level attribution.']},
             'datasets':[]}
    for ds in DATASETS:
        rows,man=frozen[ds];d={'dataset':ds,'n':len(rows),'selection_sha256':man['selection_sha256'],'comparisons':{}}
        rec={};details={}
        for arm in ARMS:
            rec[(arm,'original')]=originals[ds][arm]
            for rep in REPS:
                p=out/ds/f'{arm}_{rep}.json'
                if not p.exists():raise RuntimeError(f'Incomplete arm-repetition file: {p}')
                rec[(arm,rep)]=checked_records(p,rows,man['selection_sha256'],arm,revision)
                details[f'{arm}_{rep}']= {'sha256':file_digest(p), 'n':len(rec[(arm,rep)])}
            d['comparisons'][f'{arm}_A_vs_B']=compare(rec[(arm,'A')],rec[(arm,'B')])
            for rep in REPS:
                d['comparisons'][f'original_{arm}_vs_{rep}']=compare(rec[(arm,'original')],rec[(arm,rep)])
        for rep in ('original',)+REPS:
            c=compare(rec[('nf4',rep)],rec[('fp4',rep)])
            c['mcnemar_accuracy_p_exact_two_sided']=exact_binomial_two_sided(c['improvements'],c['improvements']+c['regressions'])
            c['mcnemar_fn_p_exact_two_sided']=exact_binomial_two_sided(c['unsafe_recoveries'],c['unsafe_recoveries']+c['unsafe_regressions'])
            d['comparisons'][f'nf4_vs_fp4_{rep}']=c
        d['files']=details;summary['datasets'].append(d)
    r.save(out/'llama_reload_cross_dataset_summary.json',summary)
    print('REPORT:',out/'llama_reload_cross_dataset_summary.json',flush=True)
    for d in summary['datasets']:
        print(d['dataset'],'n=',d['n'],
              'nf4 A/B=',d['comparisons']['nf4_A_vs_B']['changed'],
              'fp4 A/B=',d['comparisons']['fp4_A_vs_B']['changed'],
              'NF4/FP4 original,A,B=',*[d['comparisons'][f'nf4_vs_fp4_{rep}']['changed'] for rep in ('original','A','B')],flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src',type=Path,default=Path.home()/'guard-study/llama_guard_replication_results')
    ap.add_argument('--out',type=Path,default=Path.home()/'guard-study/llama_guard_reload_repeatability')
    ap.add_argument('--report-only',action='store_true')
    args=ap.parse_args();src=args.src.expanduser().resolve();out=args.out.expanduser().resolve()
    if src==out or src in out.parents or out in src.parents:raise RuntimeError('Source/output directories must be separate')
    env,revision,max_input,max_new,frozen,originals,evidence=preflight(src)
    print('Original Llama checkpoint:',revision,'samples:',{x:len(y[0]) for x,y in frozen.items()},flush=True)
    if not args.report_only:
        import torch,transformers,bitsandbytes
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
        token=os.environ.get('HF_TOKEN') or None
        tokenizer=AutoTokenizer.from_pretrained(r.MODEL,token=token,revision=revision)
        # Tokenizer revision pinned; compare reconstructed input lengths before costly inference.
        for ds,(rows,man) in frozen.items():
            for row in rows:
                n=len(tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False))
                if n>max_input or n!=row['llama_input_tokens']:
                    raise RuntimeError(f'{ds}/{row["id"]}: tokenization drift ({n})')
        run_env={'model':r.MODEL,'revision':revision,'torch':torch.__version__,
                 'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,
                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                 'max_input_tokens':max_input,'max_new_tokens':max_new,
                 'original_sha256':evidence,'created_utc':r.utc()}
        cfg=out/'reload_environment.json'
        if cfg.exists():
            old=r.load(cfg)
            for field in ('model','revision','torch','transformers','bitsandbytes','cuda','gpu','max_input_tokens','max_new_tokens','original_sha256'):
                if old[field]!=run_env[field]:raise RuntimeError(f'Reload environment changed: {field}')
        else:r.save(cfg,run_env)
        for arm in ARMS:
            for rep in REPS:
                work={};last_count=0
                for ds,(rows,man) in frozen.items():
                    path=out/ds/f'{arm}_{rep}.json';records=[]
                    if path.exists():
                        obj=r.load(path)
                        if obj['arm']!=arm or obj['repeat']!=rep or obj['selection_sha256']!=man['selection_sha256'] or obj['model_revision']!=revision:
                            raise RuntimeError(f'Existing checkpoint metadata mismatch: {path}')
                        records=obj['records']
                        if len(records)>len(rows):raise RuntimeError('Checkpoint longer than selection')
                        for i,x in enumerate(records):
                            if str(x['id'])!=str(rows[i]['id']) or x['gold']!=rows[i]['gold'] or x['binary'] not in ('safe','unsafe'):
                                raise RuntimeError(f'Checkpoint data corruption: {path} index {i}')
                        if obj.get('finished') and len(records)==len(rows):
                            print('SKIP completed:',ds,arm,rep,flush=True);continue
                    work[ds]=(rows,man,path,records)
                if not work:continue
                print('INDEPENDENT MODEL LOAD:',arm,rep,flush=True)
                model,details=r.model_load(arm,token,revision)
                r.save(out/f'{arm}_{rep}_model_provenance.json',details)
                for ds,(rows,man,path,records) in work.items():
                    start=time.monotonic()
                    for i in range(len(records),len(rows)):
                        result=r.prediction(tokenizer,model,rows[i],max_input,max_new)
                        if result['binary'] not in ('safe','unsafe'):
                            raise RuntimeError(f'Invalid prediction on {ds}/{arm}/{rep} at {i}; please inspect and halt')
                        records.append(result)
                        if (i+1)%20==0 or (i+1)==len(rows):
                            obj={'dataset':ds,'arm':arm,'repeat':rep,
                                 'selection_sha256':man['selection_sha256'],
                                 'model_revision':revision,'finished':False,'records':records,'updated_utc':r.utc()}
                            r.save(path,obj)
                            print(f'{ds} {arm}_{rep} {i+1}/{len(rows)} elapsed={time.monotonic()-start:.0f}s',flush=True)
                    obj=r.load(path);obj['finished']=True;r.save(path,obj)
                del model;gc.collect();torch.cuda.empty_cache()
    summarize(src,out,originals,frozen,env,revision,evidence)

if __name__=='__main__':main()
