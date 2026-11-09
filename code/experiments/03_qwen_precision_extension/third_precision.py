#!/usr/bin/env python3
"""Third-precision arm against a frozen Qwen3Guard BF16/NF4 pilot.
Does not rerun or modify BF16 and NF4. Refuses mismatched selection/metadata.
"""
import argparse, collections, gc, hashlib, json, os, re, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LABEL_RE = re.compile(r'Safety:\s*(Safe|Unsafe|Controversial)\b', re.I)

def read(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def write(path, value):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(p)

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

def rates(recs):
    ok=[r for r in recs if r['binary'] != 'invalid']
    bad=[r for r in recs if r['binary'] == 'invalid']
    unsafe=[r for r in ok if r['gold']=='unsafe']
    safe=[r for r in ok if r['gold']=='safe']
    fn=sum(r['binary']=='safe' for r in unsafe)
    fp=sum(r['binary']=='unsafe' for r in safe)
    return {'total':len(recs),'valid':len(ok),'invalid':len(bad),
        'accuracy':sum(r['gold']==r['binary'] for r in ok)/len(ok) if ok else None,
        'FN':fn,'unsafe_valid':len(unsafe),'FNR':fn/len(unsafe) if unsafe else None,
        'FP':fp,'safe_valid':len(safe),'FPR':fp/len(safe) if safe else None}

def compare(a,b):
    assert len(a)==len(b)
    flips=[]; improvements=0; regressions=0; unsafe_regressions=0; unsafe_recoveries=0
    for left,right in zip(a,b):
        assert left['id']==right['id'] and left['gold']==right['gold'], 'Paired ID/gold mismatch'
        if 'invalid' in (left['binary'],right['binary']): continue
        if left['binary'] != right['binary']:
            flips.append({'id':left['id'],'gold':left['gold'], 'from':left['binary'],'to':right['binary']})
            if left['binary'] != left['gold'] and right['binary']==right['gold']: improvements+=1
            if left['binary'] == left['gold'] and right['binary']!=right['gold']: regressions+=1
            if left['gold']=='unsafe' and left['binary']=='unsafe' and right['binary']=='safe': unsafe_regressions+=1
            if left['gold']=='unsafe' and left['binary']=='safe' and right['binary']=='unsafe': unsafe_recoveries+=1
    return {'flips':len(flips),'flip_rate':len(flips)/len(a),'improvements':improvements,
      'regressions':regressions,'unsafe_regressions':unsafe_regressions,
      'unsafe_recoveries':unsafe_recoveries,'flipped_ids':flips}

def load_third(model_id, arm, token):
    kwargs={'device_map':{'':'cuda:0'}, 'low_cpu_mem_usage':True,'dtype':torch.bfloat16}
    if token: kwargs['token']=token
    if arm=='fp8':
        try:
            from transformers import FineGrainedFP8Config
        except ImportError as e:
            raise RuntimeError('FineGrainedFP8Config not available: update transformers or explicitly run --precision int8') from e
        kwargs['quantization_config']=FineGrainedFP8Config()
    elif arm=='int8':
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_8bit=True)
    else: raise ValueError(arm)
    model=AutoModelForCausalLM.from_pretrained(model_id,**kwargs).eval()
    if arm=='int8':
        from bitsandbytes.nn import Linear8bitLt
        count=sum(isinstance(m,Linear8bitLt) for m in model.modules())
        if count==0: raise RuntimeError('No bitsandbytes INT8 layers found; refusing mislabeled precision run')
        print('Verified INT8 layer count:',count,flush=True)
        verification={'int8_layers':count}
    else:
        # FP8 implementation is explicitly recorded, rather than claiming native NVFP4.
        names=collections.Counter(type(m).__name__ for m in model.modules())
        candidates={name:num for name,num in names.items() if 'fp8' in name.lower() or 'finegrained' in name.lower()}
        if not candidates: raise RuntimeError('No FP8-specific layers found; refusing mislabeled precision run')
        print('Verified FP8 layer class names:',candidates,flush=True)
        verification={'fp8_layer_classes':candidates}
    return model,verification

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',required=True,help='Path to ORIGINAL completed pilot636 directory')
    ap.add_argument('--precision',required=True,choices=['fp8','int8'])
    ap.add_argument('--max-new-tokens',type=int,default=None,help='Must equal original manifest value')
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    pilot=Path(args.pilot).resolve()
    manifest=read(pilot/'manifest.json'); selection=read(pilot/'selection.json')
    bf=read(pilot/'bf16.json'); nf=read(pilot/'nf4.json')
    n=manifest['n']; model_id=manifest['model']
    if n!=len(selection): raise RuntimeError('Selection/manifest length mismatch')
    if sha(selection)!=manifest['selection_sha256']:
        raise RuntimeError('Original selection hash mismatch; do not proceed')
    if manifest.get('dataset')!='Qwen/Qwen3GuardTest' or manifest.get('split')!='thinking':
        raise RuntimeError('Unexpected source dataset/split')
    if not bf.get('finished') or not nf.get('finished'):
        raise RuntimeError('Reference predictions are not finished')
    ref_ids=[x['id'] for x in selection]
    if len(set(ref_ids))!=n: raise RuntimeError('Duplicate IDs in selection')
    for src in (bf,nf):
        records=src['records']
        if len(records)!=n or [x['id'] for x in records]!=ref_ids:
            raise RuntimeError('Reference record IDs/order mismatch')
        if [x['gold'] for x in records]!=[x['label'] for x in selection]:
            raise RuntimeError('Reference gold-label mismatch')
    if args.max_new_tokens is not None and args.max_new_tokens!=manifest['max_new_tokens']:
        raise RuntimeError('Cannot change max-new-tokens from original BF16/NF4 run')
    token=os.environ.get('HF_TOKEN') or None
    kw={'token':token} if token else {}
    tok=AutoTokenizer.from_pretrained(model_id, **kw)
    ds=load_dataset(manifest['dataset'],split=manifest['split'])
    by_id={str(x['unique_id']):x for x in ds}
    if len(by_id)!=len(ds): raise RuntimeError('Dataset contains duplicate IDs')
    selected=[]
    for x in selection:
        if x['id'] not in by_id: raise RuntimeError('Frozen ID missing from dataset: '+x['id'])
        d=by_id[x['id']]
        if str(d['label']).lower()!=x['label']: raise RuntimeError('Frozen label changed: '+x['id'])
        prompt=tok.apply_chat_template(d['message'], tokenize=False)
        length=len(tok(prompt, add_special_tokens=False)['input_ids'])
        if length != x['tokens'] or length>manifest['max_input_tokens']:
            raise RuntimeError('Tokenization/config changed for '+x['id'])
        selected.append((x,d))
    out=pilot/f'{args.precision}.json'
    if out.exists():
        saved=read(out)
        if saved.get('finished') and len(saved.get('records',[]))==n and [r['id'] for r in saved['records']]==ref_ids:
            print('Using completed third arm:',out,flush=True)
            recs=saved['records']
        else:
            raise RuntimeError('Incomplete third-arm output exists. Preserve it; do not silently overwrite: '+str(out))
    else:
        model,verify=load_third(model_id,args.precision,token)
        started=time.monotonic(); recs=[]
        for i,(s,d) in enumerate(selected):
            prompt=tok.apply_chat_template(d['message'], tokenize=False)
            encoded=tok(prompt,return_tensors='pt',add_special_tokens=False,truncation=False).to('cuda:0')
            with torch.inference_mode():
                generated=model.generate(**encoded,max_new_tokens=manifest['max_new_tokens'],do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            raw=tok.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
            m=LABEL_RE.search(raw)
            pred=m.group(1).lower() if m else 'invalid'
            binary='safe' if pred=='safe' else ('unsafe' if pred in ('unsafe','controversial') else 'invalid')
            recs.append({'id':s['id'],'gold':s['label'],'prediction':pred,'binary':binary,
              'source':s['source'],'unsafe_type':s['unsafe_type'],'tokens':s['tokens'],'output':raw})
            if (i+1)%20==0 or i+1==n:
                write(out,{'arm':args.precision,'model':model_id,'finished':False,'records':recs,
                           'verification':verify,'selection_sha256':manifest['selection_sha256']})
                print(f'{args.precision} {i+1}/{n} elapsed={time.monotonic()-started:.0f}s',flush=True)
        write(out,{'arm':args.precision,'model':model_id,'finished':True,'records':recs,
             'verification':verify,'selection_sha256':manifest['selection_sha256'],
             'elapsed_seconds':time.monotonic()-started})
        del model; gc.collect(); torch.cuda.empty_cache()
    result={'pilot':str(pilot),'n':n,'model':model_id,'third_arm':args.precision,
      'frozen_selection_sha256':manifest['selection_sha256'],
      'reference_unchanged':True,
      'rates':{'bf16':rates(bf['records']),'nf4':rates(nf['records']),args.precision:rates(recs)},
      'comparisons':{'bf16_vs_nf4':compare(bf['records'],nf['records']),
                     'bf16_vs_'+args.precision:compare(bf['records'],recs),
                     'nf4_vs_'+args.precision:compare(nf['records'],recs)},
      'limitations':'One checkpoint and dataset; exploratory; not a novelty proof; FP8 and INT8 are distinct formats.'}
    report=pilot/f'third_precision_{args.precision}_summary.json'
    write(report,result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    print('PAIR COUNTS:')
    for key,val in result['comparisons'].items():
        print(key,{k:v for k,v in val.items() if k!='flipped_ids'})
    print('Saved report:',report)

if __name__=='__main__':
    main()
