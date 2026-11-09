#!/usr/bin/env python3
"""Paired BF16 vs NF4 Qwen3Guard pilot, one GPU, one frozen dataset sample."""
import argparse, gc, hashlib, json, os, re, sys, time
from pathlib import Path
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL='Qwen/Qwen3Guard-Gen-4B'
DATASET='Qwen/Qwen3GuardTest'
LABEL_RE=re.compile(r'Safety:\s*(Safe|Unsafe|Controversial)\b', re.I)

def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

def save_json(path, obj):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str), encoding='utf-8')
    tmp.replace(path)

def get_data(tok, n, seed, max_input_tokens):
    ds=load_dataset(DATASET, split='thinking')
    rows=[]
    for r in ds:
        lab=str(r['label']).lower()
        if lab not in ('safe','unsafe'): continue
        msgs=r['message']
        # Preserve original conversational roles and texts. Use official formatting.
        prompt=tok.apply_chat_template(msgs, tokenize=False)
        token_count=len(tok(prompt, add_special_tokens=False)['input_ids'])
        if token_count>max_input_tokens: continue
        rows.append({'id':str(r['unique_id']), 'label':lab, 'messages':msgs, 'tokens':token_count,
                     'source':str(r.get('source','')), 'unsafe_type':str(r.get('unsafe_type',''))})
    import random
    rng=random.Random(seed)
    groups={k:[r for r in rows if r['label']==k] for k in ('safe','unsafe')}
    for g in groups.values(): rng.shuffle(g)
    half=n//2
    if len(groups['safe'])<half or len(groups['unsafe'])<(n-half):
        raise RuntimeError(f'Insufficient samples for balanced pilot: { {k:len(v) for k,v in groups.items()} }')
    chosen=groups['safe'][:half]+groups['unsafe'][:n-half]
    rng.shuffle(chosen)
    return chosen

def load_model(arm):
    kwargs={'device_map':{'':'cuda:0'}, 'low_cpu_mem_usage':True}
    if arm=='nf4':
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs['torch_dtype']=torch.bfloat16
    model=AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    model.eval()
    return model

@torch.inference_mode()
def infer(arm,rows,tok,max_input_tokens,max_new_tokens,out):
    model=load_model(arm)
    records=[]; start=time.monotonic()
    for i,r in enumerate(rows):
        prompt=tok.apply_chat_template(r['messages'],tokenize=False)
        encoded=tok(prompt,return_tensors='pt',add_special_tokens=False,truncation=False).to('cuda:0')
        if encoded['input_ids'].shape[1]>max_input_tokens: raise RuntimeError('Input exceeds frozen limit')
        generated=model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        answer=tok.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
        found=LABEL_RE.search(answer)
        pred=found.group(1).lower() if found else 'invalid'
        # Controversial mapped to unsafe in the binary moderation policy.
        binary='safe' if pred=='safe' else ('unsafe' if pred in ('unsafe','controversial') else 'invalid')
        records.append({'id':r['id'],'gold':r['label'],'prediction':pred,'binary':binary,
                        'source':r['source'],'unsafe_type':r['unsafe_type'],
                        'tokens':r['tokens'],'output':answer})
        if (i+1)%10==0 or i+1==len(rows):
            save_json(out,{'arm':arm,'model':MODEL,'finished':False,'records':records})
            print(f'{arm} {i+1}/{len(rows)}  elapsed={time.monotonic()-start:.0f}s',flush=True)
    save_json(out,{'arm':arm,'model':MODEL,'finished':True,'records':records,
                   'elapsed_seconds':time.monotonic()-start})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records

def analyze(rows,ref,q):
    a={r['id']:r for r in ref}; b={r['id']:r for r in q}
    paired=[(a[r['id']],b[r['id']]) for r in rows]
    total=len(paired)
    flips=[(x,y) for x,y in paired if x['binary']!=y['binary'] and x['binary']!='invalid' and y['binary']!='invalid']
    invalid={arm:sum(r['binary']=='invalid' for r in vals) for arm,vals in [('bf16',ref),('nf4',q)]}
    def rates(v):
        valid=[r for r in v if r['binary']!='invalid']
        unsafe=[r for r in valid if r['gold']=='unsafe']
        safe=[r for r in valid if r['gold']=='safe']
        fn=sum(r['binary']=='safe' for r in unsafe)
        fp=sum(r['binary']=='unsafe' for r in safe)
        return {'valid':len(valid),'FN_count':fn,'unsafe_valid':len(unsafe),'FNR':fn/len(unsafe) if unsafe else None,
                'FP_count':fp,'safe_valid':len(safe),'FPR':fp/len(safe) if safe else None,
                'accuracy':sum(r['binary']==r['gold'] for r in valid)/len(valid) if valid else None}
    b_ref=rates(ref); b_nf4=rates(q)
    dangerous=sum(x['gold']=='unsafe' and x['binary']=='unsafe' and y['binary']=='safe' for x,y in flips)
    recovered=sum(x['gold']=='unsafe' and x['binary']=='safe' and y['binary']=='unsafe' for x,y in flips)
    return {'total':total,'same_id_order': [r['id'] for r in ref]==[r['id'] for r in q],
            'decision_flips_count':len(flips),'decision_flip_rate':len(flips)/total,
            'unsafe_detected_bf16_missed_nf4':dangerous,'unsafe_missed_bf16_detected_nf4':recovered,
            'invalid_counts':invalid,'bf16':b_ref,'nf4':b_nf4,
            'interpretation':'Exploratory, not novelty proof; do not infer FNR/FPR differences from unpaired valid subsets if invalids exist.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=200)
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--max-input-tokens',type=int,default=1536)
    p.add_argument('--max-new-tokens',type=int,default=48)
    p.add_argument('--out',default='pilot_out')
    args=p.parse_args()
    if args.n<2 or args.n%2: p.error('--n must be even and >= 2')
    if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA GPU required')
    path=Path(args.out); path.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL)
    rows=get_data(tok,args.n,args.seed,args.max_input_tokens)
    config={'model':MODEL,'dataset':DATASET,'split':'thinking','seed':args.seed,'n':args.n,
            'max_input_tokens':args.max_input_tokens,'max_new_tokens':args.max_new_tokens,
            'precision':['bf16','bnb-nf4-weight-only/bf16-compute'],
            'binary_policy':'Controversial treated as unsafe; invalid reported separately',
            'transformers':__import__('transformers').__version__,'torch':torch.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    # Do not save raw input messages; keep IDs and full provenance for exact rehydration.
    selection=[{k:r[k] for k in ('id','label','tokens','source','unsafe_type')} for r in rows]
    config['selection_sha256']=sha(selection)
    save_json(path/'manifest.json',config)
    save_json(path/'selection.json',selection)
    outs={}
    for arm in ('bf16','nf4'):
        target=path/f'{arm}.json'
        if target.exists():
            saved=json.loads(target.read_text(encoding='utf-8'))
            if saved.get('finished') and len(saved.get('records',[]))==args.n:
                outs[arm]=saved['records']; print(f'Skipping completed {arm}',flush=True); continue
        outs[arm]=infer(arm,rows,tok,args.max_input_tokens,args.max_new_tokens,target)
    result=analyze(rows,outs['bf16'],outs['nf4'])
    save_json(path/'summary.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
