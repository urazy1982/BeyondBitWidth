#!/usr/bin/env python3
"""Five benchmark, per-dataset, per-arm Qwen3Guard precision study.
No implicit task pooling; immutable selection, resumable prediction files.
"""
import argparse, collections, gc, hashlib, json, os, random, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

MODEL = 'Qwen/Qwen3Guard-Gen-4B'
DATASETS = ('qwen_thinking','wildguard_prompt','toxicchat_prompt','xstest_prompt','beavertails_response')
ARMS = ('bf16','nf4','fp8')
LABEL_RE = re.compile(r'\bSafety:\s*(Safe|Unsafe|Controversial)\b', re.I)

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)

def sha(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def dump(path, x):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp'); temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); temp.replace(p)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def normalize_binary(v):
    if isinstance(v,bool): return 'safe' if v else 'unsafe'
    s=str(v).strip().lower()
    if s in ('safe','unharmful','harmless','0','false'):return 'safe'
    if s in ('unsafe','harmful','toxic','1','true'):return 'unsafe'
    return None

def get_source(name):
    from datasets import load_dataset
    if name=='qwen_thinking':
        ds=load_dataset('Qwen/Qwen3GuardTest',split='thinking')
        return ds, {'repo':'Qwen/Qwen3GuardTest','split':'thinking','task':'response_safety'}
    if name=='wildguard_prompt':
        ds=load_dataset('allenai/wildguardmix','wildguardtest',split='test')
        return ds, {'repo':'allenai/wildguardmix','config':'wildguardtest','split':'test','task':'prompt_safety'}
    if name=='toxicchat_prompt':
        ds=load_dataset('lmsys/toxic-chat','toxicchat0124',split='test')
        return ds, {'repo':'lmsys/toxic-chat','config':'toxicchat0124','split':'test','task':'prompt_toxicity_proxy'}
    if name=='xstest_prompt':
        ds=load_dataset('Paul/XSTest',split='train')
        return ds, {'repo':'Paul/XSTest','split':'train','task':'prompt_safety_contrast'}
    if name=='beavertails_response':
        ds=load_dataset('PKU-Alignment/BeaverTails',split='30k_test')
        return ds, {'repo':'PKU-Alignment/BeaverTails','split':'30k_test','task':'response_safety'}
    raise ValueError(name)

def source_record(name, r, i):
    # This is the declared gold target, not a universal safety ontology.
    if name=='qwen_thinking':
        gold=normalize_binary(r.get('label'))
        msgs=r.get('message')
        ident=str(r.get('unique_id',i))
        category=str(r.get('unsafe_type',''))
        subgroup=str(r.get('source',''))
    elif name=='wildguard_prompt':
        gold=normalize_binary(r.get('prompt_harm_label'))
        msgs=[{'role':'user','content':str(r.get('prompt') or '')}]
        ident=str(i); category=str(r.get('subcategory') or ''); subgroup=str(r.get('adversarial'))
    elif name=='toxicchat_prompt':
        toxicity=r.get('toxicity')
        gold='unsafe' if toxicity==1 else ('safe' if toxicity==0 else None)
        msgs=[{'role':'user','content':str(r.get('user_input') or '')}]
        ident=str(r.get('conv_id') or i); category='toxicity'; subgroup=str(r.get('jailbreaking'))
    elif name=='xstest_prompt':
        gold=normalize_binary(r.get('label'))
        msgs=[{'role':'user','content':str(r.get('prompt') or '')}]
        ident=str(r.get('id',i)); category=str(r.get('type') or ''); subgroup='contrast' if category.startswith('contrast') else 'safe-type'
    elif name=='beavertails_response':
        v=r.get('is_safe')
        gold=('safe' if v else 'unsafe') if isinstance(v,bool) else None
        prompt=str(r.get('prompt') or ''); response=str(r.get('response') or '')
        msgs=[{'role':'user','content':prompt},{'role':'assistant','content':response}]
        ident=str(i); category='beavertails'; subgroup='qa'
    else: raise ValueError(name)
    if gold not in ('safe','unsafe') or not isinstance(msgs,list) or not msgs:return None
    if not any(str(m.get('content') or '').strip() for m in msgs):return None
    return {'id': ident,'gold':gold,'messages':msgs,'category':category,'subgroup':subgroup}

def build_selection(name, tok, max_items, max_input_tokens, seed):
    ds,meta=get_source(name)
    if len(ds)==0:raise RuntimeError('Empty source')
    rows=[]; excluded=collections.Counter(); unique=set()
    for i,r in enumerate(ds):
        item=source_record(name,r,i)
        if item is None: excluded['unusable_label_or_text']+=1; continue
        if item['id'] in unique:raise RuntimeError('Duplicate ID in dataset: '+item['id'])
        unique.add(item['id'])
        try: prompt=tok.apply_chat_template(item['messages'],tokenize=False)
        except Exception as e:raise RuntimeError(f'Template failure for {name} ID={item["id"]}') from e
        n=len(tok(prompt,add_special_tokens=False)['input_ids'])
        if n>max_input_tokens:excluded['over_input_limit']+=1; continue
        item['tokens']=n; item['prompt_hash']=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        rows.append(item)
    eligible_count=len(rows)
    if max_items and len(rows)>max_items:
        # Random sample, not class balancing; preserves natural dataset prevalence in expectation.
        rng=random.Random(seed); rng.shuffle(rows); rows=rows[:max_items]
    rows.sort(key=lambda x:x['id'])
    meta.update({'source_rows':len(ds),'eligible_rows_before_sampling':eligible_count,
                 'selected_rows':len(rows),'excluded':dict(excluded),'label_counts':dict(collections.Counter(x['gold'] for x in rows)),
                 'category_counts':dict(collections.Counter(x['category'] for x in rows)),
                 'source_fingerprint':str(getattr(ds,'_fingerprint','unknown'))})
    if len(rows)<10:raise RuntimeError(f'{name}: too few eligible samples ({len(rows)})')
    if len(meta['label_counts'])!=2:raise RuntimeError(f'{name}: both safe and unsafe classes required')
    return rows,meta

def selection_files(root,name):
    d=root/name
    return d,d/'selection.json',d/'manifest.json'

def prepare(root, names, args):
    from transformers import AutoTokenizer
    token=os.environ.get('HF_TOKEN') or None
    tok=AutoTokenizer.from_pretrained(MODEL,token=token)
    for name in names:
        folder,sel_path,man_path=selection_files(root,name)
        if sel_path.exists() or man_path.exists():
            if not (sel_path.exists() and man_path.exists()):raise RuntimeError(f'{name}: incomplete freeze files; manual inspection required')
            rows=read(sel_path);man=read(man_path)
            if sha(rows)!=man['selection_sha256']:raise RuntimeError(f'{name}: frozen selection modified')
            for k,v in [('model',MODEL),('max_new_tokens',args.max_new_tokens),('max_input_tokens',args.max_input_tokens)]:
                if man[k]!=v:raise RuntimeError(f'{name}: {k} changed from frozen manifest')
            print(f'{name}: frozen selection reused ({len(rows)} samples)',flush=True)
            continue
        rows,meta=build_selection(name,tok,args.max_items,args.max_input_tokens,args.seed)
        manifest={'model':MODEL,'dataset':name,'source':meta,'n':len(rows),'seed':args.seed,
                  'max_items':args.max_items,'max_input_tokens':args.max_input_tokens,
                  'max_new_tokens':args.max_new_tokens,'selection_sha256':sha(rows),
                  'label_policy':'Safe=>safe; Unsafe/Controversial=>unsafe; invalid separate',
                  'sample_policy':'unbalanced random sample, no oversampling, selected rows frozen',
                  'created_utc':datetime.now(timezone.utc).isoformat()}
        dump(sel_path,rows);dump(man_path,manifest)
        print(f'{name}: prepared {len(rows)}; labels {meta["label_counts"]}',flush=True)

def load_model(arm):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw={'device_map':{'':'cuda:0'},'dtype':torch.bfloat16,'low_cpu_mem_usage':True}
    token=os.environ.get('HF_TOKEN')
    if token:kw['token']=token
    if arm=='nf4':kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False)
    elif arm=='fp8':
        from transformers import FineGrainedFP8Config
        kw['quantization_config']=FineGrainedFP8Config()
    elif arm!='bf16':raise ValueError(arm)
    model=AutoModelForCausalLM.from_pretrained(MODEL,**kw).eval()
    types=collections.Counter(type(m).__name__ for m in model.modules())
    if arm=='nf4' and not any('Linear4bit' in t for t in types):raise RuntimeError('No NF4 Linear4bit layers found')
    if arm=='fp8' and not any('fp8' in t.lower() or 'finegrained' in t.lower() for t in types):raise RuntimeError('No FP8 layer type found')
    verify={'layer_types':{k:v for k,v in types.items() if 'fp8' in k.lower() or 'finegrained' in k.lower() or 'Linear4bit' in k},
            'quantization_config':str(getattr(model,'quantization_config',None))}
    return model,verify

def predict(root,names,arm):
    import torch, transformers
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU missing')
    token=os.environ.get('HF_TOKEN')
    tok=AutoTokenizer.from_pretrained(MODEL,token=token)
    model,verification=load_model(arm)
    try:
        for name in names:
            folder,sp,mp=selection_files(root,name);rows=read(sp);man=read(mp)
            if sha(rows)!=man['selection_sha256']:raise RuntimeError(f'{name}: selection changed')
            op=folder/f'{arm}.json'; previous=None
            if op.exists():
                previous=read(op)
                if previous.get('selection_sha256')!=man['selection_sha256'] or previous.get('arm')!=arm:
                    raise RuntimeError(f'{name}/{arm}: incompatible existing output')
                recs=previous.get('records',[])
                if len(recs)>len(rows) or any(recs[i]['id']!=rows[i]['id'] or recs[i]['gold']!=rows[i]['gold'] for i in range(len(recs))):raise RuntimeError('Output ID/gold ordering corruption')
                if previous.get('finished') and len(recs)==len(rows):
                    print(f'{name}/{arm}: complete, skipping',flush=True);continue
            else: recs=[]
            start=time.monotonic()
            for i in range(len(recs),len(rows)):
                r=rows[i]; prompt=tok.apply_chat_template(r['messages'],tokenize=False)
                if hashlib.sha256(prompt.encode('utf-8')).hexdigest()!=r['prompt_hash']:
                    raise RuntimeError(f'{name}: tokenizer/template drift for {r["id"]}')
                encoded=tok(prompt,add_special_tokens=False,return_tensors='pt',truncation=False).to('cuda:0')
                if encoded.input_ids.shape[-1]>man['max_input_tokens']:raise RuntimeError('Token budget drift')
                with torch.inference_mode():
                    output=model.generate(**encoded,max_new_tokens=man['max_new_tokens'],do_sample=False,pad_token_id=tok.eos_token_id)
                raw=tok.decode(output[0,encoded.input_ids.shape[1]:],skip_special_tokens=True)
                m=LABEL_RE.search(raw); pred=m.group(1).lower() if m else 'invalid'
                binary=('safe' if pred=='safe' else 'unsafe' if pred in ('unsafe','controversial') else 'invalid')
                recs.append({'id':r['id'],'gold':r['gold'],'binary':binary,'prediction':pred,'output':raw,
                             'category':r['category'],'subgroup':r['subgroup']})
                if (i+1)%20==0 or i+1==len(rows):
                    dump(op,{'dataset':name,'arm':arm,'selection_sha256':man['selection_sha256'],
                             'finished':False,'records':recs,'verification':verification,'environment':{
                             'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
                             'gpu':torch.cuda.get_device_name(0)}})
                    print(f'{name}/{arm}: {i+1}/{len(rows)} elapsed={time.monotonic()-start:.0f}s',flush=True)
            result=read(op);result['finished']=True;result['elapsed_seconds_this_session']=time.monotonic()-start;dump(op,result)
    finally:
        del model; gc.collect();torch.cuda.empty_cache()

def rate(recs):
    valid=[r for r in recs if r['binary'] in ('safe','unsafe')]
    safe=[r for r in valid if r['gold']=='safe'];unsafe=[r for r in valid if r['gold']=='unsafe']
    fn=sum(r['binary']=='safe' for r in unsafe);fp=sum(r['binary']=='unsafe' for r in safe)
    return {'n':len(recs),'valid':len(valid),'invalid':len(recs)-len(valid),
            'safe':len(safe),'unsafe':len(unsafe), 'FN':fn,'FP':fp,
            'FNR':fn/len(unsafe) if unsafe else None,'FPR':fp/len(safe) if safe else None,
            'accuracy':sum(r['binary']==r['gold'] for r in valid)/len(valid) if valid else None}

def compare(a,b):
    flipped=[];improved=regressed=unsafe_regress=unsafe_recover=0; paired=0
    for x,y in zip(a,b):
        if x['id']!=y['id'] or x['gold']!=y['gold']:raise RuntimeError('Paired alignment failure')
        if x['binary']=='invalid' or y['binary']=='invalid':continue
        paired+=1
        if x['binary']!=y['binary']:
            flipped.append({'id':x['id'],'gold':x['gold'],'from':x['binary'],'to':y['binary'],'category':x['category']})
            if x['binary']==x['gold']:regressed+=1
            else:improved+=1
            if x['gold']=='unsafe' and x['binary']=='unsafe':unsafe_regress+=1
            if x['gold']=='unsafe' and y['binary']=='unsafe':unsafe_recover+=1
    return {'paired_valid':paired,'flips':len(flipped),'flip_rate':len(flipped)/paired if paired else None,
            'improvements':improved,'regressions':regressed,'unsafe_regressions':unsafe_regress,
            'unsafe_recoveries':unsafe_recover,'flipped_ids':flipped}

def report(root,names):
    import csv
    completed=[]
    for name in names:
        folder,sp,mp=selection_files(root,name)
        if not mp.exists():continue
        man=read(mp); d={'dataset':name,'source':man['source'],'n':man['n'],'arms':{},'pairs':{}}
        allrecs={}
        for arm in ARMS:
            p=folder/f'{arm}.json'
            if not p.exists():continue
            obj=read(p)
            if not obj.get('finished') or obj.get('selection_sha256')!=man['selection_sha256'] or len(obj['records'])!=man['n']:
                raise RuntimeError(f'{name}/{arm}: incomplete or invalid run')
            allrecs[arm]=obj['records'];d['arms'][arm]=rate(obj['records'])
        for a,b in [('bf16','nf4'),('bf16','fp8'),('nf4','fp8')]:
            if a in allrecs and b in allrecs:d['pairs'][f'{a}_vs_{b}']=compare(allrecs[a],allrecs[b])
        dump(folder/'summary.json',d);completed.append(d)
    dump(root/'cross_dataset_summary.json',{'method':'Per-dataset analyses; no pooled FNR or accuracy across distinct tasks',
        'datasets':completed})
    with (root/'summary_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['dataset','task','n','arm','accuracy','FNR','FPR','invalid'])
        for d in completed:
            for arm,metrics in d['arms'].items():writer.writerow([d['dataset'],d['source']['task'],d['n'],arm,metrics['accuracy'],metrics['FNR'],metrics['FPR'],metrics['invalid']])
    for d in completed:
        print(f"\n{d['dataset']} ({d['source']['task']}, n={d['n']})")
        for a,v in d['arms'].items():print(f"  {a}: accuracy={v['accuracy']:.4f} FNR={v['FNR']} FPR={v['FPR']} invalid={v['invalid']}")
        for p,v in d['pairs'].items():print(f"  {p}: flips={v['flips']}, regressions={v['regressions']}, recoveries={v['improvements']}")
    print('Saved:',root/'cross_dataset_summary.json',root/'summary_table.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('all',)+DATASETS,default='all')
    p.add_argument('--arm',choices=('all',)+ARMS,default='all')
    p.add_argument('--out',default='five_results')
    p.add_argument('--max-items',type=int,default=5000,help='Per dataset cap; 0 means all eligible')
    p.add_argument('--max-input-tokens',type=int,default=1536)
    p.add_argument('--max-new-tokens',type=int,default=48)
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    names=DATASETS if args.dataset=='all' else (args.dataset,)
    if args.report_only:report(root,names);return
    prepare(root,names,args)
    if args.prepare_only:return
    arms=ARMS if args.arm=='all' else (args.arm,)
    for arm in arms:predict(root,names,arm)
    report(root,names)
if __name__=='__main__':main()
