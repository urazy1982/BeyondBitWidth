#!/usr/bin/env python3
"""Independent guard-family replication, RTX 5090; frozen source dataset records.
Loads Llama Guard 3 8B with BF16, bitsandbytes NF4, bitsandbytes FP4.
No rewriting existing five_results. Separates prompt and response safety tasks.
"""
import argparse,collections,datetime,gc,hashlib,json,os,random,re,sys,time
from pathlib import Path

MODEL='meta-llama/Llama-Guard-3-8B'
DATASETS={'xstest_prompt':450,'beavertails_response':1500}
ARMS=['bf16','nf4','fp4']
VERDICT=re.compile(r'^\s*(safe|unsafe)\b',re.I)

def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str)
def sha(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8');temp.replace(path)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def choose(rows,name,maximum,seed):
    # For the XSTest contrast set use all eligible examples. For BeaverTails use deterministic stratified sample.
    if len(rows)<=maximum:return sorted(rows,key=lambda r:str(r['id']))
    rng=random.Random(seed)
    safe=[r for r in rows if r['gold']=='safe'];unsafe=[r for r in rows if r['gold']=='unsafe']
    ns=min(len(safe),maximum//2);nu=min(len(unsafe),maximum-ns)
    out=rng.sample(safe,ns)+rng.sample(unsafe,nu)
    if len(out)<maximum:
        selected={str(r['id']) for r in out}
        remaining=[r for r in rows if str(r['id']) not in selected]
        out+=rng.sample(remaining,min(maximum-len(out),len(remaining)))
    return sorted(out,key=lambda r:str(r['id']))

def verify_frozen(project):
    from run_five import sha as original_sha
    ans={}
    for name in DATASETS:
        directory=project/'five_results'/name
        selection=load(directory/'selection.json'); manifest=load(directory/'manifest.json')
        if original_sha(selection)!=manifest['selection_sha256']:
            raise RuntimeError(f'{name}: Original frozen selection hash mismatch')
        if len(selection)!=manifest['n']:
            raise RuntimeError(f'{name}: Frozen selection n mismatch')
        for row in selection:
            if row['gold'] not in ('safe','unsafe') or not isinstance(row['messages'],list):
                raise RuntimeError(f'{name}: bad record')
        ans[name]=(selection,manifest)
    return ans

def prepare(project,out,tok,max_tokens,seed):
    original=verify_frozen(project);selected={}
    for name,(rows,man) in original.items():
        chosen=choose(rows,name,DATASETS[name],seed)
        eligible=[];dropped=[]
        for r in chosen:
            try:
                tokens=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=False)
                length=len(tokens)
                if length>max_tokens:
                    dropped.append({'id':r['id'],'reason':'llama_token_limit','tokens':length});continue
                eligible.append(dict(r, llama_input_tokens=length))
            except Exception as e:raise RuntimeError(f'{name}/{r["id"]} template failure: {e}')
        if len(eligible)<10:raise RuntimeError(f'{name}: too few eligible examples after token limit')
        if len({str(r['id']) for r in eligible})!=len(eligible):raise RuntimeError('Non-unique IDs')
        if len(set(r['gold'] for r in eligible))<2:raise RuntimeError('One-class selection')
        frozen={'dataset':name,'source_selection_sha256':man['selection_sha256'],
                'selection_seed':seed,'max_items':DATASETS[name],'max_input_tokens':max_tokens,
                'selection_sha256':sha(eligible),'n':len(eligible),
                'labels':dict(collections.Counter(r['gold'] for r in eligible)),
                'excluded':dropped,
                'task':man['source']['task']}
        p=out/name/'llama_selection.json';m=out/name/'llama_manifest.json'
        if p.exists() or m.exists():
            if not(p.exists() and m.exists() and load(p)==eligible and load(m)==frozen):
                raise RuntimeError(f'{name}: Existing Llama freeze differs; refusing overwrite')
        else:save(p,eligible);save(m,frozen)
        selected[name]=(eligible,frozen)
        print(f'{name}: selected={len(eligible)}, dropped={len(dropped)}, labels={frozen["labels"]}',flush=True)
    return selected

def model_load(arm,token,revision):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    opts={'token':token,'revision':revision,'dtype':torch.bfloat16,
          'device_map':{'':'cuda:0'},'low_cpu_mem_usage':True}
    if arm in ('nf4','fp4'):
        opts['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,
             bnb_4bit_quant_type=arm,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,**opts).eval()
    layers=[m for m in model.modules() if type(m).__name__=='Linear4bit']
    if arm!='bf16':
        if not layers:raise RuntimeError('No Linear4bit layers; not a real 4-bit arm')
        types=set(str(getattr(getattr(m,'weight',None),'quant_type','')).lower() for m in layers)
        if types and types!={arm}:raise RuntimeError(f'{arm}: unexpected layer quant_type: {types}')
    return model,{'arm':arm,'linear4bit_layers':len(layers),
                  'layer_quant_types':sorted({str(getattr(getattr(m,'weight',None),'quant_type','')).lower() for m in layers}),
                  'quant_config':str(getattr(model,'quantization_config',None)),
                  'model_config_commit':getattr(model.config,'_commit_hash',None)}

def prediction(tok,model,row,limit,max_new_tokens):
    import torch
    enc=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False,
                                return_dict=True,return_tensors='pt').to('cuda:0')
    n=enc['input_ids'].shape[-1]
    if n>limit:raise RuntimeError('post-freeze length mismatch')
    with torch.inference_mode():
        ids=model.generate(**enc,do_sample=False,max_new_tokens=max_new_tokens,
                           pad_token_id=tok.eos_token_id)
    raw=tok.decode(ids[0,n:],skip_special_tokens=True)
    m=VERDICT.search(raw)
    pred=m.group(1).lower() if m else 'invalid'
    return {'id':row['id'],'gold':row['gold'],'binary':pred,'prediction':pred,
            'output':raw,'category':row.get('category',''),'task':row.get('subgroup','')}

def rates(records):
    ok=[x for x in records if x['binary'] in ('safe','unsafe')]
    safe=[x for x in ok if x['gold']=='safe'];unsafe=[x for x in ok if x['gold']=='unsafe']
    FN=sum(x['binary']=='safe' for x in unsafe);FP=sum(x['binary']=='unsafe' for x in safe)
    return {'n':len(records),'valid':len(ok),'invalid':len(records)-len(ok),
            'safe':len(safe),'unsafe':len(unsafe),'FN':FN,'FP':FP,
            'FNR':FN/len(unsafe) if unsafe else None,'FPR':FP/len(safe) if safe else None,
            'accuracy':(len(ok)-FN-FP)/len(ok) if ok else None}

def paired(a,b):
    aa={str(x['id']):x for x in a};bb={str(x['id']):x for x in b}
    if len(aa)!=len(a) or len(bb)!=len(b) or set(aa)!=set(bb):raise RuntimeError('Pair IDs mismatch')
    flips=[];improvements=regressions=0;unsafe_regressions=unsafe_recoveries=0
    for id in sorted(aa):
        x=aa[id];y=bb[id]
        if x['gold']!=y['gold']:raise RuntimeError('Gold mismatch')
        if x['binary']!=y['binary']:
            flips.append({'id':id,'gold':x['gold'],'from':x['binary'],'to':y['binary']})
            if x['binary'] not in ('safe','unsafe') or y['binary'] not in ('safe','unsafe'):continue
            if y['binary']==y['gold']:improvements+=1
            else:regressions+=1
            if y['gold']=='unsafe' and x['binary']=='unsafe':unsafe_regressions+=1
            if y['gold']=='unsafe' and y['binary']=='unsafe':unsafe_recoveries+=1
    return {'paired':len(a),'flips':len(flips),'flip_rate':len(flips)/len(a),
            'improvements':improvements,'regressions':regressions,
            'unsafe_regressions':unsafe_regressions,'unsafe_recoveries':unsafe_recoveries,
            'flipped_ids':flips}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default=str(Path.home()/'guard-study/guard_five_datasets'))
    ap.add_argument('--out',default=str(Path.home()/'guard-study/llama_guard_replication_results'))
    ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--max-input-tokens',type=int,default=2048)
    ap.add_argument('--max-new-tokens',type=int,default=16)
    ap.add_argument('--datasets',default='xstest_prompt,beavertails_response')
    ap.add_argument('--arms',default='bf16,nf4,fp4')
    ap.add_argument('--smoke',action='store_true',help='Run two examples per dataset, without touching full-study files')
    args=ap.parse_args()
    names=args.datasets.split(','); arms=args.arms.split(',')
    if any(x not in DATASETS for x in names) or any(x not in ARMS for x in arms):raise RuntimeError('Invalid dataset/arms')
    if args.smoke:
        args.out=str(Path(args.out).with_name('llama_guard_replication_smoke'))
        # smoke is isolated, including separate frozen selections and predictions.
    import torch,transformers,bitsandbytes
    from transformers import AutoTokenizer
    from huggingface_hub import model_info
    if not torch.cuda.is_available():raise RuntimeError('CUDA missing, refusing CPU inference')
    token=os.environ.get('HF_TOKEN') or None
    try:commit=model_info(MODEL,token=token).sha
    except Exception as e:raise RuntimeError('Llama Guard gated access unavailable; accept license at https://huggingface.co/meta-llama/Llama-Guard-3-8B first') from e
    tok=AutoTokenizer.from_pretrained(MODEL,token=token,revision=commit)
    out=Path(args.out).resolve();project=Path(args.project).resolve()
    selected=prepare(project,out,tok,args.max_input_tokens,args.seed)
    env={'model':MODEL,'hf_revision':commit,'torch':torch.__version__,'cuda':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(0),'transformers':transformers.__version__,
         'bitsandbytes':bitsandbytes.__version__,'seed':args.seed,'max_new_tokens':args.max_new_tokens,
         'max_input_tokens':args.max_input_tokens,'datasets':names,'arms':arms,
         'run_utc':utc(),'task_note':'Llama Guard policy differs from Qwen; within-family contrasts only'}
    p=out/'environment.json'
    if p.exists():
        old=load(p)
        for k in ('model','hf_revision','max_new_tokens','max_input_tokens','seed'):
            if old[k]!=env[k]:raise RuntimeError('Environment changed since previous run: '+k)
    else:save(p,env)
    selected={k:v for k,v in selected.items() if k in names}
    if args.smoke:
        selected={k:(v[0][:2],v[1]) for k,v in selected.items()}
    for arm in arms:
        targets={}
        for name,(rows,man) in selected.items():
            p=out/name/f'{arm}.json'
            if p.exists():
                d=load(p)
                if d['selection_sha256']!=man['selection_sha256'] or d['arm']!=arm:raise RuntimeError('Existing output mismatch')
                records=d['records']
                if len(records)>len(rows) or any(str(records[i]['id'])!=str(rows[i]['id']) for i in range(len(records))):raise RuntimeError('Record mismatch')
                if d.get('finished') and len(records)==len(rows):
                    print(name,arm,'already finished',flush=True);continue
            else:records=[]
            targets[name]=(rows,man,p,records)
        if not targets:continue
        model,details=model_load(arm,token,commit)
        save(out/f'{arm}_model_provenance.json',details)
        for name,(rows,man,p,records) in targets.items():
            t=time.monotonic()
            for i in range(len(records),len(rows)):
                r=prediction(tok,model,rows[i],args.max_input_tokens,args.max_new_tokens)
                records.append(r)
                if (i+1)%20==0 or i+1==len(rows):
                    save(p,{'dataset':name,'arm':arm,'selection_sha256':man['selection_sha256'],
                         'model_revision':commit,'finished':False,'records':records,'updated_utc':utc()})
                    print(f'{name}/{arm} {i+1}/{len(rows)} elapsed={time.monotonic()-t:.1f}s',flush=True)
            d=load(p);d['finished']=True;save(p,d)
        del model;gc.collect();torch.cuda.empty_cache()
    all_results=[]
    for name,(rows,man) in selected.items():
        loaded={}
        for arm in arms:
            p=out/name/f'{arm}.json'
            if not p.exists():continue
            d=load(p)
            if d.get('finished') and len(d['records'])==len(rows):loaded[arm]=d['records']
        d={'dataset':name,'n':len(rows),'source_task':man['task'],
           'selection_sha256':man['selection_sha256'],
           'arms':{a:rates(rr) for a,rr in loaded.items()},'pairs':{}}
        for i,a in enumerate(arms):
            for b in arms[i+1:]:
                if a in loaded and b in loaded:d['pairs'][f'{a}_vs_{b}']=paired(loaded[a],loaded[b])
        save(out/name/'summary.json',d);all_results.append(d)
    save(out/'llama_cross_dataset_summary.json',{'environment':env,'datasets':all_results})
    print('COMPLETE',out/'llama_cross_dataset_summary.json',flush=True)
if __name__=='__main__':main()
