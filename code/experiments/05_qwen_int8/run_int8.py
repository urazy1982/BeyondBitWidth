#!/usr/bin/env python3
"""Read-only extension of the frozen five-dataset Qwen3Guard study.
Runs bitsandbytes INT8 on exactly the original selection.json records. Never
modifies BF16/NF4/FP8 or the existing study's summaries.
"""
import argparse, collections, gc, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# Run from the existing study directory; import its original tested rules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_five as base

ARM='int8'
AVAILABLE=('qwen_thinking','toxicchat_prompt','xstest_prompt','beavertails_response','wildguard_prompt')

def lib_version(name):
    try:return version(name)
    except PackageNotFoundError:return None

def validate_frozen(root,name):
    folder,selection,manifest=base.selection_files(root,name)
    if not selection.exists() or not manifest.exists():
        raise RuntimeError(f'{name}: existing selection/manifest not found. Refusing to create new selection.')
    man=base.read(manifest); rows=base.read(selection)
    if man.get('model')!=base.MODEL or man.get('dataset')!=name:
        raise RuntimeError(f'{name}: unexpected source model or dataset')
    if man.get('selection_sha256') != base.sha(rows) or len(rows)!=man.get('n'):
        raise RuntimeError(f'{name}: frozen selection integrity failure')
    if len({str(r['id']) for r in rows}) !=len(rows):
        raise RuntimeError(f'{name}: duplicate IDs')
    nf4=folder/'nf4.json'
    if not nf4.exists():raise RuntimeError(f'{name}: original NF4 result absent')
    orig=base.read(nf4)
    recs=orig.get('records',[])
    if not orig.get('finished') or orig.get('arm')!='nf4' or orig.get('selection_sha256')!=man['selection_sha256'] or len(recs)!=len(rows):
        raise RuntimeError(f'{name}: NF4 run incomplete/incompatible')
    for i,(r,old) in enumerate(zip(rows,recs)):
        if str(r['id']) !=str(old['id']) or r['gold']!=old['gold']:
            raise RuntimeError(f'{name}: NF4 ID/gold mismatch at {i}')
    return folder,rows,man,orig

def load_int8():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    conf=BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                           llm_int8_enable_fp32_cpu_offload=False)
    args={'device_map':{'':'cuda:0'},'dtype':torch.bfloat16,'low_cpu_mem_usage':True,
          'quantization_config':conf}
    token=os.environ.get('HF_TOKEN')
    if token:args['token']=token
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,**args).eval()
    count=sum(type(m).__name__=='Linear8bitLt' for m in model.modules())
    if count==0:raise RuntimeError('No bitsandbytes Linear8bitLt layers; abort')
    details={'format':'bitsandbytes LLM.int8 mixed precision (NOT pure fixed-INT8 arithmetic)',
             'bitsandbytes_type':'llm_int8','threshold':6.0,
             'model_revision':str(getattr(model.config,'_commit_hash',None)),
             'linear8bit_layers':count,'quantization_config':str(conf),
             'libraries':{x:lib_version(x) for x in ('torch','transformers','bitsandbytes','accelerate')},
             'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
             'gpu_capability':str(torch.cuda.get_device_capability(0))}
    return model,details

def check_nf4_config(orig):
    data=str(orig.get('verification',{}).get('quantization_config','')).lower()
    # If serialized metadata is absent or opaque, leave explicit warning. Original
    # run_five.py is the authoritative config; compare its loader before analysis.
    if data and 'nf4' not in data:
        raise RuntimeError('NF4 verification does not mention nf4; inspect before comparing')

def run_dataset(root,name,model,tok,details):
    import torch
    folder,rows,man,orig=validate_frozen(root,name)
    check_nf4_config(orig)
    path=folder/'int8.json'
    if path.exists():
        saved=base.read(path)
        if saved.get('selection_sha256')!=man['selection_sha256'] or saved.get('arm')!='int8':
            raise RuntimeError(f'{name}: incompatible INT8 output exists; refusing overwrite')
        recs=saved.get('records',[])
        if len(recs)>len(rows) or any(str(recs[i]['id'])!=str(rows[i]['id']) or recs[i]['gold']!=rows[i]['gold'] for i in range(len(recs))):
            raise RuntimeError(f'{name}: INT8 checkpoint alignment failure')
        if saved.get('finished') and len(recs)==len(rows):
            print(f'{name}: INT8 already complete; skip',flush=True)
            return
    else: recs=[]
    start=time.monotonic()
    for i in range(len(recs),len(rows)):
        row=rows[i]
        prompt=tok.apply_chat_template(row['messages'],tokenize=False)
        if hashlib.sha256(prompt.encode()).hexdigest()!=row['prompt_hash']:
            raise RuntimeError(f'{name}: tokenizer/template mismatch for ID={row["id"]}')
        inp=tok(prompt,add_special_tokens=False,return_tensors='pt',truncation=False).to('cuda:0')
        if inp.input_ids.shape[-1]>man['max_input_tokens']:
            raise RuntimeError(f'{name}: token budget mismatch')
        with torch.inference_mode():
            result=model.generate(**inp,max_new_tokens=man['max_new_tokens'],do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        raw=tok.decode(result[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
        found=base.LABEL_RE.search(raw)
        parsed=found.group(1).lower() if found else 'invalid'
        binary='safe' if parsed=='safe' else 'unsafe' if parsed in ('unsafe','controversial') else 'invalid'
        recs.append({'id':row['id'],'gold':row['gold'],'binary':binary,'prediction':parsed,
                     'output':raw,'category':row['category'],'subgroup':row['subgroup']})
        if (i+1)%20==0 or i+1==len(rows):
            base.dump(path,{'dataset':name,'arm':'int8','selection_sha256':man['selection_sha256'],
                            'finished':False,'records':recs,'verification':details,
                            'updated_utc':datetime.now(timezone.utc).isoformat()})
            print(f'{name}/int8 {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    saved=base.read(path)
    saved['finished']=True
    saved['session_elapsed_seconds']=time.monotonic()-start
    saved['updated_utc']=datetime.now(timezone.utc).isoformat()
    base.dump(path,saved)

def report(root,names):
    results=[]
    for name in names:
        try:
            folder,rows,man,orig=validate_frozen(root,name)
        except RuntimeError as exc:
            print('SKIP report',exc,flush=True);continue
        fpfile=folder/'int8.json'
        if not fpfile.exists():continue
        f=base.read(fpfile)
        if not f.get('finished') or len(f.get('records',[]))!=len(rows) or f['selection_sha256']!=man['selection_sha256']:
            print(f'SKIP report {name}: INT8 incomplete',flush=True);continue
        row={'dataset':name,'source_task':man['source']['task'],'n':len(rows),
             'selection_sha256':man['selection_sha256'],'int8':base.rate(f['records']),
             'nf4':base.rate(orig['records']),
             'nf4_vs_int8':base.compare(orig['records'],f['records']),
             'execution':f['verification']}
        bf=folder/'bf16.json';fp8=folder/'fp8.json'
        for other,otherfile in [('bf16',bf),('fp8',fp8)]:
            if otherfile.exists():
                a=base.read(otherfile)
                if a.get('finished') and a.get('selection_sha256')==man['selection_sha256'] and len(a.get('records',[]))==len(rows):
                    row[other]=base.rate(a['records'])
                    row[other+'_vs_int8']=base.compare(a['records'],f['records'])
        results.append(row)
        base.dump(folder/'int8_comparison.json',row)
        p=row['nf4_vs_int8']
        print(f'{name}: NF4-vs-INT8 flips={p["flips"]} new_errors={p["regressions"]} corrections={p["improvements"]}, unsafe_new_misses={p["unsafe_regressions"]}',flush=True)
    base.dump(root/'int8_cross_dataset_summary.json',{
        'description':'Comparison of bitsandbytes LLM.int8 with previous precisions; analyze separate dataset tasks; no pooled FNR',
        'datasets':results})
    import csv
    with (root/'int8_summary_table.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['dataset','n','arm','accuracy','FNR','FPR','invalid','nf4_vs_int8_flips','nf4_vs_int8_improvements','nf4_vs_int8_regressions'])
        for row in results:
            for arm in ('nf4','int8'):
                q=row[arm];d=row['nf4_vs_int8']
                w.writerow([row['dataset'],row['n'],arm,q['accuracy'],q['FNR'],q['FPR'],q['invalid'],d['flips'],d['improvements'],d['regressions']])
    print(f'Report includes {len(results)} completed datasets.',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='five_results')
    ap.add_argument('--dataset',choices=('all',)+AVAILABLE,default='all')
    ap.add_argument('--report-only',action='store_true')
    args=ap.parse_args();root=Path(args.out).resolve()
    names=AVAILABLE if args.dataset=='all' else (args.dataset,)
    if args.report_only:report(root,names);return
    existing=[]
    for name in names:
        if not (root/name/'manifest.json').exists():
            print(f'SKIP {name}: no manifest',flush=True);continue
        validate_frozen(root,name)
        existing.append(name)
    if not existing:raise RuntimeError('No complete frozen NF4 reference datasets found')
    from transformers import AutoTokenizer
    import torch
    token=os.environ.get('HF_TOKEN') or None
    tok=AutoTokenizer.from_pretrained(base.MODEL,token=token)
    model,details=load_int8()
    try:
        for name in existing:run_dataset(root,name,model,tok,details)
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    report(root,existing)

if __name__=='__main__':main()
