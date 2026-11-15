#!/usr/bin/env python3
"""Pinned QwenGuard cache replay of ALL archived NF4/FP4 flips plus frozen controls."""
import argparse, hashlib, json, os, re, sys, time, traceback, gc
from datetime import datetime, timezone
from pathlib import Path

REV='6ec42827da0c1ff11e7a49dc269d2e810d27e108'
MODEL='Qwen/Qwen3Guard-Gen-4B'
DATASETS=['qwen_thinking','toxicchat_prompt','xstest_prompt','beavertails_response']

def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)
def sha_text(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()

def validate(root):
 sets={}
 for ds in DATASETS:
  p=root/'frozen'/ds;rows=read(p/'selection.json');man=read(p/'manifest.json');meta=read(p/'subset_metadata.json');refs=read(p/'reference.json')
  assert len(rows)==meta['n'] and len({str(x['id']) for x in rows})==len(rows)
  assert len(refs['nf4'])==len(rows)==len(refs['fp4']) and man['selection_sha256']==meta['selection_sha256']
  for fmt in ('nf4','fp4'):
   assert all(str(a['id'])==str(b['id']) and a['gold']==b['gold'] for a,b in zip(rows,refs[fmt]))
  flips=[str(row['id']) for i,row in enumerate(rows) if refs['nf4'][i]['binary']!=refs['fp4'][i]['binary']]
  assert set(flips)==set(meta['changed_ids']) and len(flips)==meta['flips']
  assert set(meta['control_ids']).isdisjoint(flips) and meta['controls']==len(meta['control_ids'])
  sets[ds]=(rows,man,meta,refs)
 return sets

def report(sets,out):
 result={'revision':REV,'updated_utc':utc(),'datasets':{},'total':{},'claim_scope':'matched archived predictions on sampled frozen inputs only; not conclusive proof of historical revision'}
 aggregate={'target_inputs':0,'replayed_pairs':0,'replayed_predictions':0,'decision_mismatches':0,'text_mismatches':0,'historical_flips_recovered':0,'historical_flips_checked':0}
 for ds,(rows,man,meta,refs) in sets.items():
  recs={}
  for fmt in ('nf4','fp4'):
   p=out/ds/(fmt+'.json');recs[fmt]=read(p)['records'] if p.exists() else []
  diffs={}
  for fmt in ('nf4','fp4'):
   a={str(x['id']):x for x in refs[fmt]};b={str(x['id']):x for x in recs[fmt]}
   diffs[fmt]={'completed':len(b),'decision_mismatch_ids':[id for id,x in b.items() if x['binary']!=a[id]['binary'] or x['prediction']!=a[id]['prediction']], 'raw_text_mismatch_ids':[id for id,x in b.items() if x['output']!=a[id]['output']]}
  ix={f:{str(x['id']):x for x in recs[f]} for f in recs};common=sorted(set(ix['nf4'])&set(ix['fp4']))
  recovered=[i for i in common if ix['nf4'][i]['binary']!=ix['fp4'][i]['binary'] and i in meta['changed_ids']]
  checked=[i for i in common if i in meta['changed_ids']]
  result['datasets'][ds]={'target':len(rows),'archived_flips':meta['flips'],'controls':meta['controls'],'formats':diffs,'paired_completed':len(common),'historical_flips_checked':len(checked),'historical_flips_recovered':len(recovered),'fresh_flip_ids':[i for i in common if ix['nf4'][i]['binary']!=ix['fp4'][i]['binary']]}
  aggregate['target_inputs']+=len(rows);aggregate['replayed_pairs']+=len(common);aggregate['replayed_predictions']+=sum(len(recs[f]) for f in recs)
  aggregate['decision_mismatches']+=sum(len(diffs[f]['decision_mismatch_ids']) for f in recs)
  aggregate['text_mismatches']+=sum(len(diffs[f]['raw_text_mismatch_ids']) for f in recs)
  aggregate['historical_flips_recovered']+=len(recovered);aggregate['historical_flips_checked']+=len(checked)
 aggregate['complete']=(aggregate['replayed_pairs']==aggregate['target_inputs'] and aggregate['replayed_predictions']==2*aggregate['target_inputs'])
 aggregate['all_decisions_match']=aggregate['complete'] and aggregate['decision_mismatches']==0
 aggregate['all_raw_text_match']=aggregate['complete'] and aggregate['text_mismatches']==0
 result['total']=aggregate;write(out/'comparison_summary.json',result)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).parent));ap.add_argument('--out',required=True);ap.add_argument('--max-minutes',type=int,default=150);ap.add_argument('--checkpoint-every',type=int,default=5);a=ap.parse_args()
 root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 sets=validate(root)
 print('CPU FROZEN DATA VALIDATION PASS |',sum(len(sets[d][0]) for d in DATASETS),'inputs | pinned',REV,flush=True)
 import torch,transformers,bitsandbytes
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing CPU inference')
 identity={'model':MODEL,'revision':REV,'torch':torch.__version__,'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,'gpu':torch.cuda.get_device_name(0),'started_utc':utc(),'greedy':True,'max_new_tokens':48,'format_order':['nf4','fp4']}
 existing=out/'run_identity.json'
 if existing.exists():
  prior=read(existing)
  if any(prior[k]!=identity[k] for k in ('model','revision','torch','transformers','bitsandbytes','gpu')):raise RuntimeError('Run environment changed on resume; use a fresh output directory')
 else:write(existing,identity)
 token=os.environ.get('HF_TOKEN') or None
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,token=token,local_files_only=True)
 token_commit=(tok.init_kwargs or {}).get('_commit_hash')
 for ds,(rows,man,meta,refs) in sets.items():
  for row in rows:
   prompt=tok.apply_chat_template(row['messages'],tokenize=False)
   if sha_text(prompt)!=row['prompt_hash']:raise RuntimeError(f'PROMPT HASH MISMATCH {ds}/{row["id"]}; stop before inference')
 print('PROMPT HASHES PASS for all 280 inputs; tokenizer_commit=',token_commit,flush=True)
 deadline=time.monotonic()+a.max_minutes*60
 try:
  for fmt in ('nf4','fp4'):
   incomplete=[]
   for ds,(rows,man,meta,refs) in sets.items():
    path=out/ds/(fmt+'.json');existing_recs=read(path)['records'] if path.exists() else []
    if existing_recs:
     assert len(existing_recs)<=len(rows)
     assert all(str(r['id'])==str(rows[i]['id']) and r['gold']==rows[i]['gold'] for i,r in enumerate(existing_recs))
    if len(existing_recs)<len(rows):incomplete.append(ds)
   if not incomplete:continue
   if time.monotonic()>deadline:break
   cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type=fmt,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False)
   print('LOADING',fmt,'PINNED',REV,flush=True)
   model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,token=token,local_files_only=True,device_map={'':'cuda:0'},dtype=torch.bfloat16,low_cpu_mem_usage=True,quantization_config=cfg,trust_remote_code=False).eval()
   layers=[x for x in model.modules() if type(x).__name__=='Linear4bit'];types=sorted({str(getattr(getattr(x,'weight',None),'quant_type','')).lower() for x in layers})
   if len(layers)!=252 or any(x not in ('',fmt) for x in types):raise RuntimeError(f'Quantized layer verification failed: {len(layers)} {types}')
   write(out/(fmt+'_provenance.json'),{'pinned_revision':REV,'model_config_commit':getattr(model.config,'_commit_hash',None),'tokenizer_commit':token_commit,'quant_type':fmt,'layers':len(layers),'quant_types_seen':types,'loaded_utc':utc(),'torch':torch.__version__,'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,'gpu':torch.cuda.get_device_name(0)})
   for ds in DATASETS:
    rows,man,meta,refs=sets[ds];path=out/ds/(fmt+'.json');recs=read(path)['records'] if path.exists() else []
    for i in range(len(recs),len(rows)):
     if time.monotonic()>deadline:print('TIME LIMIT reached; saving partial checkpoint',flush=True);break
     row=rows[i];prompt=tok.apply_chat_template(row['messages'],tokenize=False)
     inp=tok(prompt,add_special_tokens=False,return_tensors='pt',truncation=False).to('cuda:0')
     if inp.input_ids.shape[-1]>man['max_input_tokens']:raise RuntimeError('max input token limit mismatch')
     with torch.inference_mode(): seq=model.generate(**inp,max_new_tokens=man['max_new_tokens'],do_sample=False,pad_token_id=tok.eos_token_id)
     raw=tok.decode(seq[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
     m=re.search(r'\bSafety:\s*(Safe|Unsafe|Controversial)\b',raw,re.I)
     pred=m.group(1).lower() if m else 'invalid';binary='safe' if pred=='safe' else 'unsafe' if pred in ('unsafe','controversial') else 'invalid'
     recs.append({'id':row['id'],'gold':row['gold'],'prediction':pred,'binary':binary,'output':raw})
     if len(recs)%a.checkpoint_every==0 or len(recs)==len(rows):
      write(path,{'dataset':ds,'format':fmt,'revision':REV,'records':recs,'finished':len(recs)==len(rows)})
      print(f'{fmt} {ds}: {len(recs)}/{len(rows)}',flush=True)
    write(path,{'dataset':ds,'format':fmt,'revision':REV,'records':recs,'finished':len(recs)==len(rows)})
    if time.monotonic()>deadline:break
   del model;gc.collect();torch.cuda.empty_cache()
   report(sets,out)
   if time.monotonic()>deadline:break
 finally:
  s=report(sets,out);print('FINAL SUMMARY:',json.dumps(s['total'],indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();sys.exit(1)
