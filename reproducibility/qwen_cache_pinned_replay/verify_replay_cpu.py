#!/usr/bin/env python3
"""Independent, standard-library-only audit of targeted replay vs ORIGINAL primary records.
Deliberately does not trust generated comparison_summary.json or frozen/reference.json as authority.
"""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
REV='6ec42827da0c1ff11e7a49dc269d2e810d27e108'
DATASETS=('qwen_thinking','toxicchat_prompt','xstest_prompt','beavertails_response')
FIELDS=('id','gold','prediction','binary','output')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 total=flips=controls=matched=0
 identity=load(HERE/'results'/'run_identity.json')
 assert identity['revision']==REV and identity['greedy'] is True and identity['max_new_tokens']==48
 assert identity['torch']=='2.11.0+cu128' and identity['transformers']=='5.22.2' and identity['bitsandbytes']=='0.50.2'
 for fmt in ('nf4','fp4'):
  prov=load(HERE/'results'/(fmt+'_provenance.json'))
  assert prov['pinned_revision']==REV and prov['layers']==252 and fmt in prov['quant_types_seen']
 for ds in DATASETS:
  meta=load(HERE/'frozen'/ds/'subset_metadata.json')
  selection=load(HERE/'frozen'/ds/'selection.json')
  original={};fresh={}
  ids=[str(r['id']) for r in selection]
  assert len(ids)==len(set(ids))==meta['n'] and len(meta['changed_ids'])==meta['flips']
  assert len(meta['control_ids'])==meta['controls']==32
  assert set(meta['control_ids']).isdisjoint(meta['changed_ids'])
  assert set(ids)==set(meta['control_ids'])|set(meta['changed_ids'])
  for fmt in ('nf4','fp4'):
   orig=load(REPO/'results'/'five_results'/ds/(fmt+'.json'))
   record=load(HERE/'results'/ds/(fmt+'.json'))
   assert orig.get('finished') and record.get('finished')
   assert record['revision']==REV and record['format']==fmt
   original[fmt]={str(r['id']):r for r in orig['records']}
   fresh[fmt]={str(r['id']):r for r in record['records']}
   assert len(fresh[fmt])==len(ids) and set(fresh[fmt])==set(ids)
   for r in record['records']:
    old=original[fmt][str(r['id'])]
    assert all(r[k]==old[k] for k in FIELDS), (ds,fmt,r['id'],'record mismatch')
    matched+=1
  historical_flip={i for i in ids if original['nf4'][i]['binary']!=original['fp4'][i]['binary']}
  new_flip={i for i in ids if fresh['nf4'][i]['binary']!=fresh['fp4'][i]['binary']}
  assert historical_flip==new_flip==set(meta['changed_ids'])
  # verify no historical flips were omitted from the full original dataset
  full_flips={i for i in original['nf4'] if original['nf4'][i]['binary']!=original['fp4'][i]['binary']}
  assert full_flips==historical_flip, (ds,'flips omitted',len(full_flips),len(historical_flip))
  total+=len(ids);flips+=len(historical_flip);controls+=meta['controls']
  print(f'PASS {ds}: {len(ids)} inputs, {len(historical_flip)} historical flips; {len(ids)*2} exact record matches')
 assert (total,flips,controls,matched)==(280,152,128,560)
 print('PASS pinned Qwen replay: 280 inputs, 152/152 historical flips, 128 controls, 560/560 exact predictions and raw outputs')
 print('LIMITATION: sampled replay supports recovered snapshot but does not prove historical checkpoint identity')
if __name__=='__main__':
 try:main()
 except Exception as e:
  print('FAIL:',repr(e),file=sys.stderr);raise
