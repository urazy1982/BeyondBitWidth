import sys, unittest, json, tempfile, zipfile, copy
from pathlib import Path
sys.path.insert(0,'/mnt/data')
import guard_cpu_verify_raw as verifier

class CoreAuditTests(unittest.TestCase):
 def test_pair_rejects_different_lengths(self):
  a=[{'id':'a','gold':'unsafe','binary':'unsafe'}]
  with self.assertRaises(ValueError):verifier.pair(a,[])
 def test_pair_rejects_id_mismatch(self):
  a=[{'id':'a','gold':'unsafe','binary':'unsafe'}]
  b=[{'id':'b','gold':'unsafe','binary':'safe'}]
  with self.assertRaises(ValueError):verifier.pair(a,b)
 def test_pair_unsafe_miss(self):
  a=[{'id':'a','gold':'unsafe','binary':'unsafe','category':''}]
  b=[{'id':'a','gold':'unsafe','binary':'safe','category':''}]
  q=verifier.pair(a,b)
  self.assertEqual((q['flips'],q['improvements'],q['regressions'],q['unsafe_regressions'],q['unsafe_recoveries']),(1,0,1,1,0))
 def test_pair_unsafe_recovery(self):
  a=[{'id':'a','gold':'unsafe','binary':'safe','category':''}]
  b=[{'id':'a','gold':'unsafe','binary':'unsafe','category':''}]
  q=verifier.pair(a,b)
  self.assertEqual((q['flips'],q['improvements'],q['regressions'],q['unsafe_regressions'],q['unsafe_recoveries']),(1,1,0,0,1))
 def test_rate_counts_invalid(self):
  rs=[{'id':'a','gold':'unsafe','binary':'invalid'}, {'id':'b','gold':'safe','binary':'safe'}]
  x=verifier.rate(rs)
  self.assertEqual((x['n'],x['valid'],x['invalid'],x['safe'],x['unsafe']),(2,1,1,1,0))
 def test_hash_order(self):
  self.assertEqual(verifier.sha({'b':2,'a':1}),verifier.sha({'a':1,'b':2}))
 def test_parser(self):
  for raw,label in [('Safety: Safe','safe'),('Safety: Unsafe','unsafe'),('Safety: Controversial','controversial')]:
   self.assertEqual(verifier.LABEL_RE.search(raw).group(1).lower(),label)

 def test_complete_synthetic_archive_and_corruption(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td); entries={}; summary=[]; fp_summary=[]; int_summary=[]
   def put(key,obj):entries['five_results/'+key]=json.dumps(obj,sort_keys=True).encode()
   def make(name,n):
    rows=[];ars={}
    for i in range(n):
     gold='unsafe' if i%2==0 else 'safe'
     rows.append({'id':str(i),'gold':gold,'category':'cat','prompt_hash':'a'*64})
    selection_hash=verifier.sha(rows)
    man={'model':'Qwen/Qwen3Guard-Gen-4B','dataset':name,'n':n,
         'selection_sha256':selection_hash,'source':{'task':'response_safety',
          'label_counts':{'unsafe':sum(r['gold']=='unsafe' for r in rows),'safe':sum(r['gold']=='safe' for r in rows)}}}
    put(f'{name}/selection.json',rows);put(f'{name}/manifest.json',man)
    for arm in verifier.ARMS:
     recs=[]
     for i,r in enumerate(rows):
      gold=r['gold'];other='safe' if gold=='unsafe' else 'unsafe'
      p=gold if i%37 else other
      if arm=='nf4' and i%53==0:p=gold if p!=gold else other
      if arm=='fp4' and i%71==0:p=gold if p!=gold else other
      if arm=='fp8' and i%91==0:p=gold if p!=gold else other
      if arm=='int8' and i%113==0:p=gold if p!=gold else other
      recs.append({'id':r['id'],'gold':gold,'binary':p,'prediction':p,
                   'output':'Safety: '+p.title(),'category':'cat'})
     ars[arm]=recs
     put(f'{name}/{arm}.json',{'dataset':name,'arm':arm,
         'selection_sha256':selection_hash,'finished':True,'records':recs,
         'verification':{'model_revision':'test-revision'}})
    item={'dataset':name,'n':n,'source':man['source'],
          'arms':{k:verifier.rate(ars[k]) for k in ('bf16','nf4','fp8')},
          'pairs':{f'{a}_vs_{b}':verifier.pair(ars[a],ars[b]) for a,b in [('bf16','nf4'),('bf16','fp8'),('nf4','fp8')]}}
    put(f'{name}/summary.json',item);summary.append(item)
    for arm,target in [('fp4',fp_summary),('int8',int_summary)]:
     item2={'dataset':name,'source_task':man['source']['task'],'n':n,'selection_sha256':selection_hash,
            'nf4':verifier.rate(ars['nf4']),arm:verifier.rate(ars[arm])}
     for comp in ('bf16','nf4','fp8'):
      item2[comp]=verifier.rate(ars[comp]);item2[f'{comp}_vs_{arm}']=verifier.pair(ars[comp],ars[arm])
     put(f'{name}/{arm}_comparison.json',item2);target.append(item2)
   for name,n in verifier.DATASETS.items():make(name,n)
   put('cross_dataset_summary.json',{'datasets':summary})
   put('fp4_cross_dataset_summary.json',{'datasets':fp_summary})
   put('int8_cross_dataset_summary.json',{'datasets':int_summary})
   zpath=td/'test.zip'
   with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for k,v in entries.items():z.writestr(k,v)
   result=verifier.verify(zpath)
   self.assertEqual(result['status'],'PASS',result['failures'][:3])
   self.assertEqual(result['records_verified'],9224*5)
   # Corrupt one INT8 record's gold label while keeping the ZIP CRC valid.
   obj=json.loads(entries['five_results/qwen_thinking/int8.json'])
   obj['records'][0]['gold']='safe' if obj['records'][0]['gold']=='unsafe' else 'unsafe'
   entries['five_results/qwen_thinking/int8.json']=json.dumps(obj).encode()
   with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for k,v in entries.items():z.writestr(k,v)
   altered=verifier.verify(zpath)
   self.assertEqual(altered['status'],'FAIL')
   self.assertTrue(any('ID/gold mismatch' in f for f in altered['failures']))

if __name__=='__main__':unittest.main(verbosity=2)
