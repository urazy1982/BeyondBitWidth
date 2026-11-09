import unittest
from run_five import source_record, rate, compare, sha
class LogicTest(unittest.TestCase):
 def test_wildguard_prompt_only(self):
  v=source_record('wildguard_prompt',{'prompt':'hi','prompt_harm_label':'unharmful','response':'anything'},0)
  self.assertEqual(v['gold'],'safe');self.assertEqual(len(v['messages']),1)
 def test_beaver_response(self):
  v=source_record('beavertails_response',{'prompt':'hi','response':'hello','is_safe':True},3)
  self.assertEqual(v['gold'],'safe');self.assertEqual(len(v['messages']),2)
 def test_toxicity(self):
  self.assertEqual(source_record('toxicchat_prompt',{'user_input':'x','toxicity':1},0)['gold'],'unsafe')
 def test_pairs(self):
  a=[{'id':'1','gold':'unsafe','binary':'unsafe','category':'c'},{'id':'2','gold':'safe','binary':'safe','category':'c'}]
  b=[{'id':'1','gold':'unsafe','binary':'safe','category':'c'},{'id':'2','gold':'safe','binary':'safe','category':'c'}]
  self.assertEqual(compare(a,b)['unsafe_regressions'],1);self.assertEqual(rate(b)['FN'],1)
 def test_hash_stability(self):self.assertEqual(sha({'a':1,'b':2}),sha({'b':2,'a':1}))
if __name__=='__main__':unittest.main()
