import unittest
from replicate import choose,rates,paired,sha,VERDICT
class Tests(unittest.TestCase):
 def test_select(self):
  rows=[{'id':str(i),'gold':'safe' if i%2==0 else 'unsafe'} for i in range(200)]
  a=choose(rows,'beavertails_response',50,17);b=choose(rows,'beavertails_response',50,17)
  self.assertEqual(a,b);self.assertEqual(sum(x['gold']=='safe' for x in a),25)
 def test_pair(self):
  a=[{'id':'1','gold':'unsafe','binary':'unsafe'},{'id':'2','gold':'safe','binary':'unsafe'}]
  b=[{'id':'1','gold':'unsafe','binary':'safe'},{'id':'2','gold':'safe','binary':'safe'}]
  p=paired(a,b)
  self.assertEqual((p['flips'],p['improvements'],p['regressions'],p['unsafe_regressions']),(2,1,1,1))
 def test_rates(self):
  v=rates([{'gold':'unsafe','binary':'safe'},{'gold':'safe','binary':'unsafe'}])
  self.assertEqual((v['FN'],v['FP'],v['accuracy']),(1,1,0))
 def test_parse(self):
  self.assertEqual(VERDICT.search('unsafe\nS3').group(1),'unsafe')
  self.assertIsNone(VERDICT.search('output: unsafe'))
 def test_hash(self):self.assertEqual(sha({'a':1,'b':2}),sha({'b':2,'a':1}))
if __name__=='__main__':unittest.main()
