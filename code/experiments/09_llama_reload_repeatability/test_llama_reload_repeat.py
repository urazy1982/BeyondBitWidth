import tempfile,unittest
from pathlib import Path
import llama_reload_repeat as l

class ReloadTests(unittest.TestCase):
    def test_binomial(self):
        self.assertAlmostEqual(l.exact_binomial_two_sided(0,8),2/256)
        self.assertEqual(l.exact_binomial_two_sided(2,4),1.)
        self.assertEqual(l.exact_binomial_two_sided(0,0),1.)
    def test_compare(self):
        a=[{'id':'1','gold':'unsafe','binary':'unsafe'}, {'id':'2','gold':'safe','binary':'safe'}, {'id':'3','gold':'unsafe','binary':'safe'}]
        b=[{'id':'1','gold':'unsafe','binary':'safe'}, {'id':'2','gold':'safe','binary':'safe'}, {'id':'3','gold':'unsafe','binary':'unsafe'}]
        c=l.compare(a,b)
        self.assertEqual(c['changed'],2)
        self.assertEqual(c['unsafe_regressions'],1)
        self.assertEqual(c['unsafe_recoveries'],1)
    def test_file_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'hello')
            self.assertEqual(l.file_digest(p),hashlib.sha256(b'hello').hexdigest())

if __name__=='__main__':unittest.main()
