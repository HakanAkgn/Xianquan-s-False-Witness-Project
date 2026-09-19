import unittest
import numpy as np
import torch
from reproduction.core import *
from replication.repair import *

class Repair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):canonical_numerics(1)
    def setup_world(self):return draw(Setting(),512,torch.Generator().manual_seed(66))
    def batches(self,arm):return batch_for(Setting(),self.setup_world(),arm,torch.Generator().manual_seed(71),torch.Generator().manual_seed(72))
    def test_same_data_for_js(self):
        a,b=self.batches('augmentation'),self.batches('consistency')
        for x,y in zip(a,b):self.assertTrue(np.array_equal(pack_world(x),pack_world(y)))
    def test_exposure_matched(self):
        a,b=self.batches('augmentation'),self.batches('hint_frequency_control')
        for x,y in zip(a,b):
            i=torch.arange(len(x.x));self.assertTrue(torch.equal(x.two,y.two));self.assertTrue(torch.equal(x.present,y.present))
            self.assertTrue(torch.equal(x.hint==x.f[i,x.x],y.hint==y.f[i,y.x]))
    def test_preserves_query_labels(self):
        _,a,b,c=self.batches('augmentation')
        self.assertTrue(torch.equal(a.labels(),b.labels()));self.assertTrue(torch.all(a.labels()!=c.labels()))
    def test_hint_control_country_decisive(self):
        _,a,b,c=self.batches('hint_frequency_control')
        for w in (a,b):
            i=torch.arange(len(w.x));self.assertTrue(torch.all(w.g[i,w.hint]!=w.g[i,w.f[i,w.x]]))
    def test_ordinary_unmodified(self):
        w=self.setup_world();a=batch_for(Setting(),w,'iid',torch.Generator(),torch.Generator())
        self.assertEqual(len(a),1);self.assertIs(a[0],w)
    def test_js_identity(self):
        x=torch.randn(16,8,dtype=torch.float64);self.assertLess(abs(float(js_divergence(x,x))),1e-14)
    def test_js_nonnegative_symmetric(self):
        a,b=torch.randn(16,8,dtype=torch.float64),torch.randn(16,8,dtype=torch.float64)
        self.assertGreaterEqual(float(js_divergence(a,b)),0)
        self.assertAlmostEqual(float(js_divergence(a,b)),float(js_divergence(b,a)),places=14)
    def test_no_extra_vocabulary(self):
        for arm in ARMS:
            t=torch.cat([encode(w,Setting()) for w in self.batches(arm)])
            self.assertEqual(tuple(t.shape),(512,19));self.assertLess(int(t.max()),12)

if __name__=='__main__':unittest.main()
