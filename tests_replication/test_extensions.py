import unittest
import numpy as np
import torch
from reproduction.core import *
from replication.extensions import *

class Extensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):canonical_numerics(1)
    def test_balanced_all_layouts(self):
        for layout in ('suggestion_last','person_last'):
            for order in ('company_first','employee_first'):
                s=Setting(order=order,layout=layout);w,r,d=balanced_world(s,128,torch.Generator().manual_seed(1))
                validate(w,edit(w,r,d,1),s)
    def test_complete_cube(self):
        s=Setting();w,r,d=balanced_world(s,64,torch.Generator().manual_seed(2),4)
        for mask in range(16):
            v=edit(w,r,d,mask);validate(w,v,s)
            self.assertTrue(torch.all((v.f==w.hint[:,None]).sum(1)==mask.bit_count()))
    def test_all_compensation_unused(self):
        s=Setting();w,r,d=balanced_world(s,128,torch.Generator().manual_seed(3),4)
        for mask in range(16):
            v=edit(w,r,d,mask)
            self.assertFalse(bool((v.f[:,:,None]==d[:,None,:]).any()))
    def test_factorial_control_not_token_balanced(self):
        s=Setting();w,r,d=balanced_world(s,128,torch.Generator().manual_seed(4))
        for e,g in ((True,False),(False,True)):
            v=edit(w,r,d,1,e,g);validate(w,v,s,False)
            self.assertFalse(torch.equal(encode(w,s).sort(1).values,encode(v,s).sort(1).values))
    def test_relevant_changes_one_and_two(self):
        s=Setting();w,r,d=balanced_world(s,128,torch.Generator().manual_seed(5))
        for two in (False,True):
            w.two.fill_(two);v=relevant(w)
            self.assertTrue(bool((v.labels()!=w.labels()).all()))
    def test_walsh_roundtrip(self):
        x=np.random.default_rng(0).normal(size=(16,23))
        self.assertLess(np.max(np.abs(walsh(walsh(x))*16-x)),1e-12)
    def test_additive_predictor(self):
        bits=np.array([[(m>>j)&1 for j in range(4)] for m in range(16)])
        margins=(bits@np.array([1.,2.,3.,4.]))[:,None]+np.arange(10)[None,:]
        l=np.stack([np.zeros_like(margins),margins],-1)
        r=cube_stats(l,np.zeros(10,int),np.ones(10,int))
        self.assertEqual(r['additive_heldout_rmse'],0)
        self.assertLess(r['higher_order_share'],1e-12)
    def test_interaction_detected(self):
        margins=np.array([3*((m&1)>0)*((m&2)>0) for m in range(16)])[:,None]*np.ones((1,8))
        l=np.stack([np.zeros_like(margins),margins],-1);r=cube_stats(l,np.zeros(8,int),np.ones(8,int))
        self.assertGreater(r['higher_order_share'],0)
        self.assertGreater(r['additive_heldout_rmse'],0)
    def test_invalid_parameters(self):
        for k in (0,6):
            with self.assertRaises(ValueError):balanced_world(Setting(),2,torch.Generator(),k)
    def test_tokens_out_of_answer_range_excluded(self):
        w,r,d=balanced_world(Setting(),128,torch.Generator().manual_seed(6),1)
        idx=torch.arange(len(w.x));h=w.hint;rr=w.f[idx,r[:,0]];a=w.labels();b=w.g[idx,h]
        self.assertFalse(bool(((a==h)|(a==rr)|(b==h)|(b==rr)).any()))

if __name__=='__main__':unittest.main()
