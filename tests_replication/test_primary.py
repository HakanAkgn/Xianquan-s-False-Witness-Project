import unittest
from dataclasses import replace
import numpy as np
import torch
from reproduction.core import *
from replication.run import *

class Primary(unittest.TestCase):
    @classmethod
    def setUpClass(cls): canonical_numerics(1)
    def test_parameter_count_and_clones(self):
        m=Model();self.assertEqual(sum(p.numel() for p in m.parameters()),69576)
        for p,q in zip(m.encoder.layers[0].parameters(),m.encoder.layers[1].parameters()):
            self.assertTrue(torch.equal(p,q));self.assertNotEqual(p.data_ptr(),q.data_ptr())
    def test_group_sizes(self):
        for g in ('primary','access'):
            self.assertEqual(len(group_jobs(g)),32)
            self.assertEqual(len(set(job_name(i,s) for i,s in group_jobs(g))),32)
    def test_all_layouts_roundtrip(self):
        w=draw(Setting(),128,torch.Generator().manual_seed(1))
        for layout in ('suggestion_last','person_last','fixed_suggestion','fixed_answer'):
            for order in ('company_first','employee_first'):
                s=Setting(order=order,layout=layout);z=decode(encode(w,s),s)
                self.assertTrue(torch.equal(w.labels(),z.labels()))
                self.assertTrue(torch.equal(encode(w,s),encode(z,s)))
    def test_semantic_serialization(self):
        w=draw(Setting(),128,torch.Generator().manual_seed(2));v=unpack_world(pack_world(w))
        for k in vars(w):self.assertTrue(torch.equal(getattr(w,k),getattr(v,k)))
    def test_task_distribution(self):
        w=draw(Setting(),100000,torch.Generator().manual_seed(3));i=torch.arange(100000)
        self.assertLess(abs(float(w.two.float().mean())-.75),.01)
        self.assertLess(abs(float(w.present.float().mean())-.5),.01)
        self.assertLess(abs(float((w.hint[w.present]==w.f[i,w.x][w.present]).float().mean())-.75),.01)
    def test_masks(self):
        for order,access in [('employee_first','open'),('company_first','closed')]:
            s=Setting(order=order,access=access);a,b=masks(s);c,d=masks(replace(s,access='usual'))
            self.assertEqual(int((a!=c).sum()),64);self.assertTrue(torch.equal(b,d))
    def test_float64_manual(self):
        w=draw(Setting(),16,torch.Generator().manual_seed(4))
        for s in [Setting(),Setting(order='employee_first',access='open'),Setting(access='closed')]:
            torch.manual_seed(5);m=Model(s).double().eval();x=encode(w,s)
            with torch.no_grad():self.assertLess(float((m(x)-m.manual(x)).abs().max()),1e-12)
    def test_d2_restrictions(self):
        s=Setting();base,a,b,j=equal_pairs(s,128,10)
        validate_equal_pair(encode(a,s).numpy(),encode(b,s).numpy(),s,True)
        self.assertTrue(torch.equal(a.answers(),b.answers()))
    def test_matching_initializations(self):
        hashes=[]
        for seed,s in group_jobs('primary')[:4]:
            torch.manual_seed(seed);hashes.append(state_hash(Model(s).state_dict()))
        self.assertEqual(len(set(hashes)),1)
    def test_wrong_hint_country_collision_is_retained(self):
        s=Setting();w=draw(s,4096,torch.Generator().manual_seed(9));i=torch.arange(4096)
        w.hint=(w.f[i,w.x]+1)%8
        self.assertTrue(bool((w.g[i,w.hint]==w.g[i,w.f[i,w.x]]).any()))

if __name__=='__main__':unittest.main()
