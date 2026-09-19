import unittest
import torch
from reproduction.core import *
from replication.mechanism import capture,finish,patch

class Mechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):canonical_numerics(1)
    def test_capture_native(self):
        w=draw(Setting(),16,torch.Generator().manual_seed(44))
        for s in [Setting(),Setting(order='employee_first',access='open'),Setting(access='closed')]:
            torch.manual_seed(23);m=Model(s).double().eval();x=encode(w,s)
            with torch.no_grad():self.assertLess(float((finish(m,capture(m,x))-m(x)).abs().max()),1e-12)
    def test_patch_single_row_no_leak(self):
        a=torch.arange(3*4*19*16).reshape(3,4,19,16);b=a+10000
        rows=torch.tensor([1,5,8]);c=patch(a,b,rows)
        changed=(a!=c).any(-1)
        self.assertEqual(int(changed.sum()),12)
        for i in range(3):self.assertTrue(torch.equal(c[i,:,rows[i]],b[i,:,rows[i]]))
        self.assertTrue(torch.equal(a,torch.arange(3*4*19*16).reshape(3,4,19,16)))
    def test_relocate_values(self):
        a=torch.zeros(3,4,19,16);b=torch.arange(a.numel(),dtype=a.dtype).reshape(a.shape)
        src=torch.tensor([3,5,7]);dst=torch.tensor([8,9,10]);c=patch(a,b,dst,src)
        for i in range(3):self.assertTrue(torch.equal(c[i,:,dst[i]],b[i,:,src[i]]))
    def test_patch_identity(self):
        a=torch.randn(16,4,19,16);self.assertTrue(torch.equal(a,patch(a,a,slice(8,16))))
    def test_no_future_question_leak(self):
        s=Setting();w=draw(s,32,torch.Generator().manual_seed(1));v=w.copy();v.present=~w.present
        m=Model(s).double().eval();a,b=capture(m,encode(w,s)),capture(m,encode(v,s))
        for key in ('k','v'):self.assertTrue(torch.equal(a[key][:,:,:16],b[key][:,:,:16]))

if __name__=='__main__':unittest.main()
