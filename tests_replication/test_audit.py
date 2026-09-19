import unittest
import numpy as np
import torch
from replication.audit import exact_recursive,require
from replication.assess import stats

class AuditTests(unittest.TestCase):
    def test_nested_optimizer_state_equal(self):
        x={'state':{0:{'step':torch.tensor(5.),'exp_avg':torch.arange(4.)}},'param_groups':[{'lr':.003,'params':[0]}]}
        exact_recursive(x,x)
    def test_changed_tensor_rejected(self):
        with self.assertRaises(ValueError):exact_recursive({'x':torch.zeros(2)},{'x':torch.ones(2)})
    def test_changed_dtype_rejected(self):
        with self.assertRaises(ValueError):exact_recursive(torch.ones(2),torch.ones(2,dtype=torch.float64))
    def test_missing_key_rejected(self):
        with self.assertRaises(ValueError):exact_recursive({'a':1},{'a':1,'b':2})
    def test_require_survives_optimization(self):
        with self.assertRaises(ValueError):require(False,'failed')
    def test_resampling_treats_models_as_units(self):
        s=stats([0,0,0,0,1,1,1,1])
        self.assertEqual(s['models'],8);self.assertEqual(s['mean'],.5)
        self.assertTrue(s['model_bootstrap_95ci'][0]<=.5<=s['model_bootstrap_95ci'][1])
    def test_invalid_group_statistics_rejected(self):
        for a in ([1],[1,np.nan]):
            with self.subTest(a=a),self.assertRaises(ValueError):stats(a)

if __name__=='__main__':unittest.main()
