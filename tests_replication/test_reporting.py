import tempfile
import unittest
from pathlib import Path
from replication.assess import group_summary,read_baseline
from replication.summarize import run

class Reporting(unittest.TestCase):
    def test_group_means_do_not_weight_test_examples(self):
        rows=[{'group':'a','seed':i,'value':i,'examples':4096+i} for i in range(8)]
        r=group_summary(rows,('group',),['value'])
        self.assertEqual(r[0]['statistics']['value']['models'],8)
        self.assertEqual(r[0]['statistics']['value']['mean'],3.5)
    def test_incomplete_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):read_baseline(Path(d))
    def test_report_does_not_infer_execution_from_empty_directories(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ('primary','access','mechanism','extensions','repair'):(p/name).mkdir()
            with self.assertRaises(ValueError):run(p,p/'summary')
            self.assertFalse((p/'summary').exists())

if __name__=='__main__':unittest.main()
