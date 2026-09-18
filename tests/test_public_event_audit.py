"""Checks of an independent replay of the public event archive."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_event_audit", ROOT / "scripts/audit_public_events.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EVENTS = ROOT / "results/balanced_2000_event_audit.json"


class PublicEventAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = MODULE.audit(EVENTS)

    def test_source_blob_identity(self):
        self.assertEqual(MODULE.git_blob_sha(EVENTS.read_bytes()), MODULE.EXPECTED_BLOB)

    def test_reported_effects(self):
        expected = {"11": 0.78125, "29": 39.5263671875, "47": 0.0732421875}
        for seed, value in expected.items():
            self.assertEqual(self.result["models"][seed]["effect_percentage_points"], value)

    def test_contingency_totals(self):
        for record in self.result["models"].values():
            a = record["control_to_balanced_contingency_0_other_1_true_2_suggested"]
            self.assertEqual(sum(map(sum, a)), 4096)
            self.assertEqual(sum(a[2]), record["suggested_answer_counts_by_condition"]["control"])
            self.assertEqual(sum(r[2] for r in a), record["suggested_answer_counts_by_condition"]["balanced"])

    def test_paired_changes(self):
        for record in self.result["models"].values():
            diff = record["gained_suggested_answer"] - record["lost_suggested_answer"]
            self.assertEqual(diff * 100 / 4096, record["effect_percentage_points"])

    def test_conditional_interval(self):
        low, high = self.result["models"]["29"]["paired_normal_95_percent_interval_points"]
        self.assertAlmostEqual(low, 38.019784695871465, places=10)
        self.assertAlmostEqual(high, 41.032949679128535, places=10)

    def test_altered_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "events.json"
            changed.write_bytes(EVENTS.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                MODULE.audit(changed)


if __name__ == "__main__":
    unittest.main()
