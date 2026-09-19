"""Unit tests for paper-only verification; no model training is invoked."""
import csv
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path

from reproduction.paper_checks import (bayes_joint, formula_joint, normalize, check_posterior,
                                       compositions, recover_count, table_audit, require)

ROOT = Path(__file__).resolve().parents[1]


class Counts(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(recover_count("0.000"), 0)

    def test_full(self):
        self.assertEqual(recover_count("100.000"), 4096)

    def test_quantization(self):
        self.assertEqual(recover_count("0.049"), 2)

    def test_negative_effect(self):
        self.assertEqual(recover_count("-0.024", signed=True), -1)

    def test_negative_rate_rejected(self):
        with self.assertRaises(ValueError):
            recover_count("-0.024")

    def test_impossible_percentage(self):
        with self.assertRaises(ValueError):
            recover_count("50.001")

    def test_ambiguous_percentage(self):
        with self.assertRaises(ValueError):
            recover_count("50.000", n=1000000)

    def test_bad_population(self):
        for n in (0, -1, 2.5, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                recover_count("50", n=n)

    def test_bad_precision(self):
        with self.assertRaises(ValueError):
            recover_count("50", decimals=-1)

    def test_unique_boundary_count_is_allowed(self):
        # 12.5 rounds to 12 under ties-to-even; the compatible count is unique.
        self.assertEqual(recover_count("12", n=8, decimals=0), 1)

    def test_table_all_32_rows(self):
        r = table_audit(ROOT / "configs/paper_table3.csv")
        self.assertEqual(r["rounded_fields_checked"], 128)
        self.assertEqual(r["effect_consistency_checks"], 32)
        self.assertEqual(r["figure1_group"]["control_pct"]["exact_fraction"], "112375/4096")
        self.assertFalse(r["model_replay"])
        self.assertFalse(r["joint_per_example_outcomes_recovered"])

    def test_table_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = (ROOT / "configs/paper_table3.csv").read_text().splitlines()
            lines[-1] = lines[-2]
            p = Path(tmp) / "bad.csv"; p.write_text("\n".join(lines) + "\n")
            with self.assertRaises(ValueError):
                table_audit(p)

    def test_inconsistent_effect_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = (ROOT / "configs/paper_table3.csv").read_text().replace("57.983", "0.000", 1)
            p = Path(tmp) / "bad.csv"; p.write_text(text)
            with self.assertRaises(ValueError):
                table_audit(p)


class Posterior(unittest.TestCase):
    f = (0, 1, 1)
    g = (0, 1, 2)

    def test_hand_calculation(self):
        self.assertEqual(normalize(bayes_joint(self.f, self.g, 1, Q(3, 4))), (Q(1, 13), Q(12, 13), Q(0)))

    def test_uninformative_hint(self):
        self.assertEqual(normalize(bayes_joint(self.f, self.g, 1, Q(1, 3))), (Q(1, 3), Q(2, 3), Q(0)))

    def test_absent_match_nonperfect_hint(self):
        self.assertEqual(normalize(bayes_joint(self.f, self.g, 2, Q(3, 4))), (Q(1, 3), Q(2, 3), Q(0)))

    def test_perfect_hint_without_match_is_undefined(self):
        self.assertIsNone(normalize(bayes_joint(self.f, self.g, 2, Q(1))))

    def test_always_wrong_hint_with_everyone_matching_is_undefined(self):
        self.assertIsNone(normalize(bayes_joint((1, 1, 1), self.g, 1, Q(0))))

    def test_perfect_hint_is_point_mass(self):
        self.assertEqual(normalize(bayes_joint(self.f, self.g, 1, Q(1))), (Q(0), Q(1), Q(0)))

    def test_exact_tie(self):
        self.assertEqual(check_posterior((0, 1), (0, 1), 0, Q(1, 2)), (True, True, True))

    def test_always_wrong_endpoint(self):
        self.assertEqual(normalize(bayes_joint(self.f, self.g, 1, Q(0))), (Q(1), Q(0), Q(0)))

    def test_country_collisions_are_aggregated(self):
        f, g = (0, 1, 2), (1, 1, 2)
        self.assertEqual(formula_joint(f, g, 0, Q(3, 4)), bayes_joint(f, g, 0, Q(3, 4)))
        self.assertEqual(normalize(formula_joint(f, g, 0, Q(3, 4))), (Q(0), Q(7, 8), Q(1, 8)))

    def test_float_alpha_rejected(self):
        with self.assertRaises(ValueError):
            bayes_joint(self.f, self.g, 1, 0.75)

    def test_invalid_maps_rejected(self):
        for f, g, h in (((0,), (0,), 0), ((0, 2), (0, 1), 0), ((0, 1), (0,), 0), ((0, 1), (0, 1), 2)):
            with self.subTest(f=f, g=g, h=h), self.assertRaises(ValueError):
                bayes_joint(f, g, h, Q(1, 2))

    def test_compositions_enumerate_without_duplicates(self):
        c = list(compositions(8, 8))
        self.assertEqual(len(c), 6435)
        self.assertEqual(len(set(c)), len(c))
        self.assertTrue(all(sum(v) == 8 and len(v) == 8 for v in c))

    def test_validation_not_an_assert_statement(self):
        with self.assertRaises(ValueError):
            require(False, "always checked, including under python -O")


if __name__ == "__main__":
    unittest.main()
