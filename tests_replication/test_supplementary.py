import unittest
import torch
from reproduction.core import Setting, Model, canonical_numerics, encode
from replication.mechanism import capture, finish, patch
from replication.supplementary import input_worlds, payload_worlds, validate_input_worlds, require


class Supplementary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        canonical_numerics(1)
        cls.worlds, cls.rows, cls.company, cls.eligible = input_worlds(256)

    def test_initial_hint_absent(self):
        w = self.worlds['base']
        self.assertFalse(bool((w.f == w.hint[:, None]).any()))

    def test_same_employee_row(self):
        w = self.worlds['base']
        a = self.worlds['employee_control']
        b = self.worlds['employee_witness']
        self.assertTrue(torch.equal(a.f != w.f, b.f != w.f))
        self.assertTrue(torch.all((a.f != w.f).sum(1) == 1))

    def test_all_targets_preserved(self):
        w = self.worlds['base']
        for v in self.worlds.values():
            self.assertTrue(torch.equal(w.labels(), v.labels()))

    def test_company_controls_preserve_function(self):
        w = self.worlds['base']
        for name in ('company_control', 'company_symbol'):
            self.assertTrue(torch.equal(w.answers(), self.worlds[name].answers()))

    def test_ineligible_company_rows_unchanged(self):
        w = self.worlds['base']
        for name in ('company_control', 'company_symbol'):
            self.assertTrue(torch.equal(w.g[~self.eligible], self.worlds[name].g[~self.eligible]))

    def test_multiplicity(self):
        w = self.worlds['base']
        for count in (2, 3, 7):
            self.assertTrue(torch.all((self.worlds[f'count_{count}'].f == w.hint[:, None]).sum(1) == count))

    def test_corrupt_query_rejected(self):
        d = {k: w.copy() for k, w in self.worlds.items()}
        d['employee_witness'].x[0] = (d['employee_witness'].x[0] + 1) % 8
        with self.assertRaises(ValueError):
            validate_input_worlds(d, self.rows, self.company, self.eligible)

    def test_payload_changes_only_company_table(self):
        a, b, _, _ = payload_worlds(128)
        for field in ('f', 'x', 'two', 'hint', 'present'):
            self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)))

    def test_four_distinct_population(self):
        a, b, distinct, supported = payload_worlds(128)
        idx = torch.arange(128)
        vals = torch.stack((a.labels(), a.g[idx, a.hint], b.labels(), b.g[idx, b.hint]), 1)
        for i in range(128):
            self.assertEqual(bool(distinct[i]), len(set(vals[i].tolist())) == 4)
            self.assertEqual(bool(supported[i]), bool((a.f[i] == a.hint[i]).any()))

    def test_payload_suggestion_is_wrong(self):
        a, _, _, _ = payload_worlds(128)
        self.assertTrue(torch.all(a.hint != a.f[torch.arange(128), a.x]))

    def test_blocked_values_and_outputs_are_identical(self):
        a, b, _, _ = payload_worlds(32)
        for s in (Setting(order='employee_first'), Setting(access='closed')):
            torch.manual_seed(900)
            m = Model(s).double().eval()
            ca, cb = capture(m, encode(a, s)), capture(m, encode(b, s))
            pos = slice(0, 8) if s.order == 'employee_first' else slice(8, 16)
            self.assertTrue(torch.equal(ca['v'][:, :, pos], cb['v'][:, :, pos]))
            self.assertTrue(torch.equal(finish(m, ca), finish(m, ca, {'v': patch(ca['v'], cb['v'], pos)})))

    def test_open_values_respond_to_company_table(self):
        a, b, _, _ = payload_worlds(32)
        for s in (Setting(order='employee_first', access='open'), Setting()):
            torch.manual_seed(900)
            m = Model(s).double().eval()
            ca, cb = capture(m, encode(a, s)), capture(m, encode(b, s))
            pos = slice(0, 8) if s.order == 'employee_first' else slice(8, 16)
            self.assertGreater(float((ca['v'][:, :, pos] - cb['v'][:, :, pos]).abs().max()), 0)

    def test_validation_survives_optimization(self):
        with self.assertRaises(ValueError):
            require(False, 'invalid input')

    def test_invalid_population(self):
        for n in (0, 1):
            with self.assertRaises(ValueError):
                input_worlds(n)


if __name__ == '__main__':
    unittest.main()
