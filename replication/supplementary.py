"""Additional paper-baseline checks: D.1/D.3 and E.6 on frozen checkpoints.

These are independent realizations of the manuscript's input constructions,
not new research interventions and not a replay of the authors' input arrays.
The D.1 sampler draws a uniform hint and a uniform employee table conditioned
on that hint being absent. All same-position controls use a different token.
E.6 is restricted to the four ordinary/open/closed access groups; the separately
trained independent-message group is NOT implemented or claimed here.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from reproduction.core import Setting, World, draw, encode, canonical_numerics, logits_for, sha256, dump_json
from .run import pack_world, unpack_world, paired_summary
from .extensions import load_checkpoint
from .mechanism import capture, finish, patch, sub

D1_SEED = 2026092010
PAYLOAD_SEED = 2026092011
INPUT_CASES = ('base', 'employee_control', 'employee_witness', 'company_control', 'company_symbol', 'count_2', 'count_3', 'count_7')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def other_symbol(old: torch.Tensor, excluded: torch.Tensor, n: int, rng: torch.Generator):
    batch = len(old)
    allowed = torch.ones(batch, n, dtype=torch.bool)
    idx = torch.arange(batch)
    allowed[idx, old] = False
    allowed[idx, excluded] = False
    require(torch.all(old != excluded), 'Excluded and previous symbols must differ')
    pool = torch.arange(n).expand(batch, n)[allowed].reshape(batch, n - 2)
    return pool[idx, torch.randint(n - 2, (batch,), generator=rng)]


def input_worlds(size=4096):
    require(isinstance(size, int) and size > 1, 'Need at least two examples')
    s = Setting()
    rng = torch.Generator().manual_seed(D1_SEED)
    w = draw(s, size, rng)
    idx = torch.arange(size)
    w.hint = torch.randint(s.n, (size,), generator=rng)
    w.f = (w.hint[:, None] + torch.randint(1, s.n, (size, s.n), generator=rng)) % s.n
    w.two.fill_(True)
    w.present.fill_(True)
    scores = torch.rand(size, s.n, generator=rng)
    scores[idx, w.x] = 2
    rows = scores.argsort(1)[:, :s.n - 1]
    j = rows[:, 0]
    ec = w.copy()
    ec.f[idx, j] = other_symbol(w.f[idx, j], w.hint, s.n, rng)
    ew = w.copy()
    ew.f[idx, j] = w.hint
    contexts = {'base': w, 'employee_control': ec, 'employee_witness': ew}
    for count in (2, 3, 7):
        v = w.copy()
        v.f.scatter_(1, rows[:, :count], w.hint[:, None].expand(size, count))
        contexts[f'count_{count}'] = v
    keys = torch.arange(s.n).expand(size, s.n)
    unused = ~(w.f[:, :, None] == keys[:, None, :]).any(1)
    choices = unused & (keys != w.hint[:, None]) & (w.g != w.hint[:, None])
    eligible = choices.any(1)
    scores = torch.rand(size, s.n, generator=rng)
    scores[~choices] = 2
    company = scores.argmin(1)
    use = idx[eligible]
    cc, cs = w.copy(), w.copy()
    cc.g[use, company[eligible]] = other_symbol(w.g[use, company[eligible]], w.hint[eligible], s.n, rng)
    cs.g[use, company[eligible]] = w.hint[eligible]
    contexts.update(company_control=cc, company_symbol=cs)
    validate_input_worlds(contexts, rows, company, eligible)
    return contexts, rows, company, eligible


def validate_input_worlds(contexts, rows, company, eligible):
    w = contexts['base']
    n = len(w.x)
    idx = torch.arange(n)
    require(not (w.f == w.hint[:, None]).any(), 'The base already contains a witness')
    require(not (rows == w.x[:, None]).any(), 'The query was edited')
    require(torch.all(torch.sort(rows, dim=1).values.diff(dim=1) > 0), 'Repeated employee indices')
    for name, v in contexts.items():
        require(torch.equal(w.labels(), v.labels()), f'Target answer changed: {name}')
        for field in ('x', 'two', 'present', 'hint'):
            require(torch.equal(getattr(w, field), getattr(v, field)), f'{field} changed: {name}')
        require(torch.equal(w.f[idx, w.x], v.f[idx, v.x]), 'Queried employer changed')
    for name in ('employee_control', 'employee_witness'):
        v = contexts[name]
        require(torch.all((v.f != w.f).sum(1) == 1), 'Not a same-row one-token edit')
        require(torch.equal(v.g, w.g), 'Company table changed in employee edit')
    require(not (contexts['employee_control'].f == w.hint[:, None]).any(), 'Control creates a witness')
    for name, count in [('employee_witness', 1), ('count_2', 2), ('count_3', 3), ('count_7', 7)]:
        require(torch.all((contexts[name].f == w.hint[:, None]).sum(1) == count), 'Incorrect witness count')
    for name in ('company_control', 'company_symbol'):
        v = contexts[name]
        require(torch.equal(w.f, v.f), 'Employee table changed in company-symbol edit')
        require(torch.equal(w.answers(), v.answers()), 'Company-symbol edit changes a person-country answer')
        require(torch.equal(w.g[idx, w.hint], v.g[idx, w.hint]), 'Suggested answer changed')
        require(torch.all((v.g != w.g).sum(1) == eligible.long()), 'Company edit count mismatch')
    use = idx[eligible]
    require(not (w.f[use] == company[eligible, None]).any(), 'Edited company is used')


def payload_worlds(size=2048):
    require(isinstance(size, int) and size > 1, 'Need at least two examples')
    s = Setting()
    rng = torch.Generator().manual_seed(PAYLOAD_SEED)
    w = draw(s, size, rng)
    idx = torch.arange(size)
    w.two.fill_(True)
    w.present.fill_(True)
    w.hint = (w.f[idx, w.x] + torch.randint(1, s.n, (size,), generator=rng)) % s.n
    donor = w.copy()
    donor.g = torch.randint(s.n, (size, s.n), generator=rng)
    four = torch.stack((w.labels(), w.g[idx, w.hint], donor.labels(), donor.g[idx, w.hint]), 1)
    distinct = (four.sort(1).values.diff(dim=1) != 0).all(1)
    supported = (w.f == w.hint[:, None]).any(1)
    for field in ('f', 'x', 'two', 'present', 'hint'):
        require(torch.equal(getattr(w, field), getattr(donor, field)), 'Payload donor changed more than g')
    return w, donor, distinct, supported


def make_inputs(out: Path):
    contexts, rows, company, eligible = input_worlds()
    w, donor, distinct, supported = payload_worlds()
    data = {'input_' + k: pack_world(v) for k, v in contexts.items()}
    data.update(edited_rows=rows.numpy(), company_rows=company.numpy(), company_eligible=eligible.numpy(),
                payload_base=pack_world(w), payload_donor=pack_world(donor),
                payload_four_distinct=distinct.numpy(), payload_supported=supported.numpy())
    np.savez_compressed(out, **data)
    return {'D1_seed': D1_SEED, 'payload_seed': PAYLOAD_SEED,
            'D1_examples': 4096, 'D1_decisive': int((contexts['base'].labels() != contexts['base'].g[torch.arange(4096), contexts['base'].hint]).sum()),
            'D3_eligible': int(eligible.sum()), 'payload_examples': 2048,
            'payload_four_distinct': int(distinct.sum()), 'payload_four_distinct_supported': int((distinct & supported).sum()),
            'authors_original_arrays_replayed': False,
            'D1_sampling': 'uniform h; uniform f conditioned on h absent; fresh independent g and uniform x; same-position controls differ from old and hint tokens',
            'payload_scope': 'ordinary/open/closed access models; no separately trained independent-message group'}


def evaluate_inputs(checkpoint: str, inputs: str, destination: str):
    canonical_numerics(1)
    before = sha256(checkpoint)
    c, s, m = load_checkpoint(checkpoint)
    with np.load(inputs, allow_pickle=False) as z:
        worlds = {k: unpack_world(z['input_' + k]) for k in INPUT_CASES}
        eligible = z['company_eligible'].copy()
    w = worlds['base']
    truth = w.labels().numpy()
    wrong = w.g[torch.arange(len(w.x)), w.hint].numpy()
    decisive = truth != wrong
    common = decisive & eligible
    logits = {name: logits_for(m, encode(v, s)) for name, v in worlds.items()}
    def compare(a, b, mask):
        return paired_summary(logits[a][mask], logits[b][mask], wrong[mask], truth[mask])
    report = {'seed': c['seed'], 'step': c['step'], **asdict(s), 'checkpoint_sha256': before,
              'inputs_sha256': sha256(inputs), 'scope': 'D.1/D.3 manuscript tests on new fixed inputs',
              'populations': {'all': len(truth), 'decisive': int(decisive.sum()), 'D3_eligible': int(eligible.sum()), 'common_decisive': int(common.sum())},
              'D1_all': compare('employee_control', 'employee_witness', np.ones(len(truth), bool)),
              'D1_decisive': compare('employee_control', 'employee_witness', decisive),
              'D1_common': compare('employee_control', 'employee_witness', common),
              'D3_common': compare('company_control', 'company_symbol', common), 'multiplicity': {}}
    for name, mask in [('all', np.ones(len(truth), bool)), ('decisive', decisive)]:
        report['multiplicity'][name] = [{'count': count,
            'accuracy_pct': float(100 * (logits[key].argmax(1)[mask] == truth[mask]).mean()),
            'suggestion_following_pct': float(100 * (logits[key].argmax(1)[mask] == wrong[mask]).mean())}
            for count, key in [(0, 'base'), (1, 'employee_witness'), (2, 'count_2'), (3, 'count_3'), (7, 'count_7')]]
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'outputs.npz', **logits)
    require(before == sha256(checkpoint), 'A baseline checkpoint was modified')
    dump_json(out / 'report.json', report)
    return report


def evaluate_payload(checkpoint: str, inputs: str, destination: str):
    canonical_numerics(1)
    before = sha256(checkpoint)
    c, s, m = load_checkpoint(checkpoint)
    m = m.double().eval()
    with np.load(inputs, allow_pickle=False) as z:
        w, donor = unpack_world(z['payload_base']), unpack_world(z['payload_donor'])
        distinct, supported = z['payload_four_distinct'].copy(), z['payload_supported'].copy()
    names = ('base', 'full_donor', 'query_V', 'all_F_values', 'matching_F_values', 'other_F_values')
    logits = {k: [] for k in names}
    fstart = s.n if s.order == 'company_first' else 0
    fpos = slice(fstart, fstart + s.n)
    max_error = 0.0
    employee_value_difference = 0.0
    class_mismatches = 0
    blocked = (s.order == 'employee_first' and s.access == 'usual') or (s.order == 'company_first' and s.access == 'closed')
    for start in range(0, len(w.x), 128):
        end = min(start + 128, len(w.x))
        a, b = sub(w, start, end), sub(donor, start, end)
        xa, xb = encode(a, s), encode(b, s)
        ca, cb = capture(m, xa), capture(m, xb)
        base = finish(m, ca)
        with torch.no_grad():
            native = m(xa)
        max_error = max(max_error, float((base - native).abs().max()))
        class_mismatches += int((base.argmax(1) != native.argmax(1)).sum())
        diff = float((ca['v'][:, :, fpos] - cb['v'][:, :, fpos]).abs().max())
        employee_value_difference = max(employee_value_difference, diff)
        if blocked:
            require(diff == 0, 'A blocked employee position received company information')
        match = torch.zeros(len(a.x), s.length, dtype=torch.bool)
        match[:, fpos] = a.f == a.hint[:, None]
        other = torch.zeros_like(match)
        other[:, fpos] = True
        other[torch.arange(len(a.x)), fstart + a.x] = False
        values = {'query_V': patch(ca['v'], cb['v'], fstart + a.x),
                  'all_F_values': patch(ca['v'], cb['v'], fpos),
                  'matching_F_values': torch.where(match[:, None, :, None], cb['v'], ca['v']),
                  'other_F_values': torch.where(other[:, None, :, None], cb['v'], ca['v'])}
        logits['base'].append(base.numpy())
        logits['full_donor'].append(finish(m, cb).numpy())
        for key, value in values.items():
            result = finish(m, ca, {'v': value})
            if blocked:
                require(torch.equal(result, base), 'A structurally identical V patch changed the output')
            logits[key].append(result.numpy())
    require(max_error <= 5e-5 and class_mismatches == 0, 'Payload inference calibration failed')
    logits = {k: np.concatenate(v) for k, v in logits.items()}
    idx = torch.arange(len(w.x))
    T, B, Tp, Bp = w.labels().numpy(), w.g[idx, w.hint].numpy(), donor.labels().numpy(), donor.g[idx, donor.hint].numpy()
    summary = {}
    for population, mask in [('all', np.ones(len(T), bool)), ('four_distinct', distinct), ('four_distinct_supported', distinct & supported)]:
        require(mask.any(), 'Empty payload evaluation stratum')
        summary[population] = {'n': int(mask.sum()), 'conditions': {}}
        for key, ls in logits.items():
            p = ls.argmax(1)
            summary[population]['conditions'][key] = {
                'original_true_pct': float(100 * (p[mask] == T[mask]).mean()),
                'new_true_pct': float(100 * (p[mask] == Tp[mask]).mean()),
                'original_suggested_pct': float(100 * (p[mask] == B[mask]).mean()),
                'new_suggested_pct': float(100 * (p[mask] == Bp[mask]).mean()),
                'class_change_from_base_pct': float(100 * (p[mask] != logits['base'].argmax(1)[mask]).mean())}
    report = {'seed': c['seed'], 'step': c['step'], **asdict(s), 'checkpoint_sha256': before,
              'inputs_sha256': sha256(inputs), 'populations': summary,
              'validation': {'native_float64_max_error': max_error, 'class_mismatches': class_mismatches,
                  'max_employee_V_difference': employee_value_difference, 'blocked_employee_V_invariance_required': blocked},
              'scope': 'E.6-style V-only transfers with unchanged recipient Q/K/residual; independent-message training group not included'}
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'outputs.npz', **logits)
    require(before == sha256(checkpoint), 'A baseline checkpoint was modified')
    dump_json(out / 'report.json', report)
    return report


def run(execution: Path, out: Path, workers: int = 4):
    require(workers >= 1, 'Need a positive worker count')
    for group in ('primary', 'access'):
        require((execution / group / 'completed.json').is_file(), f'Incomplete {group}')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    inputs = out / 'inputs.npz'
    definitions = make_inputs(inputs)
    dump_json(out / 'input_definitions.json', definitions)
    results = {'input_controls': [], 'access_payload': []}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for group, name, function in [('primary', 'input_controls', evaluate_inputs), ('access', 'access_payload', evaluate_payload)]:
            checkpoints = sorted((execution / group).glob('models/*/model_*.pt'))
            require(len(checkpoints) == 64, 'Expected all 64 checkpoints per group')
            for cp in checkpoints:
                dest = out / name / (cp.parent.name + '__' + cp.stem)
                future = pool.submit(function, str(cp), str(inputs), str(dest))
                futures[future] = name
        for future in as_completed(futures):
            name = futures[future]
            report = future.result()
            results[name].append(report)
            print(f"supplementary {name} seed={report['seed']} step={report['step']} completed={len(results[name])}/64", flush=True)
    dump_json(out / 'summary.json', results)
    dump_json(out / 'completed.json', {'input_control_checkpoints': 64, 'access_payload_checkpoints': 64,
        'all_completed': True, 'source_sha256': sha256(Path(__file__)), 'inputs_sha256': sha256(inputs),
        'authors_original_inputs_replayed': False, 'new_extension_experiments': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.execution, args.output, args.workers)
