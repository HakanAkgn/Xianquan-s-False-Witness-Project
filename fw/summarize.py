"""Aggregate completed runs without selecting seeds or checkpoints.

The model seed, not a test example, is the replication unit. The full raw
predictions and probability tensors remain in outputs/ for the execution archive.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .evaluate import unpack_outcomes, paired_stats, pack_outcomes
from .theory import theory_checks

SEEDS = (11, 29, 47)
STEPS = (1000, 2000, 4000)
ARMS = ('iid', 'augmentation', 'consistency')


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def seed_summary(values):
    x = np.asarray(values, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    half = t.ppf(.975, len(x)-1) * se
    return {'n_model_seeds': len(x), 'mean': float(x.mean()),
            'sample_sd': float(x.std(ddof=1)),
            'student_t_95ci': [float(x.mean()-half), float(x.mean()+half)],
            'scope': 'descriptive pilot interval across fixed model seeds; not example-level replication'}


def aggregate(root, destination):
    root, dest = Path(root), Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    baseline, balanced, cubes, curves, repairs = [], [], [], [], []
    summaries, input_files, decision_audit = {}, [], {}

    def load(path):
        input_files.append(path)
        return read(path)

    def cube_rows(path, task, seed, step, arm='baseline'):
        c = load(path)
        for mode in ('incorrect', 'absent', 'correct'):
            r = c[mode]
            row = {'task': task, 'seed': seed, 'step': step, 'arm': arm, 'hint': mode}
            for key in ('worlds', 'contexts', 'base_accuracy', 'mean_orbit_accuracy',
                        'worst_orbit_accuracy', 'base_correct_but_fragile_fraction',
                        'prediction_changes_somewhere_fraction', 'margin_variance',
                        'higher_order_fraction_nonconstant_margin_energy',
                        'additive_singleton_heldout_margin_rmse',
                        'constant_baseline_heldout_margin_rmse', 'parseval_max_error',
                        'poincare_lower_min_slack', 'poincare_upper_min_slack', 'jensen_gain'):
                row[key] = r[key]
            cubes.append(row)
            for point in r['count_curve']:
                curves.append({'task': task, 'seed': seed, 'step': step,
                               'arm': arm, 'hint': mode, **point})
        return c

    def balanced_rows(path, seed, step, arm='baseline'):
        b = load(path)
        for mode in ('incorrect', 'absent', 'correct'):
            r = b[mode]
            row = {'seed': seed, 'step': step, 'arm': arm, 'hint': mode}
            for key in ('n', 'control_rate', 'witness_rate', 'effect', 'paired_se',
                        'became_wrong', 'became_not_wrong', 'both_wrong', 'neither_wrong'):
                row[key] = r[key]
            row['paired_ci_low'], row['paired_ci_high'] = r['paired_normal_95ci']
            row.update({f'accuracy_{k}': v for k, v in r['accuracy'].items()})
            row.update(r['factorial_margin'])
            balanced.append(row)
        return b

    # Require the entire planned baseline grid; missing runs are a hard error.
    for task in ('original', 'typed'):
        for seed in SEEDS:
            folder = root/task/f'seed_{seed}'
            for step in STEPS:
                r = load(folder/f'report_{step}.json')
                for key, p in r['pairs'].items():
                    baseline.append({'task': task, 'seed': seed, 'step': step,
                                     'condition': key, 'clean_no_hint_accuracy': r['clean']['absent']['accuracy'],
                                     'clean_correct_hint_accuracy': r['clean']['correct']['accuracy'],
                                     **{k: p[k] for k in ('n', 'control_rate', 'witness_rate', 'effect',
                                         'paired_se', 'became_wrong', 'became_not_wrong', 'both_wrong', 'neither_wrong')}})
                if task == 'original':
                    b = balanced_rows(folder/f'balanced_{step}/balanced_report.json', seed, step)
                    if step == 2000:
                        packed = load(folder/f'balanced_{step}/balanced_outcomes.json')
                        # Public compact audit retains event identity, not the full 8-class predictions.
                        arr = unpack_outcomes(packed['incorrect'])
                        y, wrong = arr[:2]
                        pred = arr[2:]
                        events = np.where(pred == wrong, 2, np.where(pred == y, 1, 0)).astype('uint8')
                        rebuilt = paired_stats(events[0] == 2, events[3] == 2)
                        for k in ('n', 'became_wrong', 'became_not_wrong', 'both_wrong', 'neither_wrong', 'effect'):
                            if rebuilt[k] != b['incorrect'][k]:
                                raise AssertionError(f'Raw outcome audit failed: seed {seed}, {k}')
                        decision_audit[str(seed)] = pack_outcomes(events)
            for step in ((2000, 4000) if task == 'original' else (4000,)):
                cube_rows(folder/f'orbit_{step}/cube_report.json', task, seed, step)

    for seed in SEEDS:
        for arm in ARMS:
            folder = root/'repair'/arm/f'seed_{seed}'
            r = load(folder/'report.json')
            b = balanced_rows(folder/'balanced_report.json', seed, 2500, arm)
            c = cube_rows(folder/'orbit/cube_report.json', 'original', seed, 2500, arm)
            repairs.append({'seed': seed, 'arm': arm, 'base_updates': 2000, 'additional_updates': 500,
                            'supervised_contexts_per_update': 512,
                            'clean_no_hint_accuracy': r['clean']['absent']['accuracy'],
                            'clean_correct_hint_accuracy': r['clean']['correct']['accuracy'],
                            'iid_wrong_hint_accuracy': r['clean']['incorrect']['accuracy'],
                            'relevant_changed_answer_accuracy': r['sensitivity']['relevant_changed_answer_accuracy'],
                            'balanced_effect': b['incorrect']['effect'],
                            'balanced_wrong_rate': b['incorrect']['witness_rate'],
                            'cube_mean_accuracy': c['incorrect']['mean_orbit_accuracy'],
                            'cube_robust_accuracy': c['incorrect']['worst_orbit_accuracy'],
                            'cube_no_hint_robust_accuracy': c['absent']['worst_orbit_accuracy'],
                            'cube_correct_hint_robust_accuracy': c['correct']['worst_orbit_accuracy']})

    for name, rows in [('baseline', baseline), ('balanced', balanced), ('cubes', cubes),
                       ('cube_curves', curves), ('repairs', repairs)]:
        write_csv(dest/f'{name}.csv', rows)
    selected = [r for r in balanced if r['step'] == 2000 and r['arm'] == 'baseline' and r['hint'] == 'incorrect']
    summaries['balanced_2000_effect'] = seed_summary([r['effect'] for r in selected])
    for arm in ARMS:
        summaries[f'{arm}_cube_robust_accuracy'] = seed_summary([r['cube_robust_accuracy'] for r in repairs if r['arm'] == arm])
    summaries['status'] = {'baseline_models': 6, 'baseline_checkpoints': 18,
                            'repair_runs': 9, 'complete_cube_evaluations': len(cubes)//3,
                            'source': 'independent reconstruction; upstream code/checkpoints not found',
                            'typed_task': 'underlearned; not evidence of high-competence binding failure'}
    summaries['numerical_identity_checks'] = theory_checks()
    summaries['maximum_cube_parseval_error'] = max(r['parseval_max_error'] for r in cubes)
    summaries['minimum_cube_jensen_gain'] = min(r['jensen_gain'] for r in cubes)
    summaries['raw_decision_audit'] = 'passed for all three balanced 2,000-update checkpoints'
    (dest/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    audit = {'encoding': 'events: 0=other answer, 1=true answer, 2=false-suggestion answer',
             'rows': ['control', 'employee_edit_only', 'unused_company_edit_only', 'balanced'],
             'evaluation_seed': 418729, 'n_pairs_per_seed': 4096, 'checkpoint_updates': 2000,
             'models': decision_audit}
    (dest/'balanced_2000_event_audit.json').write_text(json.dumps(audit, separators=(',', ':'))+'\n')
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(input_files))}
    (dest/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', default='outputs')
    p.add_argument('--destination', default='results')
    args = p.parse_args()
    print(json.dumps(aggregate(args.outputs, args.destination), indent=2))

if __name__ == '__main__':
    main()
