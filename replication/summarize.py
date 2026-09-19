"""Assemble completed measurements without dropping null or negative results.

All CSV percentages are on a 0--100 scale. Effects are percentage points.
Model bootstrap intervals resample independent training seeds, not test cases.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from reproduction.core import dump_json,sha256
from .assess import stats,group_summary,csv_write
from .repair import ARMS


def read(path:Path):
    return json.loads(path.read_text())


def mean(rows,key):
    return float(np.mean([r[key] for r in rows]))


def run(execution:Path,out:Path):
    if out.exists():raise FileExistsError(out)
    for stage in ('primary','access','mechanism','extensions','repair'):
        if not (execution/stage/'completed.json').is_file():raise ValueError(f'Incomplete {stage}')
    baseline=read(execution/'assessment/baseline_assessment.json')
    extension=read(execution/'extensions/summary.json');repair=read(execution/'repair/summary.json')
    audit=read(execution/'audit/audit.json')
    if 'training_replay' not in audit:raise ValueError('The complete report requires the training replay audit')
    if len(extension)!=64 or len(repair)!=32 or not audit['all_checks_passed']:raise ValueError('Incomplete evidence')
    out.mkdir(parents=True)
    for p in (execution/'assessment').glob('*.csv'):shutil.copy2(p,out/p.name)
    b=[];c=[];rr=[]
    for r in extension:
        setting={k:r[k] for k in ('seed','step','order','layout','access')}
        for mode,z in r['balanced'].items():
            row={**setting,'hint':mode}
            row.update({k:z[k] for k in ('control_following_pct','witness_following_pct','effect_pp',
                'control_accuracy_pct','witness_accuracy_pct','margin_change','employee_only_margin_effect',
                'compensation_only_margin_effect','factorial_margin_interaction')})
            row['paired_95ci_low_pp'],row['paired_95ci_high_pp']=z['paired_95ci_pp']
            b.append(row)
        for mode,z in r.get('cube',{}).items():
            row={**setting,'hint':mode}
            row.update({k:z[k] for k in ('robust_accuracy_pct','mean_accuracy_pct','base_accuracy_pct',
                'margin_variance','higher_order_share','additive_heldout_rmse','constant_heldout_rmse')})
            row['higher_order_share_pct']=100*z['higher_order_share'] if z['higher_order_share'] is not None else None
            for v in z['count_curve']:row['wrong_count_'+str(v['count'])+'_pct']=v['designated_wrong_pct']
            c.append(row)
    for r in repair:
        row={'seed':r['seed'],'arm':r['arm'],'base_step':2000,'extra_updates':r['extra_updates'],'batch_size':r['batch_size']}
        for k,v in r['natural'].items():row[k+'_accuracy_pct']=v['accuracy_pct']
        row.update({'relevant_changed_accuracy_pct':r['sensitivity']['changed_answer_accuracy_pct'],
            'balanced_effect_pp':r['balanced']['incorrect']['effect_pp'],
            'balanced_wrong_pct':r['balanced']['incorrect']['witness_following_pct']})
        for mode,z in r['cube'].items():
            row[mode+'_cube_robust_accuracy_pct']=z['robust_accuracy_pct']
            row[mode+'_cube_mean_accuracy_pct']=z['mean_accuracy_pct']
        row.update({k:r[k] for k in ('two_fraction','hint_fraction','hint_reliability_when_present')})
        rr.append(row)
    b=sorted(b,key=lambda x:(x['order'],x['layout'],x['step'],x['seed'],x['hint']))
    c=sorted(c,key=lambda x:(x['step'],x['seed'],x['hint']));rr=sorted(rr,key=lambda x:(x['seed'],ARMS.index(x['arm'])))
    csv_write(out/'balanced_all.csv',b);csv_write(out/'cubes_all.csv',c);csv_write(out/'repairs_all.csv',rr)
    differences=[]
    for seed in range(10,18):
        rows={r['arm']:r for r in rr if r['seed']==seed}
        for treatment,control in [('hint_frequency_control','iid'),('augmentation','iid'),
                                  ('augmentation','hint_frequency_control'),('consistency','augmentation')]:
            rec={'seed':seed,'treatment':treatment,'control':control}
            for metric in ('incorrect_cube_robust_accuracy_pct','two_absent_accuracy_pct','two_incorrect_accuracy_pct',
                           'relevant_changed_accuracy_pct','absent_cube_robust_accuracy_pct','correct_cube_robust_accuracy_pct'):
                rec[metric+'_difference_pp']=rows[treatment][metric]-rows[control][metric]
            differences.append(rec)
    csv_write(out/'repair_paired_differences.csv',differences)
    effectgroups=group_summary(b,('order','layout','step','hint'),['effect_pp','margin_change'])
    repairmetrics=['two_absent_accuracy_pct','two_incorrect_accuracy_pct','relevant_changed_accuracy_pct',
                  'incorrect_cube_robust_accuracy_pct','absent_cube_robust_accuracy_pct','correct_cube_robust_accuracy_pct']
    repairgroups=group_summary(rr,('arm',),repairmetrics)
    pairedgroups=group_summary(differences,('treatment','control'),[k for k in differences[0] if k.endswith('_difference_pp')])
    selected=[r for r in b if r['order']=='company_first' and r['layout']=='suggestion_last' and r['step']==2000 and r['hint']=='incorrect']
    cubes=[r for r in c if r['step']==2000 and r['hint']=='incorrect']
    summary={'status':'executed_independent_specification_replication_and_same_checkpoint_extensions',
        'primary_models':32,'access_models':32,'repair_runs':32,'baseline_models_from_scratch':64,
        'baseline_updates_per_model':4000,'baseline_examples_per_model':512000,
        'author_bitwise_replay':False,'full_paper_reproduction':False,
        'baseline':baseline['key_measurements'],'balanced_headline':stats([r['effect_pp'] for r in selected]),
        'balanced_headline_positive_models':sum(r['effect_pp']>0 for r in selected),
        'balanced_groups':effectgroups,'repair_groups':repairgroups,'repair_paired_groups':pairedgroups,
        'cube_initial_models':cubes,
        'initial_additive_worse_than_constant_models':sum(r['additive_heldout_rmse']>r['constant_heldout_rmse'] for r in cubes),
        'audit':{k:v for k,v in audit.items() if k!='model_checks'},
        'completion_records':{s:read(execution/s/'completed.json') for s in ('primary','access','mechanism','extensions','repair')},
        'remaining_paper_groups':baseline['remaining_unexecuted'],
        'interpretation_limits':['New CPU initialization/training streams are not the authors GPU realizations.',
            'Balanced compensation changes an unused company-country fact and only preserves person-country answers.',
            'The four-witness cohort is specially conditioned; compare counts within its worlds, not against one-swap cohorts.',
            'Cube nonadditivity does not identify a unique internal mechanism.',
            'Hint-frequency control matches task/hint marginals, not every table statistic.',
            'Robustness is over 16 constructed contexts, not all equivalent prompts.',
            'No pretrained-language-model, false-versus-true claim, mathematical, or formal-proof transfer is established.']}
    dump_json(out/'summary.json',summary)
    write_report(out,baseline,summary,selected,cubes,rr)
    print(json.dumps({'primary_models':32,'access_models':32,'repair_runs':32,'balanced_headline':summary['balanced_headline'],
                     'initial_additive_worse_than_constant_models':summary['initial_additive_worse_than_constant_models']},indent=2),flush=True)


def write_report(out,base,result,balanced,cubes,repairs):
    f=base['key_measurements'];lines=[
        '# False Witnesses: baseline reproduction and same-setting extensions','',
        '## What was actually done','',
        'This execution trained 32 primary and 32 access-intervention models from the manuscript specification. '
        'Every baseline model received 4,000 updates with batch size 128. The primary grid covers both table orders '
        'and both main suffix layouts. Within each seed, conditions share identical initial tensors and semantic training examples. '
        'Natural outputs were retained at all nine checkpoints from zero to 4,000 updates, and model/optimizer states at 2,000 and 4,000.','',
        'The model uses two cloned pre-normalized transformer layers, width 64, four heads, feed-forward width 128, '
        'exact GELU, no dropout, and 69,576 parameters. The training mixture is 25% one-step and 75% two-step, '
        'with suggestions present in half of examples and correct 75% of the time. AdamW uses the paper\'s '
        'learning rate 0.003, weight decay 0.01 on every parameter, batch size 128 and clipping threshold 1.','',
        'After comparing the baseline to the paper, local Q/K/V interventions and the new token-balanced tests '
        'use its unchanged checkpoints. Four retraining arms start from copies of each company-first, suggestion-last '
        '2,000-update model and optimizer. These are new measurements, not renamed earlier pilot results.','',
        '**Scope:** this is independent reproduction of specified core experiments, not bitwise author-run replay '
        'or completion of the entire manuscript. Training ran on CPU with PyTorch 2.10.0 rather than the paper\'s '
        'PyTorch 2.6.0 / RTX A5000. The source implementation\'s random-call order and evaluation arrays were unavailable; '
        'all newly used streams and inputs are saved. The fixed-position reliability grid, extended 16,000-update '
        'runs, independent-message training control and several appendix sensitivity groups remain unexecuted.','',
        '## 1. Did the main false-witness result reproduce?','',
        'The headline comparison changes another employee\'s company between two companies in the same country. '
        'Every person\'s country answer remains fixed. The metric is the percentage of answers naming the country '
        'implied by the wrong suggestion, not clean-task accuracy.','',
        '| Measurement | Paper | Independent run |','|---|---:|---:|',
        f"| Control suggestion following | 27.4353% | {f['headline_control_pct']['mean']:.4f}% |",
        f"| Witness suggestion following | 54.8004% | {f['headline_witness_pct']['mean']:.4f}% |",
        f"| Increase | 27.3651 points | {f['headline_effect_pp']['mean']:.4f} points |",'',
        'All eight models contribute equally. The two groups use the same experimental definitions but different '
        'random realizations; seed numbers do not pair an author model with our corresponding CPU model. '
        'The paper values are recovered from its Table 3, whereas the independent values are computed from saved logits.','',
        '### Full primary comparison','',
        '| Table order | Final token | Updates | Clean two-step accuracy (%) | Wrong-hint accuracy (%) | Witness effect (points) |',
        '|---|---|---:|---:|---:|---:|']
    for r in base['primary_groups']:
        x=r['statistics'];lines.append(f"| {r['order']} | {r['layout']} | {r['step']} | {x['two_absent_accuracy_pct']['mean']:.4f} | {x['two_incorrect_accuracy_pct']['mean']:.4f} | {x['effect_pp']['mean']:.4f} |")
    lines +=['','Clean and wrong-hint accuracy measure different capabilities. A model can execute the lookup correctly '
        'without a suggestion yet select a different employee when a misleading suggestion is present. '
        'Small or absent witness effects can also arise when the model is already following the suggestion in both paired inputs.','',
        '## 2. Does changing information access produce the reported direction?','',
        'Only the 64 first-layer connections between employee queries and company keys are changed. '
        'The second layer, parameter count, initialization, training stream and encoding stay fixed within each pair.','',
        '| Order | Access | Updates | Wrong-hint accuracy (%) | Witness effect (points) |','|---|---|---:|---:|---:|']
    for r in base['access_groups']:
        x=r['statistics'];lines.append(f"| {r['order']} | {r['access']} | {r['step']} | {x['two_incorrect_accuracy_pct']['mean']:.4f} | {x['effect_pp']['mean']:.4f} |")
    lines +=['',f"At 4,000 updates, opening access improves wrong-hint accuracy in {f['opening_accuracy_improved_pairs_at_4000']} of 8 pairs; closing it lowers accuracy in {f['closing_accuracy_worsened_pairs_at_4000']} of 8 pairs.",
        'The paper reports 8 of 8 in each direction. The complete per-seed paper/local comparisons are in paper_access_comparison.csv.','',
        '## 3. Mechanistic checks','',
        'The second-layer interventions separately change employee or company keys, values, or the final query. '
        'A crossed test takes the edited employee\'s key from the matching-company input, but its value from a '
        'different input where that employee has a third country. Unpatched activations and the recipient residual '
        'stay fixed. A query-only intervention takes the final query from a no-suggestion run asking either the '
        'same person or a different eligible person.','',
        '| Company-first, suggestion-last | Updates | K+V third-country answers (%) | Same V at queried employee (%) | K/V interaction (points) |',
        '|---|---:|---:|---:|---:|']
    for r in base['mechanism_groups']:
        if r['order']=='company_first' and r['layout']=='suggestion_last':
            x=r['statistics'];lines.append(f"| Independent run | {r['step']} | {x['K_and_V_target_pct']['mean']:.4f} | {x['V_at_query_target_pct']['mean']:.4f} | {x['crossed_interaction_pp']['mean']:.4f} |")
    lines+=['','The interaction is the combined change minus the two separate changes, using a common unmodified '
        'baseline. It tests whether selection and the transferred information act together; it is not a proof that '
        'all computation occurs at a single head or record. Native float64 output and self-patch checks are retained '
        'for every checkpoint. paper_crossed_comparison.csv preserves the Table 6 comparison.','',
        '## 4. Added test: exactly balance every symbol count','',
        'The ordinary equal-function edit changes an employee\'s company code from r to h. The added test '
        'compensates by changing a country-table entry at an unused company from h to r. The total number '
        'of every input symbol is identical, as are input length, the query, its employer and every person\'s '
        'country answer. This uses the shared numeric alphabet; it does not directly extend to disjoint company '
        'and country vocabularies. The unused company\'s own country fact changes.','',
        'Employee-only and compensation-only conditions are evaluated alongside the combined edit. Therefore '
        'the experiment does not simply assume that the compensating edit is inert. It is a stronger repetition '
        'control, not the first repetition control in the original paper.','',
        '| Seed | Control follows wrong suggestion (%) | Balanced witness (%) | Difference (points) |','|---|---:|---:|---:|']
    for r in balanced:lines.append(f"| {r['seed']} | {r['control_following_pct']:.4f} | {r['witness_following_pct']:.4f} | {r['effect_pp']:.4f} |")
    z=result['balanced_headline'];lines+=['',f"Mean paired effect: **{z['mean']:.4f} percentage points**; model-level standard deviation {z['sample_sd']:.4f} points. "
        'These are the original-format company-first, suggestion-last models at 2,000 updates. The table retains null and negative results.','',
        '## 5. Added test: all combinations of four witnesses','',
        'Each of 1,024 underlying worlds is evaluated in 16 contexts, switching four balanced edits on or off. '
        'Every context in a world has the same symbols and correct person-country answers. An additive predictor '
        'uses the zero-witness context and each single-witness context, then predicts the remaining eleven combinations. '
        'The higher-order share measures how much of the variation in the wrong-minus-correct logit margin is '
        'associated with interactions; it is not an error rate.','',
        '| Seed | Higher-order share (%) | Additive RMSE | Constant RMSE | Correct on all 16 (%) |','|---|---:|---:|---:|---:|']
    for r in cubes:lines.append(f"| {r['seed']} | {r['higher_order_share_pct'] if r['higher_order_share_pct'] is None else format(r['higher_order_share_pct'], '.4f')} | {r['additive_heldout_rmse']:.4f} | {r['constant_heldout_rmse']:.4f} | {r['robust_accuracy_pct']:.4f} |")
    lines+=['',f"The additive predictor is worse than the zero-witness constant baseline in {result['initial_additive_worse_than_constant_models']} of 8 models. "
        'This is a within-world combination test, not prediction of new worlds without calibration. '
        'The special four-witness cohort has additional table constraints; its zero-witness rate must not be '
        'treated as the same population as the separate one-swap test.','',
        '## 6. Added test: four matched training continuations','',
        'All four arms start from the same model and optimizer at 2,000 updates and receive 500 more updates '
        'with 512 supervised contexts each. Ordinary continuation retains the paper distribution. Augmentation '
        'uses answer-preserving paired contexts plus relevant edits that change the correct answer. Consistency '
        'uses exactly those same data and additionally penalizes disagreement on the answer-preserving pair. '
        'The hint-frequency control matches the changed task/hint exposure and decisive wrong hints without paired edits. '
        'It does not match every conditional table statistic.','',
        '| Seed | Arm | Clean accuracy (%) | Wrong-hint cube robust (%) | No-hint cube robust (%) | Relevant-change accuracy (%) |',
        '|---|---|---:|---:|---:|---:|']
    for r in repairs:lines.append(f"| {r['seed']} | {r['arm']} | {r['two_absent_accuracy_pct']:.4f} | {r['incorrect_cube_robust_accuracy_pct']:.4f} | {r['absent_cube_robust_accuracy_pct']:.4f} | {r['relevant_changed_accuracy_pct']:.4f} |")
    lines+=['','Cube robust accuracy is the fraction of worlds answered correctly in all 16 versions. '
        'Training uses one-witness edits; testing includes four-witness combinations. All outcomes use the ordinary '
        'transformer at inference, without an external symbolic solver. A successful relevant-change check excludes '
        'a constant-output defense, but cannot distinguish provenance learning from discounting hints. '
        'The frequency-control comparison and no-hint/correct-hint tradeoffs must be included in any claim of improvement.','',
        '## Validation and reproducibility','',
        f"Saved output statistics were independently recomputed for {result['audit']['saved_arrays_recomputed']:,} arrays, "
        f"covering {result['audit']['saved_classes_recomputed']:,} class predictions. Reloaded checkpoints reproduced "
        f"{result['audit']['checkpoint_replayed_classes']:,} predictions with maximum logit error "
        f"{result['audit']['maximum_replay_logit_error']:.3g}. Seed 10\'s complete 4,000-update training stream was replayed "
        'from the saved initialization, comparing model and optimizer tensors at 2,000 and 4,000 updates. '
        'These checks establish reproducibility of this execution, not agreement with missing author tensors.','',
        'All model-level tables, checkpoint inputs, logits, training streams, source hashes and tests are retained. '
        'The earlier three-seed pilot is separate and was not used as data for these results.','',
        '## Source map','',
        '1. false_witnesses_v2.pdf: Appendix C (architecture, training and matched groups), D.2 (equal-function population), '
        'E.3-E.5 (activation interventions), H (numerical conventions), J Tables 3, 4 and 6 (reported comparison values). '
        'The original PDF was available through text retrieval, not local byte-level or visual verification.',
        '2. Unchanged reconstruction core: repository commit 0e0fc39c8d6f0f305208958509313e21f1d9ebf1, '
        'reproduction/core.py, Git blob 72ba488367e69b0552cf067feef1bd88cd034231.',
        '3. New execution: primary/, access/, mechanism/, assessment/, extensions/, repair/, audit/ and this summary. '
        'The script does not infer execution from the presence of a plan: completed-stage records are required.','',
        '**Not established:** entire-paper or bitwise-author reproduction; transfer to pretrained language models, '
        'genuinely false claims against authoritative facts, mathematics, or formal proof search.']
    # Join adjacent literals above without altering numeric values or scope flags.
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.execution,a.output)
