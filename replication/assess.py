"""Compare independently trained groups with the paper before new experiments.

Group means use models as the statistical units. The equal seed integers in the
paper and this CPU replication do NOT make their trajectories paired. Paired
comparisons below are only between conditions sharing our saved initialization
and semantic training streams. No threshold is tuned to force a reproduction.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from reproduction.core import dump_json,sha256


def stats(values)->dict:
    a=np.asarray(values,dtype=float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Need at least two finite model measurements')
    rng=np.random.default_rng(819)
    boot=a[rng.integers(0,len(a),size=(20000,len(a)))].mean(1)
    return {'models':len(a),'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),
        'min':float(a.min()),'max':float(a.max()),'model_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
        'scope':'descriptive model-resampling interval; small seed count; no multiple-testing correction'}


def csv_write(path:Path,rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('No rows to write')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_baseline(root:Path)->list[dict]:
    if not (root/'completed.json').is_file():raise ValueError(f'Incomplete baseline {root}')
    files=sorted(root.glob('models/*/report.json'))
    if len(files)!=32:raise ValueError('Missing baseline model report')
    rows=[]
    for p in files:
        records=json.loads(p.read_text())
        if [r['step'] for r in records]!=list(range(0,4001,500)):raise ValueError('Missing natural checkpoint')
        for r in records:
            row={k:r[k] for k in ('seed','step','order','layout','access')}
            for key,v in r['natural'].items():row[key+'_accuracy_pct']=v['accuracy_pct']
            if 'equal_function' in r:
                row.update({k:r['equal_function'][k] for k in ('control_following_pct','witness_following_pct','effect_pp')})
            else:row.update(dict.fromkeys(('control_following_pct','witness_following_pct','effect_pp'),None))
            rows.append(row)
    return rows


def group_summary(rows:list[dict],keys:tuple[str,...],metrics:list[str])->list[dict]:
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for vals,rs in sorted(groups.items()):
        item=dict(zip(keys,vals));item['statistics']={m:stats([r[m] for r in rs]) for m in metrics};out.append(item)
    return out


def compare_paper(rows,config:Path,kind:str):
    with config.open(newline='') as f:paper=list(csv.DictReader(f))
    out=[]
    for p in paper:
        matches=[r for r in rows if r['seed']==int(p['seed']) and r['step']==int(p['updates']) and
                 r['order']==p['order'] and r['layout']=='suggestion_last' and r['access']==p.get('access','usual')]
        if len(matches)!=1:raise ValueError('Paper row did not identify one replication measurement')
        r=matches[0];rec={'seed':r['seed'],'step':r['step'],'order':r['order'],'access':r['access']}
        for key in ('wrong_hint_accuracy_pct','control_following_pct','witness_following_pct','effect_pp'):
            if key not in p:continue
            ours=r['two_incorrect_accuracy_pct'] if key=='wrong_hint_accuracy_pct' else r[key]
            rec['paper_'+key]=float(p[key]);rec['replication_'+key]=ours;rec['difference_'+key]=ours-float(p[key])
        out.append(rec)
    return out


def run(execution:Path,out:Path):
    if out.exists():raise FileExistsError(out)
    primary=read_baseline(execution/'primary');access=read_baseline(execution/'access')
    mp=execution/'mechanism/summary.json'
    if not mp.is_file():raise ValueError('Mechanistic tests must finish before baseline assessment')
    mech=json.loads(mp.read_text())
    if len(mech)!=64:raise ValueError('Incomplete primary mechanistic grid')
    out.mkdir(parents=True)
    csv_write(out/'primary_all.csv',primary);csv_write(out/'access_all.csv',access)
    config=Path(__file__).resolve().parents[1]/'configs'
    pc=compare_paper(primary,config/'paper_table3.csv','primary');ac=compare_paper(access,config/'paper_table4.csv','access')
    csv_write(out/'paper_primary_comparison.csv',pc);csv_write(out/'paper_access_comparison.csv',ac)
    endpoint=[r for r in primary if r['step'] in (2000,4000)]
    psummary=group_summary(endpoint,('order','layout','step'),['two_absent_accuracy_pct','two_incorrect_accuracy_pct','control_following_pct','witness_following_pct','effect_pp'])
    asummary=group_summary([r for r in access if r['step'] in (2000,4000)],('order','access','step'),['two_absent_accuracy_pct','two_incorrect_accuracy_pct','effect_pp'])
    pairs=[]
    for seed in range(30,38):
        for step in (2000,4000):
            for order,treatment in [('employee_first','open'),('company_first','closed')]:
                rs={r['access']:r for r in access if r['seed']==seed and r['step']==step and r['order']==order}
                pairs.append({'seed':seed,'step':step,'order':order,'treatment':treatment,
                    'accuracy_difference_pp':rs[treatment]['two_incorrect_accuracy_pct']-rs['usual']['two_incorrect_accuracy_pct'],
                    'effect_difference_pp':rs[treatment]['effect_pp']-rs['usual']['effect_pp']})
    csv_write(out/'access_paired_differences.csv',pairs)
    mrows=[]
    for r in mech:
        row={k:r[k] for k in ('seed','step','order','layout')}
        row.update({k+'_margin_change':v['margin_change'] for k,v in r['channels'].items()})
        row.update({k+'_target_pct':v['target_pct'] for k,v in r['crossed'].items()})
        row['crossed_interaction_pp']=r['crossed_interaction_pp']
        row.update({'query_'+k:v for k,v in r['query']['eligible'].items()})
        mrows.append(row)
    csv_write(out/'mechanism_all.csv',mrows)
    with (config/'paper_table6.csv').open(newline='') as f: crossed_paper=list(csv.DictReader(f))
    crossed_comparison=[]
    for p in crossed_paper:
        rr=[r for r in mech if r['seed']==int(p['seed']) and r['step']==int(p['updates']) and
            r['order']=='company_first' and r['layout']=='suggestion_last']
        if len(rr)!=1:raise ValueError('Missing Table 6 counterpart')
        r=rr[0];row={'seed':r['seed'],'step':r['step']}
        for key in ('neither','K','V','K_and_V','V_at_query'):
            row['paper_'+key+'_pct']=float(p[key]);row['replication_'+key+'_pct']=r['crossed'][key]['target_pct']
        row['paper_interaction_pp']=float(p['interaction_pp']);row['replication_interaction_pp']=r['crossed_interaction_pp']
        crossed_comparison.append(row)
    csv_write(out/'paper_crossed_comparison.csv',crossed_comparison)
    ms=group_summary(mrows,('order','layout','step'),[k for k in mrows[0] if k not in ('seed','step','order','layout','query_n')])
    target=[r for r in endpoint if r['order']=='company_first' and r['layout']=='suggestion_last' and r['step']==2000]
    finding={'headline_control_pct':stats([r['control_following_pct'] for r in target]),
        'headline_witness_pct':stats([r['witness_following_pct'] for r in target]),
        'headline_effect_pp':stats([r['effect_pp'] for r in target]),
        'headline_positive_effect_models':sum(r['effect_pp']>0 for r in target),
        'person_last_minimum_two_step_accuracy_pct':min(r['two_incorrect_accuracy_pct'] for r in endpoint if r['layout']=='person_last'),
        'opening_accuracy_improved_pairs_at_4000':sum(r['accuracy_difference_pp']>0 for r in pairs if r['step']==4000 and r['treatment']=='open'),
        'closing_accuracy_worsened_pairs_at_4000':sum(r['accuracy_difference_pp']<0 for r in pairs if r['step']==4000 and r['treatment']=='closed')}
    result={'status':'baseline_executed_and_compared_not_author_bitwise_replay',
        'primary_models':32,'access_models':32,'natural_output_arrays':32*2*9*6,
        'primary_intervention_checkpoints':64,'mechanistic_checkpoint_evaluations':64,
        'source':'measured new training and inference; paper columns explicitly separated',
        'paper_primary_csv_sha256':sha256(config/'paper_table3.csv'),
        'paper_access_csv_sha256':sha256(config/'paper_table4.csv'),
        'paper_crossed_csv_sha256':sha256(config/'paper_table6.csv'),
        'primary_groups':psummary,'access_groups':asummary,'mechanism_groups':ms,
        'access_paired_groups':group_summary(pairs,('order','treatment','step'),['accuracy_difference_pp','effect_difference_pp']),
        'key_measurements':finding,
        'full_paper_reproduced':False,'author_exact_predictions_replayed':False,
        'extensions_basis':'same frozen checkpoint tensors; interpret relative to this measured specification-level baseline',
        'remaining_unexecuted':['fixed-position reliability grid and 16000-update runs','independent-message training control',
            'nine-model initial discovery','separate suggestion vocabulary','domain-size/learning-rate sweeps'],
        'limitations':['CPU/PyTorch version differs from original GPU environment; seeds identify new random trajectories.',
            'New fixed evaluation inputs satisfy published constraints but are not archived original arrays.',
            'Small model groups; bootstrap intervals are descriptive, not equivalence tests.',
            'Primary/access/E.3-E.5 completion does not certify the complete paper.']}
    dump_json(out/'baseline_assessment.json',result)
    lines=['# Baseline assessment before same-setting extensions','','Independent replication from the paper specification. No original output arrays are substituted.','',
        '| Order | Last token | Updates | Clean accuracy (%) | Wrong-hint accuracy (%) | Equal-function effect (points) |',
        '|---|---|---:|---:|---:|---:|']
    for r in psummary:
        x=r['statistics'];lines.append(f"| {r['order']} | {r['layout']} | {r['step']} | {x['two_absent_accuracy_pct']['mean']:.4f} | {x['two_incorrect_accuracy_pct']['mean']:.4f} | {x['effect_pp']['mean']:.4f} |")
    lines+=['','These means retain all eight seeds. See the CSVs for every model and the side-by-side published measurements.','',
        'The additional experiments reuse these weights and do not establish results for the authors\' unrecovered exact weights. Remaining paper groups are listed in baseline_assessment.json.']
    (out/'BASELINE_ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(finding,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.execution,a.output)
