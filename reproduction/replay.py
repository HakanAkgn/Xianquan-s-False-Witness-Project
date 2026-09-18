"""Strict author-artifact replay for the initial Figure 1/Table 3 comparison.

A pass verifies only this stage, NOT the rest of the paper or training replay.
The current repository has no author artifacts, so the supplied manifest blocks.
Hashes verify integrity, not authorship; origin must be checked against the author
release independently. No regenerated-corpus or synthetic-reference fallback.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .core import Setting, Model, canonical_numerics, decode, encode, validate_equal_pair, logits_for, sha256, dump_json
from .diagnostic import reference_rows

SCOPE='figure1_table3_company_first_2000'
SEEDS=tuple(range(10,18))
MODES=tuple(f'{t}_{h}' for t in ('one','two') for h in ('absent','correct','incorrect'))
CASES=MODES+('pair_control','pair_witness')


class ReplayBlocked(ValueError):
    pass


def checked_file(root: Path, item: dict, name: str) -> Path:
    if not isinstance(item,dict) or not item.get('path') or not item.get('sha256'):
        raise ReplayBlocked(f'Missing author artifact: {name}')
    path=(root/item['path']).resolve()
    if not path.is_file():raise ReplayBlocked(f'Unavailable author artifact: {name}')
    digest=item['sha256']
    if not isinstance(digest,str) or len(digest)!=64 or sha256(path)!=digest:
        raise ReplayBlocked(f'Hash mismatch: {name}')
    return path


def validate_inputs(data: dict, s: Setting, n: int=4096):
    worlds={}
    for key in CASES:
        if key not in data:raise ReplayBlocked(f'Missing input condition {key}')
        a=data[key]
        if a.shape!=(n,s.length) or a.dtype.kind not in 'iu':
            raise ReplayBlocked(f'Invalid shape or noninteger tokens in {key}')
        worlds[key]=decode(a,s)
    natural0=worlds['one_absent']
    for key in MODES:
        w=worlds[key];task,hint=key.split('_',1)
        for field in ('f','g','x'):
            if not torch.equal(getattr(w,field),getattr(natural0,field)):
                raise ReplayBlocked('Natural conditions do not share underlying tables/query')
        if not (w.two==(task=='two')).all() or not (w.present==(hint!='absent')).all():
            raise ReplayBlocked(f'Incorrect task/hint mode in {key}')
        if hint!='absent':
            mid=w.f[torch.arange(n),w.x]
            if not ((w.hint==mid)==(hint=='correct')).all():
                raise ReplayBlocked(f'Incorrect suggestion validity in {key}')
    if not torch.equal(worlds['one_incorrect'].hint,worlds['two_incorrect'].hint):
        raise ReplayBlocked('One/two-step wrong-hint conditions do not share suggestions')
    validate_equal_pair(data['pair_control'],data['pair_witness'],s,require_d2=True)
    return worlds


def compare_logits(actual: np.ndarray, expected: np.ndarray, tolerance: float=5e-5):
    if actual.shape!=expected.shape or actual.ndim!=2:
        raise ReplayBlocked('Reference/model output shape mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ReplayBlocked('Nonfinite output array')
    error=float(np.max(np.abs(actual-expected)))
    mismatches=int(np.count_nonzero(actual.argmax(1)!=expected.argmax(1)))
    return {'maximum_logit_error':error,'class_mismatches':mismatches,
            'passed':bool(error<=tolerance and mismatches==0)}


def load_author_model(item: dict, root: Path, s: Setting) -> Model:
    path=checked_file(root,item.get('checkpoint'),'checkpoint')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict):raise ReplayBlocked('Checkpoint is not a tensor mapping/container')
    status=str(payload.get('metadata',{}).get('status','')) if isinstance(payload.get('metadata',{}),dict) else ''
    if 'diagnostic' in status.lower() or 'pilot' in status.lower():
        raise ReplayBlocked('A reconstruction checkpoint is not an author checkpoint')
    field=item.get('state_dict_key')
    state=payload if field is None else payload.get(field)
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ReplayBlocked('Explicit state_dict_key does not identify a tensor state dictionary')
    model=Model(s)
    keymap=item.get('key_map')
    if keymap is not None:
        if not isinstance(keymap,dict) or set(keymap)!=set(model.state_dict()):
            raise ReplayBlocked('Explicit target-to-author key map must cover all model parameters')
        if len(set(keymap.values()))!=len(keymap):raise ReplayBlocked('Noninjective key mapping')
        try:state={target:state[source] for target,source in keymap.items()}
        except KeyError as e:raise ReplayBlocked(f'Unknown mapped source parameter {e}') from e
    if set(state)!=set(model.state_dict()) or not all(torch.isfinite(v).all() for v in state.values()):
        raise ReplayBlocked('Parameter coverage/nonfiniteness check failed')
    model.load_state_dict(state,strict=True)
    return model.double().eval()


def replay(manifest: str | Path):
    manifest=Path(manifest);root=manifest.parent
    m=json.loads(manifest.read_text());s=Setting()
    if m.get('scope')!=SCOPE:raise ReplayBlocked('Unsupported scope; this verifier covers the initial comparison only')
    origin=m.get('origin',{})
    if origin.get('kind')!='author_release' or not origin.get('locator') or not origin.get('verification_note'):
        raise ReplayBlocked('Author source provenance has not been established')
    checked_file(root,origin.get('archive'),'author release archive')
    inputs_path=checked_file(root,m.get('evaluation_inputs'),'fixed evaluation inputs')
    models=m.get('models',[])
    if len(models)!=8 or sorted(item.get('seed',-1) for item in models)!=list(SEEDS):
        raise ReplayBlocked('Require every author seed 10--17 exactly once')
    if any(item.get('updates')!=2000 for item in models):raise ReplayBlocked('Require the 2,000-update checkpoints')
    env=canonical_numerics(1)
    with np.load(inputs_path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    worlds=validate_inputs(data,s)
    ref=reference_rows(Path(__file__).parent.parent/'configs/paper_table3.csv')
    results=[]
    for item in sorted(models,key=lambda x:x['seed']):
        model=load_author_model(item,root,s)
        expected_path=checked_file(root,item.get('reference_outputs'),f"reference outputs seed {item['seed']}")
        with np.load(expected_path,allow_pickle=False) as f:expected={k:f[k] for k in f.files}
        comparisons={};actual={}
        for key in CASES:
            if key+'_logits' not in expected:raise ReplayBlocked(f'Missing exact-GELU reference logits: {key}')
            actual[key]=logits_for(model,data[key])
            if expected[key+'_logits'].shape!=(4096,8):raise ReplayBlocked('Reference logit population is not 4096 by 8')
            comparisons[key]=compare_logits(actual[key],expected[key+'_logits'])
        w=worlds['pair_control'];wrong=w.g[torch.arange(4096),w.hint].numpy()
        control=float(100*np.mean(actual['pair_control'].argmax(1)==wrong))
        witness=float(100*np.mean(actual['pair_witness'].argmax(1)==wrong))
        numbers={'wrong_hint_accuracy_pct':float(100*np.mean(actual['two_incorrect'].argmax(1)==worlds['two_incorrect'].labels().numpy())),
                 'control_following_pct':control,'witness_following_pct':witness,'effect_pp':witness-control}
        paper=next(v for v in ref if v['seed']==item['seed'] and v['updates']==2000 and v['order']=='company_first')
        table={key:abs(value-paper[key])<=.000501 for key,value in numbers.items()}
        results.append({'seed':item['seed'],'conditions':comparisons,'measurements':numbers,'table3_checks':table,
                        'passed':all(c['passed'] for c in comparisons.values()) and all(table.values())})
    passed=all(item['passed'] for item in results)
    return {'scope':SCOPE,'manifest_sha256':sha256(manifest),'environment':env,'models':results,
            'initial_comparison_replayed':passed,'full_paper_reproduced':False,'extensions_allowed':False,
            'status':'initial_comparison_verified_other_stages_pending' if passed else 'author_replay_failed',
            'limitations':['This does not replay training from scratch.',
                'Figures 2--7, other primary layouts/orders, access and reliability groups remain separate stages.',
                'Hashes establish consistency of supplied files, not authorship; origin review remains necessary.']}


def audit(manifest: str | Path):
    try:return replay(manifest)
    except (ValueError,OSError,KeyError,RuntimeError,TypeError) as e:
        return {'status':'blocked','initial_comparison_replayed':False,'full_paper_reproduced':False,
                'extensions_allowed':False,'reason':str(e),'error_type':type(e).__name__}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.manifest);dump_json(a.output,r)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r.get('initial_comparison_replayed') else 2)
