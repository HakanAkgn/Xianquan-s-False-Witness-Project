"""Train the complete primary/access grids from the manuscript specification.

Uses the unchanged reproduction.core module (Git blob 72ba488367e69b0552cf067feef1bd88cd034231).
CPU execution with new, explicitly saved streams is scientific replication, not
bitwise replay of the author's GPU runs. Never overwrite completed executions.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from reproduction.core import (Setting, Model, World, canonical_numerics, draw,
    encode, decode, equal_pairs, logits_for, sha256, state_hash, dump_json)

BASE_COMMIT = '0e0fc39c8d6f0f305208958509313e21f1d9ebf1'
NATURAL_SEED = 2026091901
PAIR_SEED = 2026091902


def pack_world(w: World) -> np.ndarray:
    return torch.cat((w.f, w.g, w.x[:,None], w.two[:,None],
                      w.hint[:,None], w.present[:,None]), 1).numpy().astype(np.uint8)


def unpack_world(a: np.ndarray, n: int=8) -> World:
    a = torch.from_numpy(np.array(a, dtype=np.int64, copy=True))
    if a.ndim != 2 or a.shape[1] != 2*n+4:
        raise ValueError('Invalid packed world shape')
    return World(a[:,:n], a[:,n:2*n], a[:,2*n], a[:,2*n+1].bool(),
                 a[:,2*n+2], a[:,2*n+3].bool())


def group_jobs(group: str) -> list[tuple[int, Setting]]:
    if group == 'primary':
        return [(seed, Setting(order=order, layout=layout)) for seed in range(10,18)
                for order in ('company_first','employee_first')
                for layout in ('suggestion_last','person_last')]
    if group == 'access':
        return [(seed, Setting(order=order, access=access)) for seed in range(30,38)
                for order,access in [('employee_first','usual'),('employee_first','open'),
                                     ('company_first','usual'),('company_first','closed')]]
    raise ValueError('Unknown training group')


def job_name(seed: int, s: Setting) -> str:
    return f'{s.order}__{s.layout}__{s.access}__seed_{seed}'


def optimizer(model: Model) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01,
                             betas=(.9,.999), eps=1e-8)


def prepare(root: Path, group: str, steps: int=4000) -> dict:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f'{root}: choose a new execution directory')
    root.mkdir(parents=True)
    env = canonical_numerics(1)
    jobs = group_jobs(group)
    if steps != 4000:
        raise ValueError('The primary and access protocol both require 4,000 updates')
    s = Setting(); shared = root/'shared'; shared.mkdir()
    rng = torch.Generator().manual_seed(NATURAL_SEED)
    w = draw(s,4096,rng)
    i=torch.arange(4096)
    middle = w.f[i,w.x]
    wrong_hint = (middle + torch.randint(1,s.n,(4096,),generator=rng)) % s.n
    w.hint = wrong_hint
    base, control, witness, rows = equal_pairs(s,4096,PAIR_SEED)
    data = {'natural':pack_world(w), 'base':pack_world(base), 'control':pack_world(control),
            'witness':pack_world(witness), 'edited_rows':rows}
    np.savez_compressed(shared/'evaluation.npz', **data)
    stream_meta = {}
    for seed in sorted(set(seed for seed,_ in jobs)):
        torch.manual_seed(seed)
        model=Model(s)
        initial=model.state_dict()
        # A single canonical initialization is loaded into every paired condition.
        torch.save({'state_dict':initial, 'seed':seed}, shared/f'initial_{seed}.pt')
        rng=torch.Generator().manual_seed(35600+seed)
        a=np.empty((steps,128,2*s.n+4),dtype=np.uint8)
        for step in range(steps): a[step]=pack_world(draw(s,128,rng))
        np.save(shared/f'train_{seed}.npy',a,allow_pickle=False)
        torch.save({'rng_state':rng.get_state()},shared/f'rng_end_{seed}.pt')
        present=a[:,:,-1].astype(bool)
        middle_np=np.take_along_axis(a[:,:,:s.n],a[:,:,2*s.n,None].astype(np.int64),axis=2).squeeze(2)
        stream_meta[str(seed)]={'initial_state_hash':state_hash(initial),
            'initial_file_sha256':sha256(shared/f'initial_{seed}.pt'),
            'stream_sha256':sha256(shared/f'train_{seed}.npy'),
            'examples':int(steps*128), 'two_step_fraction':float(a[:,:,2*s.n+1].mean()),
            'hint_fraction':float(present.mean()),
            'hint_reliability_when_present':float((middle_np[present]==a[:,:,2*s.n+2][present]).mean())}
    manifest={'base_commit':BASE_COMMIT,'group':group,'environment':env,'steps':steps,
        'batch_size':128,'seeds':sorted(set(seed for seed,_ in jobs)),
        'settings':[{'seed':seed,**asdict(s)} for seed,s in jobs],
        'evaluation_seed_natural':NATURAL_SEED,'evaluation_seed_pairs':PAIR_SEED,
        'evaluation_inputs_sha256':sha256(shared/'evaluation.npz'), 'streams':stream_meta,
        'core_sha256':sha256(Path(__file__).resolve().parents[1]/'reproduction/core.py'),
        'run_script_sha256':sha256(__file__),
        'status':'independent_specification_replication_not_author_replay',
        'fresh_parameters_each_model':True,'paired_initialization_and_training_streams':True,
        'natural_checkpoints':list(range(0,4001,500)),'intervention_checkpoints':[2000,4000]}
    dump_json(root/'manifest.json',manifest)
    print(json.dumps({'prepared':group,'models':len(jobs),'root':str(root)}),flush=True)
    return manifest


def natural_inputs(data: dict, s: Setting) -> dict:
    w=unpack_world(data['natural'],s.n);i=torch.arange(len(w.x))
    cases={}
    for two in (False,True):
        for mode in ('absent','correct','incorrect'):
            c=w.copy();c.two.fill_(two);c.present.fill_(mode!='absent')
            if mode=='correct':c.hint=c.f[i,c.x]
            name=('two' if two else 'one')+'_'+mode
            cases[name]=(encode(c,s),c.labels().numpy())
    return cases


def paired_summary(a: np.ndarray, b: np.ndarray, wrong: np.ndarray, truth: np.ndarray) -> dict:
    pa,pb=a.argmax(1),b.argmax(1)
    ia,ib=pa==wrong,pb==wrong
    diff=ib.astype(float)-ia.astype(float)
    se=float(diff.std(ddof=1)/len(diff)**.5)
    return {'n':len(diff),'control_following_pct':float(100*ia.mean()),
        'witness_following_pct':float(100*ib.mean()),'effect_pp':float(100*diff.mean()),
        'paired_95ci_pp':[float(100*(diff.mean()-1.96*se)),float(100*(diff.mean()+1.96*se))],
        'became_wrong':int((~ia&ib).sum()),'became_not_wrong':int((ia&~ib).sum()),
        'both_wrong':int((ia&ib).sum()),'neither_wrong':int((~ia&~ib).sum()),
        'control_accuracy_pct':float(100*(pa==truth).mean()),
        'witness_accuracy_pct':float(100*(pb==truth).mean()),
        'margin_change':float(np.mean((b[np.arange(len(b)),wrong]-b[np.arange(len(b)),truth])-
                                      (a[np.arange(len(a)),wrong]-a[np.arange(len(a)),truth])))}


def train_one(root_string: str, seed: int, setting_dict: dict) -> dict:
    root=Path(root_string);s=Setting(**setting_dict);canonical_numerics(1)
    out=root/'models'/job_name(seed,s)
    out.mkdir(parents=True,exist_ok=False)
    m=Model(s)
    c=torch.load(root/'shared'/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
    m.load_state_dict(c['state_dict']); initial_hash=state_hash(m.state_dict())
    opt=optimizer(m)
    stream_path=root/'shared'/f'train_{seed}.npy'
    streams=np.load(stream_path,mmap_mode='r',allow_pickle=False)
    with np.load(root/'shared/evaluation.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    cases=natural_inputs(data,s)
    wa,wb=unpack_world(data['control'],s.n),unpack_world(data['witness'],s.n)
    xa,xb=encode(wa,s),encode(wb,s)
    truth=wa.labels().numpy();wrong=wa.g[torch.arange(len(wa.x)),wa.hint].numpy()
    reports=[];trace=[];start=time.monotonic()
    for step in range(len(streams)+1):
        if step%500==0:
            arrays={};r={'seed':seed,'step':step,**asdict(s),'natural':{}}
            for key,(tokens,label) in cases.items():
                l=logits_for(m,tokens)
                arrays[key]=l
                r['natural'][key]={'accuracy_pct':float(100*(l.argmax(1)==label).mean()),'n':len(label)}
            if step in (2000,4000):
                la,lb=logits_for(m,xa),logits_for(m,xb)
                arrays['pair_control']=la;arrays['pair_witness']=lb
                r['equal_function']=paired_summary(la,lb,wrong,truth)
                payload={'state_dict':m.state_dict(),'optimizer':opt.state_dict(),
                    'seed':seed,'step':step,'setting':asdict(s),'initial_state_hash':initial_hash,
                    'stream_sha256':sha256(stream_path),'stream_position':step,
                    'torch_rng_state':torch.get_rng_state(),
                    'status':'independent_specification_replication_not_author_replay'}
                torch.save(payload,out/f'model_{step}.pt')
                r['checkpoint_sha256']=sha256(out/f'model_{step}.pt')
            np.savez_compressed(out/f'predictions_{step}.npz',**arrays)
            r['elapsed_seconds']=time.monotonic()-start; reports.append(r)
            dump_json(out/'report.json',reports)
            print(json.dumps({'group':root.name,'model':out.name,'step':step,
                'no_hint':r['natural']['two_absent']['accuracy_pct'],
                'wrong_hint':r['natural']['two_incorrect']['accuracy_pct'],
                'effect':r.get('equal_function',{}).get('effect_pp')}),flush=True)
        if step==len(streams):break
        m.train();w=unpack_world(streams[step],s.n)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(encode(w,s)),w.labels())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True);opt.step()
        if (step+1)%100==0:
            trace.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(gn),
                          'elapsed_seconds':time.monotonic()-start})
    dump_json(out/'trace.json',trace)
    result={'seed':seed,**asdict(s),'initial_state_hash':initial_hash,'stream_sha256':sha256(stream_path),
            'final_state_hash':state_hash(m.state_dict()),'completed_updates':len(streams),
            'elapsed_seconds':time.monotonic()-start}
    dump_json(out/'completed.json',result)
    return result


def execute(root: Path, group: str, workers: int=4) -> None:
    if workers<1:raise ValueError('workers must be positive')
    root=Path(root);prepare(root,group)
    jobs=group_jobs(group);done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(train_one,str(root),seed,asdict(s)):(seed,s) for seed,s in jobs}
        for future in as_completed(futures):
            done.append(future.result())
            dump_json(root/'progress.json',{'finished':len(done),'expected':len(jobs),'models':done})
    # Invariants are checked using actual completed runs, not only configuration.
    for seed in sorted({seed for seed,_ in jobs}):
        entries=[r for r in done if r['seed']==seed]
        if len({r['initial_state_hash'] for r in entries})!=1 or len({r['stream_sha256'] for r in entries})!=1:
            raise RuntimeError('Paired initialization/stream mismatch')
    dump_json(root/'completed.json',{'models':len(done),'all_models_completed':True,
        'paired_initialization_and_streams_verified':True,'source':'new training runs',
        'author_bitwise_replay':False,'full_paper_reproduced':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--group',choices=['primary','access'],required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();execute(a.root,a.group,a.workers)
