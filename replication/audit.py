"""Independent checks of saved outputs, matched runs, and training replay.

This verifies our execution artifacts. It does not compare to missing author
checkpoint tensors, and it never substitutes reported paper rates for outputs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from reproduction.core import (Setting,Model,canonical_numerics,encode,logits_for,
                               state_hash,sha256,dump_json)
from .run import natural_inputs,unpack_world,optimizer,paired_summary,job_name


def require(ok:bool,message:str):
    if not ok:raise ValueError(message)


def exact_recursive(a,b,path='root'):
    """Compare nested state dictionaries, including every optimizer tensor."""
    if isinstance(a,torch.Tensor):
        require(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),f'Tensor mismatch {path}')
    elif isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),f'Dict keys mismatch {path}')
        for k in a:exact_recursive(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(tuple,list)):
        require(type(a)==type(b) and len(a)==len(b),f'Sequence mismatch {path}')
        for i,(x,y) in enumerate(zip(a,b)):exact_recursive(x,y,f'{path}[{i}]')
    else:require(a==b,f'Scalar mismatch {path}')


def audit_model(group_root:str,model_root:str):
    canonical_numerics(1);root=Path(group_root);out=Path(model_root)
    records=json.loads((out/'report.json').read_text())
    with np.load(root/'shared/evaluation.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    s=Setting(**{k:records[0][k] for k in asdict(Setting())})
    cases=natural_inputs(data,s);wa,wb=unpack_world(data['control']),unpack_world(data['witness'])
    true=wa.labels().numpy();wrong=wa.g[torch.arange(len(wa.x)),wa.hint].numpy()
    arrays=classes=replayed=maxerr=0
    for r in records:
        step=r['step']
        with np.load(out/f'predictions_{step}.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
        for key,(_,labels) in cases.items():
            l=saved[key];require(l.shape==(4096,8) and np.isfinite(l).all(),'Invalid saved logits')
            acc=float(100*(l.argmax(1)==labels).mean())
            require(acc==r['natural'][key]['accuracy_pct'],'Saved natural statistic mismatch')
            arrays+=1;classes+=len(l)
        if step in (2000,4000):
            ps=paired_summary(saved['pair_control'],saved['pair_witness'],wrong,true)
            require(ps==r['equal_function'],'Saved paired statistics mismatch')
            arrays+=2;classes+=2*len(true)
            cp=out/f'model_{step}.pt'
            require(sha256(cp)==r['checkpoint_sha256'],'Checkpoint file digest changed')
            c=torch.load(cp,map_location='cpu',weights_only=True)
            require(c['setting']==asdict(s),'Checkpoint setting mismatch')
            m=Model(s);m.load_state_dict(c['state_dict']);m.eval()
            # Full decisive pairs and a natural condition are independently replayed.
            for key,tokens in [('two_incorrect',cases['two_incorrect'][0]),('pair_control',encode(wa,s)),('pair_witness',encode(wb,s))]:
                actual=logits_for(m,tokens)
                err=float(np.max(np.abs(actual-saved[key])))
                maxerr=max(maxerr,err)
                require(np.array_equal(actual,saved[key]),'Checkpoint inference did not reproduce saved logits bitwise')
                replayed+=len(actual)
    return {'group':root.name,'model':out.name,'saved_arrays_recomputed':arrays,
        'saved_classes_recomputed':classes,'replayed_classes':replayed,'maximum_replay_logit_error':maxerr,
        'saved_statistics_match':True,'checkpoint_inference_bitwise_match':True}


def full_training_replay(primary:Path,out:Path):
    canonical_numerics(1);s=Setting();seed=10
    init=torch.load(primary/'shared'/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
    model=Model(s);model.load_state_dict(init['state_dict']);opt=optimizer(model)
    source=primary/'models'/job_name(seed,s)
    stream=np.load(primary/'shared'/f'train_{seed}.npy',allow_pickle=False,mmap_mode='r')
    start=time.monotonic();checked=[]
    for step in range(4000):
        model.train();w=unpack_world(stream[step]);opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(encode(w,s)),w.labels());loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step+1 in (2000,4000):
            reference=torch.load(source/f'model_{step+1}.pt',map_location='cpu',weights_only=True)
            exact_recursive(model.state_dict(),reference['state_dict'],'model')
            exact_recursive(opt.state_dict(),reference['optimizer'],'optimizer')
            checked.append({'update':step+1,'state_hash':state_hash(model.state_dict()),'weights_bitwise_equal':True,'optimizer_bitwise_equal':True})
    r={'seed':seed,'setting':asdict(s),'updates':4000,'examples':512000,'checkpoints':checked,
       'elapsed_seconds':time.monotonic()-start,'scope':'new independent run replay; not author training replay'}
    dump_json(out,r);return r


def run(execution:Path,out:Path,workers:int=4,replay_training:bool=True):
    if out.exists():raise FileExistsError(out)
    for group in ('primary','access'):
        require((execution/group/'completed.json').is_file(),f'Incomplete {group} grid')
    out.mkdir(parents=True);checks=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(audit_model,str(execution/g),str(p)) for g in ('primary','access')
                 for p in sorted((execution/g/'models').iterdir()) if p.is_dir()]
        for f in as_completed(futures):checks.append(f.result())
    require(len(checks)==64,'Expected exactly 64 models')
    r={'models':len(checks),'saved_arrays_recomputed':sum(v['saved_arrays_recomputed'] for v in checks),
       'saved_classes_recomputed':sum(v['saved_classes_recomputed'] for v in checks),
       'checkpoint_replayed_classes':sum(v['replayed_classes'] for v in checks),
       'maximum_replay_logit_error':max(v['maximum_replay_logit_error'] for v in checks),
       'all_checks_passed':True,'model_checks':checks,'author_replay':False}
    if replay_training:r['training_replay']=full_training_replay(execution/'primary',out/'training_replay.json')
    dump_json(out/'audit.json',r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('model_checks','training_replay')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--skip-training-replay',action='store_true')
    a=p.parse_args();run(a.execution,a.output,a.workers,not a.skip_training_replay)
