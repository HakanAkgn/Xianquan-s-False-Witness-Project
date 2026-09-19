"""Regenerated-data baseline diagnostic, never an author-replication certificate."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .core import Setting, Model, canonical_numerics, draw, encode, equal_pairs, logits_for, sha256, state_hash, dump_json


def make_inputs(s: Setting, n: int=4096, seed: int=20260919):
    """Own PCG64/CPU corpus, NOT the author's unrecovered arrays."""
    rng=torch.Generator(device='cpu').manual_seed(seed+1)
    w=draw(s,n,rng);idx=torch.arange(n);middle=w.f[idx,w.x]
    wrong=(middle+torch.randint(1,s.n,(n,),generator=rng))%s.n
    data={}
    for task in ('one','two'):
        for hint in ('absent','correct','incorrect'):
            v=w.copy();v.two.fill_(task=='two');v.present.fill_(hint!='absent')
            v.hint=middle.clone() if hint=='correct' else wrong.clone()
            key=task+'_'+hint
            data[key]=encode(v,s).numpy().astype(np.uint8)
            data[key+'_labels']=v.labels().numpy().astype(np.uint8)
    base,a,b,j=equal_pairs(s,n,seed)
    for name,obj in [('pair_base',base),('pair_control',a),('pair_witness',b)]:
        data[name]=encode(obj,s).numpy().astype(np.uint8)
    data['pair_labels']=a.labels().numpy().astype(np.uint8)
    data['pair_wrong']=a.g[idx,a.hint].numpy().astype(np.uint8)
    data['pair_row']=j
    return data


def score(model: Model, data: dict, out: Path | None=None):
    result={};arrays={}
    for task in ('one','two'):
        for hint in ('absent','correct','incorrect'):
            key=task+'_'+hint;l=logits_for(model,data[key]);pred=l.argmax(1)
            result[key+'_accuracy_pct']=float(100*np.mean(pred==data[key+'_labels']))
            arrays[key+'_logits']=l
    for key in ('pair_base','pair_control','pair_witness'):
        arrays[key+'_logits']=logits_for(model,data[key])
    ca=arrays['pair_control_logits'].argmax(1);wa=arrays['pair_witness_logits'].argmax(1)
    b=data['pair_wrong'];t=data['pair_labels']
    result.update(control_following_pct=float(100*np.mean(ca==b)),
        witness_following_pct=float(100*np.mean(wa==b)),
        effect_pp=float(100*np.mean((wa==b).astype(float)-(ca==b))),
        control_accuracy_pct=float(100*np.mean(ca==t)),witness_accuracy_pct=float(100*np.mean(wa==t)),
        became_suggested=int(((ca!=b)&(wa==b)).sum()),ceased_suggested=int(((ca==b)&(wa!=b)).sum()),
        both_suggested=int(((ca==b)&(wa==b)).sum()),neither_suggested=int(((ca!=b)&(wa!=b)).sum()))
    if out is not None:np.savez_compressed(out,**arrays)
    return result


def train_one(root: str, seed: int, updates: int, settings: dict):
    root=Path(root);s=Setting(**settings);env=canonical_numerics(1)
    out=root/f'seed_{seed}';out.mkdir(exist_ok=False)
    torch.manual_seed(seed);model=Model(s);initial=state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    rng=torch.Generator(device='cpu').manual_seed(35600+seed)
    with np.load(root/'evaluation_inputs.npz',allow_pickle=False) as d:data={k:d[k] for k in d.files}
    xtrain=np.empty((updates,128,s.length),dtype=np.uint8);ytrain=np.empty((updates,128),dtype=np.uint8)
    metadata={'seed':seed,'setting':asdict(s),'setting_sha256':s.fingerprint(),
        'environment':json.loads(json.dumps(env)),'training_rng_seed':35600+seed,'training_rng_device':'cpu',
        'training_random_call_order':'f,g,x,two_step,present,valid,nonzero_offset',
        'status':'regenerated_CPU_baseline_diagnostic_NOT_author_replication',
        'initial_state_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),
        'source_hashes':json.loads((root/'plan.json').read_text())['source_hashes'],
        'evaluation_inputs_sha256':sha256(root/'evaluation_inputs.npz')}
    torch.save({'state_dict':model.state_dict(),'metadata':metadata,'step':0},out/'initial.pt')
    trace=[];evals=[];start=time.perf_counter()
    for step in range(updates+1):
        if step%500==0 or step==updates:
            rec={'seed':seed,'updates':step,**score(model,data,out/f'predictions_{step}.npz')};evals.append(rec)
            dump_json(out/'evaluations.json',evals)
            print(json.dumps({'seed':seed,'step':step,'absent_accuracy':rec['two_absent_accuracy_pct'],
                'wrong_accuracy':rec['two_incorrect_accuracy_pct'],'equal_effect':rec['effect_pp']}),flush=True)
        if step==updates:break
        model.train();w=draw(s,128,rng);x=encode(w,s);y=w.labels();xtrain[step]=x.numpy();ytrain[step]=y.numpy()
        loss=F.cross_entropy(model(x),y);opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (step+1)%100==0:trace.append({'step':step+1,'loss':float(loss.detach()),
            'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-start})
    metadata['training_updates']=updates;metadata['elapsed_seconds']=time.perf_counter()-start
    np.savez_compressed(out/'training_stream.npz',tokens=xtrain,labels=ytrain)
    metadata['training_stream_sha256']=sha256(out/'training_stream.npz')
    metadata['final_state_sha256']=state_hash(model.state_dict())
    torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'training_rng_state':rng.get_state(),
        'torch_rng_state':torch.get_rng_state(),'metadata':metadata,'step':updates},out/f'model_{updates}.pt')
    dump_json(out/'metadata.json',metadata);dump_json(out/'training_trace.json',trace)
    return evals[-1]


def reference_rows(path: Path):
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k in ('seed','updates'):row[k]=int(row[k])
        for k in ('wrong_hint_accuracy_pct','control_following_pct','witness_following_pct','effect_pp'):row[k]=float(row[k])
    return rows


def run(root: str, workers: int=4, seeds=tuple(range(10,18)), updates: int=2000, order='company_first'):
    if workers<1 or updates<1 or not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid run parameters')
    root=Path(root);root.mkdir(parents=True,exist_ok=False);s=Setting(order=order);src=Path(__file__).parent
    plan={'status':'diagnostic_only','seed_list':list(seeds),'updates':updates,'setting':asdict(s),
        'setting_sha256':s.fingerprint(),'evaluation_seed':20260919,'evaluation_seed_is_author_seed':False,
        'source_hashes':{p.name:sha256(p) for p in [src/'core.py',src/'diagnostic.py']},
        'stop_rule':'No extensions. Regenerated CPU streams cannot unlock author-replay gate.',
        'scope':'Initial Figure 1/Table 3 company-first suggestion-last group; NOT the full manuscript'}
    dump_json(root/'plan.json',plan);np.savez_compressed(root/'evaluation_inputs.npz',**make_inputs(s))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs=[ex.submit(train_one,str(root),seed,updates,asdict(s)) for seed in seeds]
        for job in as_completed(jobs):results.append(job.result())
    results.sort(key=lambda r:r['seed']);ref=reference_rows(src.parent/'configs/paper_table3.csv');comparisons=[]
    for r in results:
        matches=[v for v in ref if v['seed']==r['seed'] and v['updates']==updates and v['order']==order];c=dict(r)
        if matches:
            p=matches[0]
            for k in ('wrong_hint_accuracy_pct','control_following_pct','witness_following_pct','effect_pp'):
                rk='two_incorrect_accuracy_pct' if k=='wrong_hint_accuracy_pct' else k
                c['paper_'+k]=p[k];c['difference_'+k]=r[rk]-p[k]
            c['matches_all_table3_rounded_values']=all(abs(c['difference_'+k])<=.000501 for k in
                ('wrong_hint_accuracy_pct','control_following_pct','witness_following_pct','effect_pp'))
        comparisons.append(c)
    with (root/'comparison.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(comparisons[0]));wr.writeheader();wr.writerows(comparisons)
    summary={'status':'exact_author_reproduction_NOT_established','extensions_allowed':False,'n_models':len(results),
        'updates':updates,'comparison':comparisons,'means':{k:float(np.mean([r[k] for r in results])) for k in
        ('two_absent_accuracy_pct','two_incorrect_accuracy_pct','control_following_pct','witness_following_pct','effect_pp')},
        'missing':['author source and constructor draw order','author GPU training streams',
                   'author fixed evaluation arrays','author checkpoints and reference logits'],
        'interpretation':'Newly generated data. Not evidence for or against correctness of the original runs.'}
    dump_json(root/'summary.json',summary)
    dump_json(root/'artifact_hashes.json',{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+',default=list(range(10,18)))
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--order',choices=['company_first','employee_first'],default='company_first')
    p.add_argument('--acknowledge-regenerated-data',action='store_true',required=True)
    a=p.parse_args();run(a.output,a.workers,tuple(a.seeds),a.updates,a.order)
