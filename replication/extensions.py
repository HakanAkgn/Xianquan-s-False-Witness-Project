"""Answer/token-preserving evaluations on exactly the baseline checkpoints.

No architecture, vocabulary, layout or checkpoint tensors are changed.
Balancing uses the common numeric alphabet and an unused-company compensation;
it preserves all person-country answers, not standalone company-country facts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from reproduction.core import Setting, Model, World, draw, encode, canonical_numerics, logits_for, sha256, dump_json
from .run import pack_world,unpack_world,paired_summary

SEED=2026091910


def balanced_world(s: Setting,n: int,rng: torch.Generator,k: int=1):
    if s.n<4 or not 1<=k<=s.n-3 or n<1:raise ValueError('Invalid balanced construction size')
    w=draw(s,n,rng);idx=torch.arange(n)
    order=torch.rand(n,s.n,generator=rng).argsort(1)
    a,h,r=order[:,:3].unbind(1);dummy=order[:,3:3+k]
    scores=torch.rand(n,s.n,generator=rng);scores[idx,h]=2;scores[idx,r]=2
    B,T=scores.argsort(1)[:,:2].unbind(1)
    w.g[idx,h]=B;w.g[idx,r]=B;w.g[idx,a]=T
    allowed=torch.ones(n,s.n,dtype=torch.bool);allowed[idx,h]=False;allowed[idx,r]=False
    allowed.scatter_(1,dummy,False)
    pool=torch.arange(s.n).expand(n,s.n)[allowed].reshape(n,s.n-k-2)
    choices=torch.randint(pool.shape[1],(n,s.n),generator=rng)
    w.f=torch.gather(pool,1,choices);w.f[idx,w.x]=a
    scores=torch.rand(n,s.n,generator=rng);scores[idx,w.x]=2
    rows=scores.argsort(1)[:,:k]
    for j in range(k):w.f[idx,rows[:,j]]=r;w.g[idx,dummy[:,j]]=h
    w.hint=h;w.present.fill_(True);w.two.fill_(True)
    if torch.any(w.answers()[idx,w.x]==B):raise AssertionError('Nondecisive balanced world')
    return w,rows,dummy


def edit(w: World,rows: torch.Tensor,dummy: torch.Tensor,mask: int,employee=True,compensation=True)->World:
    k=rows.shape[1]
    if mask<0 or mask>=2**k:raise ValueError('Mask out of range')
    v=w.copy();idx=torch.arange(len(w.x))
    for j in range(k):
        if mask&(1<<j):
            if employee:v.f[idx,rows[:,j]]=w.g[idx,dummy[:,j]]
            if compensation:v.g[idx,dummy[:,j]]=w.f[idx,rows[:,j]]
    return v


def validate(w: World,v: World,s: Setting,token_balance: bool=True):
    idx=torch.arange(len(w.x))
    if not torch.equal(w.answers(),v.answers()):raise AssertionError('Country-answer function changed')
    if not torch.equal(w.labels(),v.labels()):raise AssertionError('Queried label changed')
    if not torch.equal(w.f[idx,w.x],v.f[idx,v.x]):raise AssertionError('Queried employer changed')
    for key in ('x','two','present','hint'):
        if not torch.equal(getattr(w,key),getattr(v,key)):raise AssertionError(f'{key} changed')
    if token_balance and not torch.equal(encode(w,s).sort(1).values,encode(v,s).sort(1).values):
        raise AssertionError('Token multiplicity changed')


def relevant(w: World)->World:
    v=w.copy();v.f[torch.arange(len(w.x)),w.x]=w.hint
    if torch.any(v.labels()==w.labels()):raise AssertionError('Relevant edit failed to change label')
    return v


def make_inputs(path: Path):
    if path.exists():raise FileExistsError(path)
    s=Setting();data={}
    for name,n,k,seed in [('balanced',4096,1,SEED),('cube',1024,4,SEED+1)]:
        w,rows,dummy=balanced_world(s,n,torch.Generator().manual_seed(seed),k)
        for mask in range(1<<k):validate(w,edit(w,rows,dummy,mask),s)
        data[name]=pack_world(w);data[name+'_rows']=rows.numpy();data[name+'_dummy']=dummy.numpy()
    np.savez_compressed(path,**data)


def walsh(values:np.ndarray)->np.ndarray:
    x=np.asarray(values,dtype=np.float64).copy();n=x.shape[0]
    if n<2 or n&(n-1):raise ValueError('Need a power-of-two first axis')
    step=1
    while step<n:
        for start in range(0,n,2*step):
            a=x[start:start+step].copy();b=x[start+step:start+2*step].copy()
            x[start:start+step]=a+b;x[start+step:start+2*step]=a-b
        step*=2
    return x/n


def cube_stats(logits: np.ndarray, truth:np.ndarray,wrong:np.ndarray)->dict:
    l=np.asarray(logits,dtype=np.float64);nmask,batch,nclass=l.shape;k=nmask.bit_length()-1
    if nmask!=1<<k:raise ValueError('Incomplete cube')
    i=np.arange(batch);D=l[:,i,wrong]-l[:,i,truth]
    degree=np.array([m.bit_count() for m in range(nmask)]);coef=walsh(D)
    energies=[float((coef[degree==d]**2).sum(0).mean()) for d in range(k+1)]
    var=float(D.var(0).mean());high=sum(energies[2:])/var if var>1e-20 else None
    prediction=D[0][None,:]+sum(((np.arange(nmask)&(1<<j))!=0)[:,None]*(D[1<<j]-D[0])[None,:] for j in range(k))
    held=degree>=2;pred=l.argmax(-1);correct=pred==truth[None,:]
    def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
    return {'worlds':batch,'contexts':nmask*batch,'dimensions':k,
        'mean_accuracy_pct':float(100*correct.mean()),'robust_accuracy_pct':float(100*correct.all(0).mean()),
        'base_accuracy_pct':float(100*correct[0].mean()),
        'count_curve':[{'count':j,'designated_wrong_pct':float(100*(pred[degree==j]==wrong[None,:]).mean()),
                        'accuracy_pct':float(100*correct[degree==j].mean())} for j in range(k+1)],
        'margin_variance':var,'energy_by_degree':energies,'higher_order_share':high,
        'additive_heldout_rmse':rmse(D[held],prediction[held]),'constant_heldout_rmse':rmse(D[held],D[0]),
        'parseval_max_error':float(np.max(np.abs(D.var(0)-(coef[1:]**2).sum(0)))),
        'prediction_scope':'calibrated on zero/singletons in each world; held-out combinations of same worlds',
        'robustness_scope':'all 16 tested contexts, not every equivalent prompt'}


def load_checkpoint(path: str):
    c=torch.load(path,map_location='cpu',weights_only=True);s=Setting(**c['setting'])
    m=Model(s);m.load_state_dict(c['state_dict']);m.eval()
    return c,s,m


def evaluate_model(m:Model,data:dict,out:Path,with_cube=True)->dict:
    s=m.setting;report={'balanced':{},'sensitivity':{}};raw={}
    w=unpack_world(data['balanced']);rows=torch.tensor(data['balanced_rows']);dummy=torch.tensor(data['balanced_dummy'])
    contexts={'control':w,'employee_only':edit(w,rows,dummy,1,True,False),
        'compensation_only':edit(w,rows,dummy,1,False,True),'balanced':edit(w,rows,dummy,1)}
    for label,c in contexts.items():validate(w,c,s,token_balance=label in ('control','balanced'))
    i=np.arange(len(w.x));true=w.labels().numpy();wrong=w.g[torch.arange(len(w.x)),w.hint].numpy()
    for mode in ('incorrect','absent','correct'):
        ls={}
        for label,context in contexts.items():
            c=context.copy()
            if mode=='absent':c.present.fill_(False)
            if mode=='correct':c.hint=c.f[torch.arange(len(c.x)),c.x]
            ls[label]=logits_for(m,encode(c,s));raw['balanced_'+mode+'_'+label]=ls[label]
        stats=paired_summary(ls['control'],ls['balanced'],wrong,true)
        margin={label:l[i,wrong]-l[i,true] for label,l in ls.items()}
        stats['employee_only_margin_effect']=float((margin['employee_only']-margin['control']).mean())
        stats['compensation_only_margin_effect']=float((margin['compensation_only']-margin['control']).mean())
        stats['factorial_margin_interaction']=float((margin['balanced']-margin['employee_only']-margin['compensation_only']+margin['control']).mean())
        stats['rate_scope']='designated wrong country; only incorrect mode actually supplies wrong suggestion'
        report['balanced'][mode]=stats
    # A no-hint relevant edit requires a different answer and tests context use.
    c=w.copy();c.present.fill_(False);v=relevant(c)
    l0=logits_for(m,encode(c,s));l1=logits_for(m,encode(v,s));raw['relevant_control']=l0;raw['relevant_changed']=l1
    report['sensitivity']={'control_accuracy_pct':float(100*(l0.argmax(1)==c.labels().numpy()).mean()),
        'changed_answer_accuracy_pct':float(100*(l1.argmax(1)==v.labels().numpy()).mean()),
        'prediction_change_pct':float(100*(l0.argmax(1)!=l1.argmax(1)).mean())}
    if with_cube:
        report['cube']={};w=unpack_world(data['cube']);rows=torch.tensor(data['cube_rows']);dummy=torch.tensor(data['cube_dummy'])
        true=w.labels().numpy();wrong=w.g[torch.arange(len(w.x)),w.hint].numpy()
        for mode in ('incorrect','absent','correct'):
            logits=[]
            for mask in range(16):
                c=edit(w,rows,dummy,mask);validate(w,c,s)
                if mode=='absent':c.present.fill_(False)
                if mode=='correct':c.hint=c.f[torch.arange(len(c.x)),c.x]
                logits.append(logits_for(m,encode(c,s)))
            l=np.stack(logits);raw['cube_'+mode]=l
            report['cube'][mode]=cube_stats(l,true,wrong)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'outputs.npz',**raw);dump_json(out/'report.json',report)
    return report


def evaluate_one(checkpoint:str,inputs:str,destination:str,with_cube:bool)->dict:
    canonical_numerics(1);c,s,m=load_checkpoint(checkpoint)
    with np.load(inputs,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    report=evaluate_model(m,data,Path(destination),with_cube)
    report.update({'seed':c['seed'],'step':c['step'],**asdict(s),'checkpoint_sha256':sha256(checkpoint),'inputs_sha256':sha256(inputs)})
    dump_json(Path(destination)/'report.json',report)
    print(f"extension {Path(checkpoint).parent.name} {c['step']} effect={report['balanced']['incorrect']['effect_pp']:.4f}",flush=True)
    return report


def run(primary:Path,out:Path,assessment:Path,workers:int=4):
    if not (primary/'completed.json').is_file() or not assessment.is_file():
        raise ValueError('Complete and assess the baseline before same-setting extensions')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);inputs=out/'inputs.npz';make_inputs(inputs)
    cps=sorted(primary.glob('models/*/model_*.pt'))
    if len(cps)!=64:raise ValueError('Retain all primary checkpoints')
    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(evaluate_one,str(p),str(inputs),str(out/(p.parent.name+'__'+p.stem)),
             'company_first__suggestion_last' in p.parent.name) for p in cps]
        for f in as_completed(fs):reports.append(f.result())
    dump_json(out/'summary.json',reports)
    dump_json(out/'completed.json',{'checkpoints':len(reports),'all_completed':True,'baseline_assessment_sha256':sha256(assessment)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--assessment',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.primary,a.output,a.assessment,a.workers)
