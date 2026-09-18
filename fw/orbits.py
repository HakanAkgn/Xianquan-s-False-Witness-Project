"""Complete token-balanced counterfactual cubes and Walsh interaction spectra.

The Fourier identities are standard. The experiment applies them to a family
of contexts having exactly identical task answers and token multisets.
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path
import numpy as np
import torch
from .data import Task, World, encode, equal_function_pair, validate_pair, balanced_original_pair
from .model import Architecture, Transformer
from .evaluate import predict, pack_outcomes


def cube_worlds(task, batch, dimensions, seed):
    if not 1 <= dimensions <= min(task.n-1,10):
        raise ValueError('Invalid or excessive cube dimension')
    rng=torch.Generator().manual_seed(seed)
    if task.relations==2:
        a, _, rows = equal_function_pair(task,batch,rng,dimensions)
        dummy=None
    else:
        a, _, rows, dummy = balanced_original_pair(task,batch,rng,dimensions)
    idx=torch.arange(batch)
    base_tokens=encode(a,task).sort(1).values
    for mask in range(1<<dimensions):
        w=a.copy()
        for j in range(dimensions):
            if mask & (1<<j):
                pos=rows[:,j]
                if task.relations==2:
                    w.f[idx,a.relation,pos]=a.f[idx,1-a.relation,pos]
                    w.f[idx,1-a.relation,pos]=a.f[idx,a.relation,pos]
                else:
                    w.f[idx,0,pos]=a.g[idx,dummy[:,j]]
                    w.g[idx,dummy[:,j]]=a.f[idx,0,pos]
        validate_pair(a,w,task,rows)
        if not torch.equal(base_tokens,encode(w,task).sort(1).values):
            raise AssertionError('Cube token multiset changed')
        yield mask,w


def walsh(values):
    """Normalized Walsh transform along axis 0, mask order in binary."""
    x=np.asarray(values,dtype=np.float64).copy()
    n=x.shape[0]
    if n<2 or n&(n-1):
        raise ValueError('First axis must be a power of two >= 2')
    step=1
    while step<n:
        for start in range(0,n,step*2):
            a=x[start:start+step].copy()
            b=x[start+step:start+2*step].copy()
            x[start:start+step]=a+b
            x[start+step:start+2*step]=a-b
        step*=2
    return x/n


def cube_statistics(margins, probabilities, labels, wrong):
    # Arrays: margin [2^k,B], probabilities [2^k,B,N].
    v=np.asarray(margins,dtype=np.float64)
    p=np.asarray(probabilities,dtype=np.float64)
    nmask,batch=v.shape
    k=nmask.bit_length()-1
    degrees=np.array([m.bit_count() for m in range(nmask)])
    coeff=walsh(v)
    variance=v.var(axis=0)
    energies=[float((coeff[degrees==d]**2).sum(0).mean()) for d in range(k+1)]
    nonconstant=sum(energies[1:])
    hi=sum(energies[2:])/nonconstant if nonconstant>1e-20 else None
    grad_energy=sum(np.mean((v-v[np.arange(nmask)^(1<<j)])**2,axis=0) for j in range(k))/4
    prediction=v[0][None,:]+sum(
        ((np.arange(nmask)&(1<<j))!=0)[:,None]*(v[1<<j]-v[0])[None,:] for j in range(k))
    held=degrees>=2
    rmse=float(np.sqrt(np.mean((v[held]-prediction[held])**2))) if held.any() else None
    constant_rmse=float(np.sqrt(np.mean((v[held]-v[0])**2))) if held.any() else None
    phat=p.argmax(-1)
    correct=phat==labels[None,:]
    follows=phat==wrong[None,:]
    avg=p.mean(0)
    target_prob=p[:,np.arange(batch),labels]
    mean_ce=float(-np.log(np.clip(target_prob,1e-30,1)).mean())
    averaged_ce=float(-np.log(np.clip(avg[np.arange(batch),labels],1e-30,1)).mean())
    return {
        'dimensions':k,'worlds':batch,'contexts':nmask*batch,
        'count_curve':[{'count':d,'incorrect_suggestion_following':float(follows[degrees==d].mean()),
                        'accuracy':float(correct[degrees==d].mean()),
                        'mean_wrong_vs_correct_margin':float(v[degrees==d].mean())}
                       for d in range(k+1)],
        'base_accuracy':float(correct[0].mean()),
        'mean_orbit_accuracy':float(correct.mean()),
        'worst_orbit_accuracy':float(correct.all(0).mean()),
        'base_correct_but_fragile_fraction':float((correct[0]&~correct.all(0)).mean()),
        'prediction_changes_somewhere_fraction':float((phat!=phat[0]).any(0).mean()),
        'mean_probability_range_wrong':float(np.ptp(p[:,np.arange(batch),wrong],axis=0).mean()),
        'margin_variance':float(variance.mean()),
        'walsh_energy_by_degree':energies,
        'higher_order_fraction_nonconstant_margin_energy':hi,
        'additive_singleton_heldout_margin_rmse':rmse,
        'constant_baseline_heldout_margin_rmse':constant_rmse,
        'additive_calibration_masks':[0]+[1<<j for j in range(k)],
        'heldout_masks':np.where(held)[0].tolist(),
        'parseval_max_error':float(np.max(np.abs(variance-(coeff[1:]**2).sum(0)))),
        'poincare_lower_min_slack':float(np.min(grad_energy-variance)),
        'poincare_upper_min_slack':float(np.min(k*variance-grad_energy)),
        'orbit_average_accuracy':float((avg.argmax(1)==labels).mean()),
        'mean_context_cross_entropy':mean_ce,
        'orbit_average_cross_entropy':averaged_ce,
        'jensen_gain':mean_ce-averaged_ce,
        'projection_scope':'oracle exact finite cube; no end-to-end learned robustness claim'
    }


def run_checkpoint(checkpoint,out,worlds=1024,dimensions=4,seed=907331):
    torch.set_num_threads(1)
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    task=Task(**c['metadata']['task'])
    model=Transformer(task,Architecture(**c['metadata']['architecture']))
    model.load_state_dict(c['state_dict']); model.eval()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results={}
    raw={}
    for mode in ['incorrect','absent','correct']:
        probs=[];margins=[]
        for mask,w in cube_worlds(task,worlds,dimensions,seed):
            true=w.labels().numpy()
            wrong=w.g[torch.arange(worlds),w.hint].numpy()
            relation=w.relation.numpy()
            if mode=='absent':w.present.fill_(False)
            elif mode=='correct':w.hint=w.intermediate()
            p,l=predict(model,w)
            probs.append(p)
            margins.append(l[np.arange(worlds),wrong]-l[np.arange(worlds),true])
        ps,ds=np.stack(probs),np.stack(margins)
        stat=cube_statistics(ds,ps,true,wrong)
        stat['query_relation_strata']={}
        for r in range(task.relations):
            sel=relation==r
            stat['query_relation_strata'][str(r)]=cube_statistics(ds[:,sel],ps[:,sel],true[sel],wrong[sel])
        results[mode]=stat
        raw[mode]=pack_outcomes(np.concatenate([true[None,:],wrong[None,:],relation[None,:],ps.argmax(-1)],axis=0))
        np.savez_compressed(out/f'cube_{mode}.npz',margins=ds,probabilities=ps,labels=true,wrong=wrong,relation=relation)
    results.update({'seed':c['metadata']['seed'],'step':c['step'],'evaluation_seed':seed, 'relations':task.relations, 'construction':'typed_exchange' if task.relations==2 else 'unused_company_compensation'})
    (out/'cube_report.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (out/'cube_outcomes.json').write_text(json.dumps(raw,indent=2)+'\n')
    print(json.dumps({'checkpoint':str(checkpoint),'cube':results['incorrect']}),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--worlds',type=int,default=1024);p.add_argument('--dimensions',type=int,default=4)
    a=p.parse_args();run_checkpoint(a.checkpoint,a.output,a.worlds,a.dimensions)

if __name__=='__main__':main()
