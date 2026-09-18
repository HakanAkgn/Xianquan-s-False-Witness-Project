"""Compute-matched counterfactual training with a separate augmentation control.

Each update sees 512 supervised contexts in each arm. All arms start from
exactly the same 2,000-update checkpoint and optimizer state within each seed.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Task,draw,encode,balanced_original_pair,relevant_change
from .model import Architecture,Transformer
from .evaluate import evaluate
from .balanced import run as balanced_evaluate
from .orbits import run_checkpoint as cube_evaluate


def js_divergence(logits_a,logits_b):
    loga,logb=logits_a.log_softmax(-1),logits_b.log_softmax(-1)
    pa,pb=loga.exp(),logb.exp()
    logm=((pa+pb)/2).clamp_min(1e-30).log()
    return .5*((pa*(loga-logm)).sum(-1)+(pb*(logb-logm)).sum(-1)).mean()


def run(checkpoint,out,arm,steps=500,weight=1.):
    if arm not in ['iid','augmentation','consistency']:
        raise ValueError('Unknown arm')
    torch.set_num_threads(1)
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    task=Task(**c['metadata']['task'])
    if task.relations!=1:raise ValueError('This repair experiment uses the original task')
    torch.manual_seed(c['metadata']['seed']+777)
    torch.use_deterministic_algorithms(True)
    model=Transformer(task,Architecture(**c['metadata']['architecture']))
    model.load_state_dict(c['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    opt.load_state_dict(c['optimizer'])
    seed=c['metadata']['seed']
    iid_rng=torch.Generator().manual_seed(seed+500000)
    pair_rng=torch.Generator().manual_seed(seed+600000)
    extra_rng=torch.Generator().manual_seed(seed+700000)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    trace=[];start=time.perf_counter()
    for step in range(1,steps+1):
        model.train()
        iid=draw(task,128,iid_rng)
        if arm=='iid':
            extra=draw(task,384,extra_rng)
            x=torch.cat((encode(iid,task),encode(extra,task)))
            y=torch.cat((iid.labels(),extra.labels()))
        else:
            a,b,_,_=balanced_original_pair(task,128,pair_rng,1)
            changed=relevant_change(a,task)
            x=torch.cat([encode(w,task) for w in [iid,a,b,changed]])
            y=torch.cat([w.labels() for w in [iid,a,b,changed]])
        logits=model(x)
        ce=F.cross_entropy(logits,y)
        js=js_divergence(logits[128:256],logits[256:384]) if arm!='iid' else logits.new_zeros(())
        loss=ce+(weight*js if arm=='consistency' else 0.)
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%100==0:
            trace.append({'step':step,'ce':float(ce.detach()),'js':float(js.detach()),'loss':float(loss.detach()),
                          'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-start})
    meta={**c['metadata'],'repair_arm':arm,'repair_steps':steps,'repair_supervised_batch':512,
          'repair_js_weight':weight if arm=='consistency' else 0.,'base_checkpoint':str(checkpoint),
          'base_step':c['step'],'training_distribution':'iid' if arm=='iid' else '128 iid + 128 controls + 128 witnesses + 128 relevant changes'}
    path=out/'model.pt'
    torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'metadata':meta,'step':c['step']+steps},path)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'training_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    r,raw=evaluate(model,n=4096,mechanism=False)
    r.update({'seed':seed,'arm':arm,'base_step':c['step'],'extra_steps':steps})
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    (out/'outcomes.json').write_text(json.dumps(raw,indent=2)+'\n')
    br=balanced_evaluate(path,out,n=4096)
    cr=cube_evaluate(path,out/'orbit',worlds=1024,dimensions=4)
    print(json.dumps({'repair':arm,'seed':seed,'clean':r['clean']['absent']['accuracy'],
                      'balanced_effect':br['incorrect']['effect'],
                      'cube_robust_accuracy':cr['incorrect']['worst_orbit_accuracy'],
                      'seconds':time.perf_counter()-start}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--arm',required=True,choices=['iid','augmentation','consistency'])
    p.add_argument('--steps',type=int,default=500);p.add_argument('--weight',type=float,default=1.)
    a=p.parse_args();run(a.checkpoint,a.output,a.arm,a.steps,a.weight)
