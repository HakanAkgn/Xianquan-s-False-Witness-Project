"""Four controlled continuations of each original baseline checkpoint.

500 updates x 512 contexts in every arm; the paper's 25/75 task mix remains.
The IID arm uses the next 256,000 saved ordinary examples, regrouped into 512s.
Augmentation/consistency share exactly the same examples. A separate control
matches increased wrong-hint exposure and decisive country differences without
using paired answer-preserving context edits. It does not match every table
statistic induced by the balanced generator.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from dataclasses import asdict
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from reproduction.core import Setting,World,Model,draw,encode,canonical_numerics,sha256,state_hash,dump_json
from .run import unpack_world,pack_world,natural_inputs,optimizer,job_name
from .extensions import balanced_world,edit,relevant,evaluate_model,load_checkpoint

ARMS=('iid','hint_frequency_control','augmentation','consistency')


def js_divergence(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    la,lb=a.log_softmax(-1),b.log_softmax(-1);pa,pb=la.exp(),lb.exp()
    lm=torch.logaddexp(la,lb)-np.log(2.)
    return .5*((pa*(la-lm)).sum(-1)+(pb*(lb-lm)).sum(-1)).mean()


def sliced(w:World,start:int,end:int)->World:
    return World(**{k:v[start:end].clone() for k,v in vars(w).items()})


def wrong_iid(s:Setting,n:int,rng:torch.Generator)->World:
    """IID worlds conditioned on a decisive wrong intermediate suggestion."""
    w=draw(s,n,rng);i=torch.arange(n)
    w.hint=(w.f[i,w.x]+torch.randint(1,s.n,(n,),generator=rng))%s.n
    bad=w.g[i,w.hint]==w.g[i,w.f[i,w.x]]
    while bad.any():
        pos=bad.nonzero().flatten();v=draw(s,len(pos),rng);j=torch.arange(len(pos))
        v.hint=(v.f[j,v.x]+torch.randint(1,s.n,(len(pos),),generator=rng))%s.n
        for field in vars(w):getattr(w,field)[pos]=getattr(v,field)
        bad=w.g[i,w.hint]==w.g[i,w.f[i,w.x]]
    w.present.fill_(True)
    return w


def batch_for(s:Setting,ordinary:World,arm:str,pair_rng:torch.Generator,control_rng:torch.Generator):
    if arm not in ARMS:raise ValueError('Unknown repair arm')
    if len(ordinary.x)!=512:raise ValueError('Expected 512 ordinary examples')
    if arm=='iid':return [ordinary]
    iid=sliced(ordinary,0,128);two=ordinary.two[128:256].clone()
    if arm in ('augmentation','consistency'):
        a,rows,dummy=balanced_world(s,128,pair_rng,1);b=edit(a,rows,dummy,1);c=relevant(a)
    else:
        a=wrong_iid(s,128,control_rng);b=wrong_iid(s,128,control_rng)
        c=draw(s,128,control_rng);c.hint=c.f[torch.arange(128),c.x];c.present.fill_(True)
    for w in (a,b,c):w.two=two.clone()
    return [iid,a,b,c]


def run_one(checkpoint:str,primary_root:str,extension_inputs:str,out_string:str,arm:str)->dict:
    canonical_numerics(1)
    before=sha256(checkpoint);c,s,m=load_checkpoint(checkpoint)
    if c['step']!=2000 or s!=Setting():raise ValueError('Repair starts at the exact primary company-first suggestion-last 2,000-update setting')
    seed=c['seed'];out=Path(out_string);out.mkdir(parents=True,exist_ok=False)
    opt=optimizer(m);opt.load_state_dict(c['optimizer'])
    start_hash=state_hash(m.state_dict())
    stream=np.load(Path(primary_root)/'shared'/f'train_{seed}.npy',allow_pickle=False,mmap_mode='r')[2000:4000]
    if stream.shape!=(2000,128,20):raise ValueError('Unexpected continuation data')
    stream=stream.reshape(500,512,20)
    prng=torch.Generator().manual_seed(2026092000+seed)
    crng=torch.Generator().manual_seed(2026093000+seed)
    trace=[];train_hash=hashlib.sha256();semantic=np.empty((500,512,20),dtype=np.uint8)
    counts={'examples':0,'two':0,'present':0,'correct_present':0};start=time.monotonic()
    for step in range(500):
        worlds=batch_for(s,unpack_world(stream[step]),arm,prng,crng)
        tokens=torch.cat([encode(w,s) for w in worlds]);labels=torch.cat([w.labels() for w in worlds])
        packed=np.concatenate([pack_world(w) for w in worlds]);semantic[step]=packed;train_hash.update(packed.tobytes())
        for w in worlds:
            idx=torch.arange(len(w.x));counts['examples']+=len(w.x);counts['two']+=int(w.two.sum())
            counts['present']+=int(w.present.sum());counts['correct_present']+=int(((w.hint==w.f[idx,w.x])&w.present).sum())
        m.train();opt.zero_grad(set_to_none=True);logits=m(tokens);ce=F.cross_entropy(logits,labels)
        js=js_divergence(logits[128:256],logits[256:384]) if arm=='consistency' else logits.new_zeros(())
        loss=ce+js
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite repair loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%100==0:
            trace.append({'step':step+1,'cross_entropy':float(ce.detach()),'js':float(js.detach()),'loss':float(loss.detach())})
    model_path=out/'model.pt'
    torch.save({'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'step':2500,
        'setting':asdict(s),'arm':arm,'base_checkpoint_sha256':before,'status':'same_setting_continuation'},model_path)
    np.savez_compressed(out/'training_stream.npz',worlds=semantic)
    with np.load(extension_inputs,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    result=evaluate_model(m,data,out/'evaluation',True)
    with np.load(Path(primary_root)/'shared/evaluation.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    cases=natural_inputs(d,s);nat={};raw={}
    from reproduction.core import logits_for
    for key,(tokens,labels) in cases.items():
        l=logits_for(m,tokens);raw[key]=l;nat[key]={'accuracy_pct':float(100*(l.argmax(1)==labels).mean()),'n':len(labels)}
    np.savez_compressed(out/'natural_outputs.npz',**raw)
    if sha256(checkpoint)!=before or start_hash!=state_hash(c['state_dict']):raise RuntimeError('Baseline checkpoint changed')
    result.update({'seed':seed,'arm':arm,'base_step':2000,'extra_updates':500,'batch_size':512,
        'setting':asdict(s),'natural':nat,'base_checkpoint_sha256':before,'initial_state_hash':start_hash,
        'training_semantic_sha256':train_hash.hexdigest(),'training_counts':counts,
        'two_fraction':counts['two']/counts['examples'],'hint_fraction':counts['present']/counts['examples'],
        'hint_reliability_when_present':counts['correct_present']/counts['present'],
        'checkpoint_sha256':sha256(model_path),'elapsed_seconds':time.monotonic()-start,
        'inference':'ordinary unchanged transformer; no symbolic oracle'})
    dump_json(out/'trace.json',trace);dump_json(out/'report.json',result)
    print(f"repair seed={seed} arm={arm} robust={result['cube']['incorrect']['robust_accuracy_pct']:.4f}",flush=True)
    return result


def run(primary:Path,extensions:Path,out:Path,workers:int=4):
    if not (extensions/'completed.json').is_file():raise ValueError('Assess the same-checkpoint extension evaluations first')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[]
        for seed in range(10,18):
            cp=primary/'models'/job_name(seed,Setting())/'model_2000.pt'
            for arm in ARMS:
                fs.append(pool.submit(run_one,str(cp),str(primary),str(extensions/'inputs.npz'),str(out/f'{arm}_seed_{seed}'),arm))
        for f in as_completed(fs):reports.append(f.result())
    for seed in range(10,18):
        rs={r['arm']:r for r in reports if r['seed']==seed}
        if len({r['base_checkpoint_sha256'] for r in rs.values()})!=1:raise RuntimeError('Repair starts differ')
        if rs['augmentation']['training_semantic_sha256']!=rs['consistency']['training_semantic_sha256']:
            raise RuntimeError('Consistency and augmentation training data differ')
        for metric in ('two_fraction','hint_fraction','hint_reliability_when_present'):
            if rs['augmentation'][metric]!=rs['hint_frequency_control'][metric]:
                raise RuntimeError('Hint/task frequency control does not match augmented exposure')
    dump_json(out/'summary.json',reports)
    dump_json(out/'completed.json',{'runs':len(reports),'matched_starts_verified':True,'matched_augmentation_consistency_data':True,
        'matched_hint_control_exposure':True,'all_completed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--extensions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.primary,a.extensions,a.output,a.workers)
