"""Local second-layer Q/K/V interventions on unchanged baseline checkpoints.

Manuscript E.3/E.4/E.5 constructions. New PCG64 inputs are retained; reference
GPU arrays are not substituted. All unpatched vectors and residuals remain the
recipient's. Explicit float64 evaluation is calibrated to the native encoder.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from reproduction.core import Model, Setting, World, encode, equal_pairs, draw, canonical_numerics, sha256, dump_json
from .run import pack_world, unpack_world


@torch.no_grad()
def capture(m: Model, tokens: torch.Tensor) -> dict:
    x=m.token(tokens)+m.position
    x=m.encoder.layers[0](x,src_mask=m.first_mask,is_causal=False)
    layer=m.encoder.layers[1];z=layer.norm1(x)
    q,k,v=F.linear(z,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,-1)
    b,l,d=x.shape;h=m.setting.heads
    split=lambda t:t.reshape(b,l,h,d//h).transpose(1,2)
    return {'q':split(q),'k':split(k),'v':split(v),'residual':x[:,-1]}


@torch.no_grad()
def finish(m: Model, cache: dict, changes: dict | None=None) -> torch.Tensor:
    if changes and set(changes)-{'q','k','v'}:raise ValueError('Only projected Q/K/V may be changed')
    c={**cache,**(changes or {})};q,k,v=c['q'],c['k'],c['v'];b=q.shape[0]
    scores=(q[:,:,-1:,:]@k.transpose(-1,-2))/(q.shape[-1]**.5)
    scores=scores.masked_fill(m.last_mask[-1][None,None,None,:],float('-inf'))
    z=(scores.softmax(-1)@v).transpose(1,2).reshape(b,m.setting.width)
    layer=m.encoder.layers[1]
    x=c['residual']+layer.self_attn.out_proj(z)
    x=x+layer.linear2(F.gelu(layer.linear1(layer.norm2(x)),approximate='none'))
    return m.head(m.norm(x))


def patch(recipient: torch.Tensor, donor: torch.Tensor, destination, source=None) -> torch.Tensor:
    """Indices are a slice, a fixed integer, or one position per batch row."""
    out=recipient.clone();source=destination if source is None else source
    b=torch.arange(len(out))
    dst=(b[:,None],torch.arange(out.shape[1])[None,:],destination[:,None]) if isinstance(destination,torch.Tensor) else (slice(None),slice(None),destination)
    src=(b[:,None],torch.arange(out.shape[1])[None,:],source[:,None]) if isinstance(source,torch.Tensor) else (slice(None),slice(None),source)
    out[dst]=donor[src]
    return out


def sub(w: World,a:int,b:int)->World:
    return World(**{k:v[a:b] for k,v in vars(w).items()})


def make_inputs(path: Path) -> None:
    if path.exists():raise FileExistsError(path)
    s=Setting();_,a,b,rows=equal_pairs(s,2048,400)
    i=torch.arange(2048);truth=a.labels().numpy();wrong=a.g[i,a.hint].numpy()
    rng=np.random.default_rng(400001)
    third=np.array([rng.choice([c for c in range(s.n) if c not in (t,h)]) for t,h in zip(truth,wrong)])
    v=a.copy();r=a.f[i,rows];v.g[i,r]=torch.tensor(third);v.g[i,a.hint]=torch.tensor(third)
    changed=a.answers()!=v.answers()
    if not torch.equal(v.labels(),a.labels()) or not torch.all(changed.sum(1)==1) or not changed[i,rows].all():
        raise ValueError('The E.4 value donor changed the wrong answer function')
    w=draw(s,4096,torch.Generator().manual_seed(2026091903));i=torch.arange(4096)
    w.two.fill_(True);w.present.fill_(True)
    w.hint=(w.f[i,w.x]+torch.randint(1,8,(4096,),generator=torch.Generator().manual_seed(2026091904)))%8
    answers=w.answers().numpy();true=w.labels().numpy();wrongq=w.g[i,w.hint].numpy()
    alt=np.zeros(4096,dtype=np.int64);eligible=np.zeros(4096,bool)
    rng=np.random.default_rng(2026091905)
    for j in range(4096):
        candidates=np.flatnonzero((answers[j]!=true[j])&(answers[j]!=wrongq[j]))
        if len(candidates):alt[j]=rng.choice(candidates);eligible[j]=True
        else:alt[j]=int(w.x[j])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,control=pack_world(a),witness=pack_world(b),value_donor=pack_world(v),
        edited_rows=rows,third=third,query_base=pack_world(w),alternate_x=alt,eligible=eligible)


def effect_summary(logits: dict, truth: np.ndarray, target: np.ndarray, reference: str) -> dict:
    rows=np.arange(len(truth));baseline=logits[reference];bp=baseline.argmax(1)
    bm=baseline[rows,target]-baseline[rows,truth]
    result={}
    for key,l in logits.items():
        p=l.argmax(1)
        result[key]={'target_pct':float(100*(p==target).mean()),'accuracy_pct':float(100*(p==truth).mean()),
            'target_change_pp':float(100*((p==target).astype(float)-(bp==target)).mean()),
            'margin_change':float((l[rows,target]-l[rows,truth]-bm).mean())}
    return result


def evaluate_one(checkpoint: str, inputs: str, destination: str) -> dict:
    canonical_numerics(1);c=torch.load(checkpoint,map_location='cpu',weights_only=True)
    s=Setting(**c['setting']);m=Model(s);m.load_state_dict(c['state_dict']);m=m.double().eval()
    with np.load(inputs,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    a,b,v=(unpack_world(d[k]) for k in ('control','witness','value_donor'))
    rows=torch.tensor(d['edited_rows']);fstart=s.n if s.order=='company_first' else 0
    fpos=slice(fstart,fstart+s.n);gstart=0 if fstart else s.n;gpos=slice(gstart,gstart+s.n)
    channel={k:[] for k in ('base','full','F_keys','F_values','G_keys','G_values','final_Q')}
    crossed={k:[] for k in ('neither','K','V','K_and_V','V_at_query')}
    max_error=0.;class_mismatches=0;identity_error=0.
    for start in range(0,len(rows),128):
        end=min(start+128,len(rows));aa,bb,vv=(sub(w,start,end) for w in (a,b,v))
        xa,xb,xv=(encode(w,s) for w in (aa,bb,vv))
        ca,cb,cv=(capture(m,x) for x in (xa,xb,xv))
        base=finish(m,ca)
        with torch.no_grad():native=m(xa)
        max_error=max(max_error,float((base-native).abs().max()))
        class_mismatches+=int((base.argmax(1)!=native.argmax(1)).sum())
        identity_error=max(identity_error,float((finish(m,ca,{'k':ca['k'].clone()})-base).abs().max()))
        channel['base'].append(base.numpy());channel['full'].append(finish(m,cb).numpy())
        for label,field,positions in [('F_keys','k',fpos),('F_values','v',fpos),('G_keys','k',gpos),('G_values','v',gpos),('final_Q','q',-1)]:
            changed=patch(ca[field],cb[field],positions)
            channel[label].append(finish(m,ca,{field:changed}).numpy())
        j=fstart+rows[start:end];x=fstart+aa.x
        kj=patch(ca['k'],cb['k'],j);vj=patch(ca['v'],cv['v'],j);vx=patch(ca['v'],cv['v'],x,j)
        for key,changes in [('neither',{}),('K',{'k':kj}),('V',{'v':vj}),('K_and_V',{'k':kj,'v':vj}),('V_at_query',{'v':vx})]:
            crossed[key].append(finish(m,ca,changes).numpy())
    channel={k:np.concatenate(v) for k,v in channel.items()};crossed={k:np.concatenate(v) for k,v in crossed.items()}
    truth=a.labels().numpy();wrong=a.g[torch.arange(len(a.x)),a.hint].numpy();third=d['third']
    ch=effect_summary(channel,truth,wrong,'base');cr=effect_summary(crossed,truth,third,'neither')
    interaction=cr['K_and_V']['target_pct']-cr['K']['target_pct']-cr['V']['target_pct']+cr['neither']['target_pct']
    qbase=unpack_world(d['query_base']);eligible=d['eligible'];alt=torch.tensor(d['alternate_x'])
    qs={k:[] for k in ('base','same_Q','alternate_Q')}
    table_identity=0.
    for start in range(0,len(qbase.x),128):
        end=min(start+128,len(qbase.x));w=sub(qbase,start,end)
        same=w.copy();same.present.fill_(False)
        other=same.copy();other.x=alt[start:end]
        cc,cs,co=(capture(m,encode(x,s)) for x in (w,same,other))
        for field in ('k','v'):
            table_identity=max(table_identity,float((cc[field][:,:,:2*s.n]-cs[field][:,:,:2*s.n]).abs().max()),
                               float((cc[field][:,:,:2*s.n]-co[field][:,:,:2*s.n]).abs().max()))
        qs['base'].append(finish(m,cc).numpy())
        qs['same_Q'].append(finish(m,cc,{'q':patch(cc['q'],cs['q'],-1)}).numpy())
        qs['alternate_Q'].append(finish(m,cc,{'q':patch(cc['q'],co['q'],-1)}).numpy())
    qs={k:np.concatenate(v) for k,v in qs.items()}
    qt=qbase.labels().numpy();qa=qbase.answers()[torch.arange(len(alt)),alt].numpy()
    qsummary={}
    for name,sel in [('all',np.ones(len(alt),bool)),('eligible',eligible),
                     ('eligible_decisive',eligible&(qt!=qbase.g[torch.arange(len(alt)),qbase.hint].numpy()))]:
        if not sel.any():continue
        qsummary[name]={'n':int(sel.sum()),
            'base_true_pct':float(100*(qs['base'].argmax(1)[sel]==qt[sel]).mean()),
            'same_Q_true_pct':float(100*(qs['same_Q'].argmax(1)[sel]==qt[sel]).mean()),
            'base_alternate_pct':float(100*(qs['base'].argmax(1)[sel]==qa[sel]).mean()),
            'alternate_Q_alternate_pct':float(100*(qs['alternate_Q'].argmax(1)[sel]==qa[sel]).mean())}
    report={'seed':c['seed'],'step':c['step'],**asdict(s),'checkpoint_sha256':sha256(checkpoint),
        'inputs_sha256':sha256(inputs),'channels':ch,'crossed':cr,'crossed_interaction_pp':interaction,
        'query':qsummary,'validation':{'native_float64_max_logit_error':max_error,
        'native_float64_class_mismatches':class_mismatches,'self_patch_error':identity_error,
        'query_donor_table_KV_max_error':table_identity},
        'scope':'new manuscript-specification inputs; local second-layer patches; no first-layer residual transplant'}
    if max_error>5e-5 or class_mismatches or identity_error or table_identity:
        raise RuntimeError('Mechanistic inference calibration failed')
    np.savez_compressed(out/'outputs.npz',**{'channel_'+k:v for k,v in channel.items()},
        **{'crossed_'+k:v for k,v in crossed.items()},**{'query_'+k:v for k,v in qs.items()})
    dump_json(out/'report.json',report)
    print(f"mechanism {Path(checkpoint).parent.name} {c['step']} interaction={interaction:.4f}",flush=True)
    return report


def run(root: Path,out: Path,workers: int=4):
    if not (root/'completed.json').is_file():raise ValueError('Finish the full primary grid first')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);inputs=out/'inputs.npz';make_inputs(inputs)
    checkpoints=sorted(root.glob('models/*/model_*.pt'))
    if len(checkpoints)!=64:raise ValueError('Expected all 64 primary checkpoints')
    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(evaluate_one,str(p),str(inputs),str(out/(p.parent.name+'__'+p.stem))) for p in checkpoints]
        for f in as_completed(fs):reports.append(f.result())
    dump_json(out/'summary.json',reports)
    dump_json(out/'completed.json',{'checkpoints':len(reports),'all_completed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.primary,a.output,a.workers)
