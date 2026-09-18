"""Original-format token-balanced pairs with compensation-only factorial controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import Task,balanced_original_pair
from .model import Architecture,Transformer
from .evaluate import predict,paired_stats,pack_outcomes


def run(checkpoint,output,n=4096,seed=418729):
    torch.set_num_threads(1)
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    t=Task(**c['metadata']['task'])
    model=Transformer(t,Architecture(**c['metadata']['architecture']))
    model.load_state_dict(c['state_dict']);model.eval()
    w,v,rows,dummy=balanced_original_pair(t,n,torch.Generator().manual_seed(seed),1)
    fonly=w.copy();fonly.f=v.f.clone()
    gonly=w.copy();gonly.g=v.g.clone()
    y=w.labels().numpy();wrong=w.g[torch.arange(n),w.hint].numpy()
    report={};raw={}
    for mode in ['incorrect','absent','correct']:
        preds={};margins={}
        for key,original in [('control',w),('witness_only',fonly),('compensation_only',gonly),('balanced',v)]:
            a=original.copy()
            if mode=='absent':a.present.fill_(False)
            elif mode=='correct':a.hint=a.intermediate()
            p,l=predict(model,a)
            preds[key]=p.argmax(1)
            margins[key]=l[np.arange(n),wrong]-l[np.arange(n),y]
        r=paired_stats(preds['control']==wrong,preds['balanced']==wrong)
        r['accuracy']={k:float((p==y).mean()) for k,p in preds.items()}
        r['following']={k:float((p==wrong).mean()) for k,p in preds.items()}
        r['factorial_margin']={
            'witness_only_effect':float((margins['witness_only']-margins['control']).mean()),
            'compensation_only_effect':float((margins['compensation_only']-margins['control']).mean()),
            'interaction':float((margins['balanced']-margins['witness_only']-margins['compensation_only']+margins['control']).mean()),
            'total_balanced_effect':float((margins['balanced']-margins['control']).mean())}
        report[mode]=r
        raw[mode]=pack_outcomes(np.stack([y,wrong]+[preds[k] for k in ['control','witness_only','compensation_only','balanced']]))
    report.update({'seed':c['metadata']['seed'],'step':c['step'],'evaluation_seed':seed,
                   'n':n,'scope':'g o f and literal token multiset fixed; g changes only outside image(f)'})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'balanced_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'balanced_outcomes.json').write_text(json.dumps(raw,indent=2)+'\n')
    print(json.dumps({'seed':report['seed'],'step':report['step'],'balanced_effect':report['incorrect']['effect'],
                      'no_hint_control_accuracy':report['absent']['accuracy']['control']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=4096)
    a=p.parse_args();run(a.checkpoint,a.output,a.n)
