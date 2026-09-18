"""Replay our diagnostic artifacts; this is NOT validation against author data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import Setting,Model,canonical_numerics,decode,logits_for,sha256,state_hash,dump_json


def run(root: str, output: str, training_seed: int=10):
    root=Path(root);canonical_numerics(1)
    hashes=json.loads((root/'artifact_hashes.json').read_text())
    for name,digest in hashes.items():
        if sha256(root/name)!=digest:raise ValueError(f'Artifact changed: {name}')
    with np.load(root/'evaluation_inputs.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    rows=[]
    for path in sorted(root.glob('seed_*/model_2000.pt')):
        c=torch.load(path,map_location='cpu',weights_only=True);s=Setting(**c['metadata']['setting'])
        model=Model(s);model.load_state_dict(c['state_dict']);model.eval()
        if state_hash(model.state_dict())!=c['metadata']['final_state_sha256']:raise ValueError('State hash mismatch')
        with np.load(path.parent/'predictions_2000.npz',allow_pickle=False) as f:ref={k:f[k] for k in f.files}
        maximum=0.;mismatches=0
        for key,expected in ref.items():
            actual=logits_for(model,data[key.removesuffix('_logits')])
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            mismatches+=int(np.count_nonzero(actual.argmax(1)!=expected.argmax(1)))
        model=model.double();x=data['pair_control'][:64]
        a=logits_for(model,x);b=logits_for(model,x,manual=True)
        manual_error=float(np.max(np.abs(a-b)))
        if maximum!=0 or mismatches or manual_error>1e-12:raise ValueError('Diagnostic replay discrepancy')
        rows.append({'seed':c['metadata']['seed'],'replayed_classes':sum(len(v) for v in ref.values()),
                     'maximum_logit_error':maximum,'class_mismatches':mismatches,
                     'native_vs_manual_float64_maximum_error':manual_error})
    if len(rows)!=8:raise ValueError('Require all eight executed diagnostic models')
    folder=root/f'seed_{training_seed}'
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(folder/'model_2000.pt',map_location='cpu',weights_only=True)
    s=Setting(**initial['metadata']['setting']);model=Model(s);model.load_state_dict(initial['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    with np.load(folder/'training_stream.npz',allow_pickle=False) as f:
        xs=f['tokens'];ys=f['labels']
    if xs.shape!=(2000,128,19) or ys.shape!=(2000,128):raise ValueError('Incomplete training stream')
    model.train()
    for x,y in zip(xs,ys):
        x=torch.tensor(x,dtype=torch.long);y=torch.tensor(y,dtype=torch.long)
        if not torch.equal(decode(x,s).labels(),y):raise ValueError('Training label differs from true table target')
        loss=F.cross_entropy(model(x),y);opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    training_match=state_hash(model.state_dict())==final['metadata']['final_state_sha256']
    if not training_match:raise ValueError('Training replay final weights do not match')
    report={'status':'own_diagnostic_replay_passed_NOT_author_replication','models':rows,
            'hash_checked_artifacts':len(hashes),'training_seed':training_seed,'training_updates_replayed':2000,
            'training_examples_replayed':256000,'final_weights_bitwise_identical':training_match,
            'extensions_allowed':False}
    dump_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.output),indent=2))
