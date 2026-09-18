"""Executable training/evaluation entry point with fixed seeds and raw artifacts."""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Task, draw, encode
from .model import Architecture, Transformer
from .evaluate import evaluate
from .theory import theory_checks


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def train(name, task, seed, cfg, out, threads=1, device='cpu'):
    out=Path(out)/name/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=torch.Generator().manual_seed(seed+100000)
    model=Transformer(task).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],
                         weight_decay=cfg['weight_decay'],betas=(.9,.999),eps=1e-8)
    metadata={**model.metadata(),"seed":seed,"config":cfg,
              "python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
              "platform":platform.platform(),"threads":threads,"device":device,
              "source_status":"independent implementation; upstream code/checkpoints unavailable"}
    dump(out/'metadata.json',metadata)
    trace=[]
    start=time.perf_counter()
    for step in range(1,cfg['steps']+1):
        model.train()
        w=draw(task,cfg['batch_size'],rng)
        logits=model(encode(w,task).to(device))
        loss=F.cross_entropy(logits,w.labels().to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%100==0:
            trace.append({"step":step,"loss":float(loss.item()),"gradient_norm":float(grad),
                          "elapsed_seconds":time.perf_counter()-start})
        if step in cfg['checkpoints']:
            report,raw=evaluate(model,cfg['evaluation_pairs'],cfg['witness_counts'],
                                mechanism=(step==cfg['steps']))
            report.update({"seed":seed,"step":step,"name":name})
            dump(out/f'report_{step}.json',report)
            dump(out/f'outcomes_{step}.json',raw)
            torch.save({"state_dict":model.state_dict(),"optimizer":opt.state_dict(),
                        "rng":rng.get_state(),"metadata":metadata,"step":step},out/f'model_{step}.pt')
            dump(out/'training_trace.json',trace)
            summary={"name":name,"seed":seed,"step":step,"loss":float(loss.item()),
                     "clean_no_hint":report['clean']['absent']['accuracy'],
                     "m1_effect":report['pairs']['m1_incorrect']['effect'],
                     "elapsed_seconds":time.perf_counter()-start}
            print(json.dumps(summary),flush=True)
    dump(out/'completed.json',{"elapsed_seconds":time.perf_counter()-start,
                              "checkpoint_sha256":hashlib.sha256((out/f'model_{cfg["steps"]}.pt').read_bytes()).hexdigest()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pilot.json')
    p.add_argument('--task',choices=['original','typed'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',default='outputs')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    train(a.task,Task(**cfg['tasks'][a.task]),a.seed,cfg,a.output,a.threads,a.device)

if __name__=='__main__': main()
