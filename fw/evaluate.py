"""Paired evaluation, row-level outcomes, and Q/K/V causal interventions."""
import base64
import json
import zlib
from pathlib import Path
import numpy as np
import torch
from .data import Task, World, draw, encode, equal_function_pair, relevant_change


def sliced(w, start, end):
    return World(**{k: v[start:end] for k, v in vars(w).items()})


@torch.no_grad()
def predict(model, w, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    ps, logits = [], []
    for i in range(0, w.batch, batch_size):
        l = model(encode(sliced(w, i, i+batch_size), model.task).to(device))
        ps.append(l.softmax(-1).cpu().numpy())
        logits.append(l.cpu().numpy())
    return np.concatenate(ps), np.concatenate(logits)


def paired_stats(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Require paired one-dimensional samples of equal length >= 2")
    n = len(a)
    positive = int((~a & b).sum())
    negative = int((a & ~b).sum())
    delta = (positive - negative)/n
    # Conditional-on-model, normal paired sampling interval; not a seed CI.
    se = np.std(b.astype(float)-a.astype(float), ddof=1)/np.sqrt(n)
    return {"n": n, "control_rate": float(a.mean()), "witness_rate": float(b.mean()),
            "effect": float(delta), "paired_se": float(se),
            "paired_normal_95ci": [float(delta-1.96*se), float(delta+1.96*se)],
            "became_wrong": positive, "became_not_wrong": negative,
            "both_wrong": int((a & b).sum()), "neither_wrong": int((~a & ~b).sum())}


def pack_outcomes(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    return {"dtype": "uint8", "shape": list(arr.shape), "encoding": "zlib+base64",
            "data": base64.b64encode(zlib.compress(arr.tobytes(), 9)).decode("ascii")}


def unpack_outcomes(obj):
    if obj["encoding"] != "zlib+base64" or obj["dtype"] != "uint8":
        raise ValueError("Unsupported outcome encoding")
    return np.frombuffer(zlib.decompress(base64.b64decode(obj["data"])),
                         dtype=np.uint8).reshape(obj["shape"])


@torch.no_grad()
def qkv_factorial(model, control, witness, size=512):
    """Donor Q/K/V crossed at the second layer; recipient residual held fixed.

    All positions/heads are patched, not just the witness row. This is a layer
    causal decomposition and MUST NOT be described as localized mediation.
    111 need not equal the full witness run: recipient residual is unchanged.
    """
    n = min(size, control.batch)
    device = next(model.parameters()).device
    w, v = sliced(control, 0, n), sliced(witness, 0, n)
    x0 = encode(w, model.task).to(device)
    x1 = encode(v, model.task).to(device)
    l0, c0 = model(x0, capture=True)
    l1, c1 = model(x1, capture=True)
    b = torch.arange(n, device=device)
    y = w.labels().to(device)
    wrong = w.g[torch.arange(n), w.hint].to(device)
    def margin(l):
        return (l[b, wrong] - l[b, y]).cpu().numpy()
    d = {}
    for qi in range(2):
        for ki in range(2):
            for vi in range(2):
                patch = {model.cfg.layers-1: {
                    "q": (c1 if qi else c0)[-1]["q"],
                    "k": (c1 if ki else c0)[-1]["k"],
                    "v": (c1 if vi else c0)[-1]["v"]}}
                d[f"{qi}{ki}{vi}"] = margin(model(x0, patch=patch))
    error = float(np.max(np.abs(d['000']-margin(l0))))
    selection = d['010']-d['000']
    payload = d['001']-d['000']
    interaction = d['011']-d['010']-d['001']+d['000']
    return {"n": n, "margin_means": {k: float(v.mean()) for k,v in d.items()},
            "full_context_effect": float((margin(l1)-margin(l0)).mean()),
            "fixed_query_key_effect": float(selection.mean()),
            "fixed_query_value_effect": float(payload.mean()),
            "fixed_query_key_value_interaction": float(interaction.mean()),
            "unpatched_residual_remainder": float((margin(l1)-d['111']).mean()),
            "identity_patch_max_error": error,
            "scope": "all second-layer heads/positions; recipient first-layer residual fixed"}


@torch.no_grad()
def evaluate(model, n=4096, counts=(1,2,4), seed=8171, mechanism=False):
    task = model.task
    report = {"evaluation_seed": seed, "n_per_condition": n, "clean": {}, "pairs": {}}
    raw = {}
    # Each clean mode reuses identical i.i.d. tables, query and task.
    rng = torch.Generator().manual_seed(seed)
    clean = draw(task, n, rng)
    clean.two_step.fill_(True)
    idx = torch.arange(n)
    truth = clean.labels().numpy()
    for mode in ('absent', 'correct', 'incorrect'):
        c = clean.copy()
        if mode == 'absent':
            c.present.fill_(False)
        else:
            c.present.fill_(True)
            c.hint = c.intermediate() if mode == 'correct' else (
                c.intermediate()+torch.randint(1,task.n,(n,),generator=rng))%task.n
        p, _ = predict(model,c)
        pred = p.argmax(1)
        report['clean'][mode] = {"accuracy": float((pred==truth).mean()), "n":n}
        if mode == 'incorrect':
            wrong = c.g[idx,c.hint].numpy()
            mask = wrong != truth
            report['clean'][mode]['decisive_n'] = int(mask.sum())
            report['clean'][mode]['decisive_accuracy'] = float((pred[mask]==truth[mask]).mean())
            report['clean'][mode]['decisive_hint_following'] = float((pred[mask]==wrong[mask]).mean())
        raw['clean_'+mode] = pack_outcomes(np.stack((truth, pred)))
    for count in counts:
        # Same data seed across model seeds; independence unit is model seed.
        rng = torch.Generator().manual_seed(seed + 1000*count)
        a, b, rows = equal_function_pair(task,n,rng,count)
        y = a.labels().numpy()
        wrong = a.g[idx,a.hint].numpy()
        for mode in ('incorrect', 'absent', 'correct'):
            a0, b0 = a.copy(), b.copy()
            if mode == 'absent':
                a0.present.fill_(False); b0.present.fill_(False)
            elif mode == 'correct':
                a0.hint = a0.intermediate(); b0.hint = b0.intermediate()
            pa, la = predict(model,a0); pb, lb = predict(model,b0)
            ahat, bhat = pa.argmax(1), pb.argmax(1)
            key=f'm{count}_{mode}'
            r=paired_stats(ahat==wrong,bhat==wrong)
            r.update({"control_accuracy": float((ahat==y).mean()),
                      "witness_accuracy": float((bhat==y).mean()),
                      "mean_wrong_probability_change": float((pb[np.arange(n),wrong]-pa[np.arange(n),wrong]).mean()),
                      "mean_total_variation": float((np.abs(pb-pa).sum(1)/2).mean()),
                      "query_relation_strata": {}})
            for rel in range(task.relations):
                mask = a.relation.numpy()==rel
                r['query_relation_strata'][str(rel)] = paired_stats(ahat[mask]==wrong[mask],bhat[mask]==wrong[mask])
            report['pairs'][key]=r
            raw[key]=pack_outcomes(np.stack((y,wrong,ahat,bhat,a.relation.numpy())))
        if count==1:
            relevant = relevant_change(a, task)
            other = a.copy()
            if task.relations==2:
                other.f[idx,1-other.relation,other.x]=other.hint
            a.present.fill_(False); relevant.present.fill_(False); other.present.fill_(False)
            p0,_=predict(model,a); pr,_=predict(model,relevant); po,_=predict(model,other)
            pred0, predr, predo=p0.argmax(1),pr.argmax(1),po.argmax(1)
            report['sensitivity']={
                "n":n,"control_accuracy":float((pred0==y).mean()),
                "relevant_changed_answer_accuracy":float((predr==relevant.labels().numpy()).mean()),
                "relevant_prediction_change":float((pred0!=predr).mean()),
                "other_relation_prediction_change":float((pred0!=predo).mean()),
                "other_relation_control_applicable":task.relations==2}
            raw['sensitivity']=pack_outcomes(np.stack((y,relevant.labels().numpy(),pred0,predr,predo)))
            if mechanism:
                a.present.fill_(True)
                report['mechanism']=qkv_factorial(model,a,b)
    return report,raw
