"""Manuscript-specified architecture/data, with explicit reconstruction boundaries.

Sources: false_witnesses_v2.pdf, Appendices C.1--C.4, D.2, H.
This module matches stated settings. The author's initialization draw order,
GPU training streams and evaluation arrays are NOT recovered by choosing a seed.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import platform
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Setting:
    n: int = 8
    width: int = 64
    heads: int = 4
    hidden: int = 128
    layers: int = 2
    two_step_probability: float = 0.75
    hint_probability: float = 0.5
    hint_reliability: float = 0.75
    order: str = "company_first"
    layout: str = "suggestion_last"
    access: str = "usual"

    def __post_init__(self):
        if self.n < 4 or self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("Invalid dimensions")
        if self.layers != 2:
            raise ValueError("This baseline implements the stated two-layer setting")
        if self.order not in ("company_first", "employee_first"):
            raise ValueError("Invalid order")
        if self.layout not in ("suggestion_last", "person_last", "fixed_suggestion", "fixed_answer"):
            raise ValueError("Invalid layout")
        if self.access not in ("usual", "open", "closed"):
            raise ValueError("Invalid access")
        if self.access == "open" and self.order != "employee_first":
            raise ValueError("Opening comparison uses employee-first order")
        if self.access == "closed" and self.order != "company_first":
            raise ValueError("Closing comparison uses company-first order")
        for p in (self.two_step_probability, self.hint_probability, self.hint_reliability):
            if not 0 <= p <= 1:
                raise ValueError("Probabilities must lie in [0,1]")

    @property
    def length(self) -> int:
        return 2 * self.n + 3 + int(self.layout == "fixed_answer")

    @property
    def vocab(self) -> int:
        return self.n + 4 + int(self.layout == "fixed_answer")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def canonical_numerics(threads: int = 1) -> dict[str, Any]:
    """Disable encoder fast path and TF32, including in evaluation."""
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "activation": "exact GELU", "threads": threads,
        "author_environment": "PyTorch 2.6.0 / NVIDIA RTX A5000 (Appendices C,H)",
    }


def masks(s: Setting) -> tuple[torch.Tensor, torch.Tensor]:
    """nn.Transformer boolean mask: True blocks attention."""
    ordinary = torch.ones(s.length, s.length, dtype=torch.bool).triu(1)
    first = ordinary.clone()
    n = s.n
    fpos, gpos = (slice(n, 2*n), slice(0, n)) if s.order == "company_first" else (slice(0,n),slice(n,2*n))
    if s.access == "open":
        first[fpos, gpos] = False
    elif s.access == "closed":
        first[fpos, gpos] = True
    return first, ordinary


class Model(nn.Module):
    """Native PyTorch initialization, cloned layers with independent storage.

    Constructor order is a recorded reconstruction choice, not an assertion of
    equality to the unavailable author constructor. Native MHA initialization
    is important: an ordinary Linear QKV has DIFFERENT defaults.
    """
    def __init__(self, setting: Setting = Setting()):
        super().__init__()
        self.setting = setting
        s = setting
        self.token = nn.Embedding(s.vocab, s.width)
        self.position = nn.Parameter(torch.empty(s.length, s.width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(s.width, s.heads, s.hidden,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            bias=True, layer_norm_eps=1e-5)
        self.encoder = nn.TransformerEncoder(layer, num_layers=s.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(s.width)
        self.head = nn.Linear(s.width, s.n)
        m0, m1 = masks(s)
        self.register_buffer("first_mask", m0, persistent=False)
        self.register_buffer("last_mask", m1, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.setting.length:
            raise ValueError("Unexpected token shape")
        x = self.token(tokens) + self.position
        # Explicit layer calls permit the first-layer-only access intervention.
        x = self.encoder.layers[0](x, src_mask=self.first_mask, is_causal=False)
        x = self.encoder.layers[1](x, src_mask=self.last_mask, is_causal=False)
        return self.head(self.norm(x[:, -1]))

    def manual(self, tokens: torch.Tensor) -> torch.Tensor:
        """Independent explicit QKV path for numerical forward verification."""
        x = self.token(tokens) + self.position
        b, length, width = x.shape
        heads = self.setting.heads
        for i, layer in enumerate(self.encoder.layers):
            z = layer.norm1(x)
            q,k,v = F.linear(z,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,-1)
            def split(a):
                return a.reshape(b,length,heads,width//heads).transpose(1,2)
            q,k,v = map(split,(q,k,v))
            score = q @ k.transpose(-1,-2) / (width//heads)**0.5
            score = score.masked_fill(self.first_mask if i==0 else self.last_mask, float("-inf"))
            z = (score.softmax(-1) @ v).transpose(1,2).reshape(b,length,width)
            x = x + layer.self_attn.out_proj(z)
            x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x)), approximate="none"))
        return self.head(self.norm(x[:, -1]))


@dataclass
class World:
    f: torch.Tensor
    g: torch.Tensor
    x: torch.Tensor
    two: torch.Tensor
    hint: torch.Tensor
    present: torch.Tensor

    def copy(self) -> "World":
        return World(**{k:v.clone() for k,v in vars(self).items()})

    def labels(self) -> torch.Tensor:
        i = torch.arange(len(self.x), device=self.x.device)
        middle = self.f[i,self.x]
        return torch.where(self.two,self.g[i,middle],middle)

    def answers(self) -> torch.Tensor:
        return self.g.gather(1,self.f)


def draw(s: Setting, batch: int, rng: torch.Generator) -> World:
    """Stated distribution. Order of random calls is explicitly reconstructed."""
    if batch < 1:
        raise ValueError("batch must be positive")
    device = rng.device
    f = torch.randint(s.n,(batch,s.n),generator=rng,device=device)
    g = torch.randint(s.n,(batch,s.n),generator=rng,device=device)
    x = torch.randint(s.n,(batch,),generator=rng,device=device)
    two = torch.rand(batch,generator=rng,device=device) < s.two_step_probability
    present = torch.rand(batch,generator=rng,device=device) < s.hint_probability
    valid = torch.rand(batch,generator=rng,device=device) < s.hint_reliability
    offset = torch.randint(1,s.n,(batch,),generator=rng,device=device)
    middle = f[torch.arange(batch,device=device),x]
    hint = torch.where(valid,middle,(middle+offset)%s.n)
    return World(f,g,x,two,hint,present)


def encode(w: World, s: Setting) -> torch.Tensor:
    a = torch.full_like(w.x,s.n)
    h = torch.where(w.present,w.hint,a)
    t = s.n+2+w.two.long()
    if s.layout == "person_last":
        suffix = torch.stack((h,t,w.x),1)
    else:
        suffix = torch.stack((w.x,t,h),1)
        if s.layout == "suggestion_last":
            suffix[~w.present,0]=s.n
            suffix[~w.present,2]=w.x[~w.present]
    if s.layout == "fixed_answer":
        suffix = torch.cat((suffix,torch.full_like(a[:,None],s.n+4)),1)
    tables = (w.g,w.f) if s.order == "company_first" else (w.f,w.g)
    return torch.cat((*tables,suffix),1)


def decode(tokens: np.ndarray | torch.Tensor, s: Setting) -> World:
    a = torch.as_tensor(tokens,dtype=torch.long).cpu()
    if a.ndim!=2 or a.shape[1]!=s.length or ((a<0)|(a>=s.vocab)).any():
        raise ValueError("Invalid encoded input")
    n=s.n
    g,f = (a[:,:n],a[:,n:2*n]) if s.order=="company_first" else (a[:,n:2*n],a[:,:n])
    suffix=a[:,2*n:]
    two=suffix[:,1] == n+3
    if not ((suffix[:,1]==n+2)|two).all() or (f>=n).any() or (g>=n).any():
        raise ValueError("Invalid table/task tokens")
    if s.layout=="person_last":
        present=suffix[:,0]!=n; x=suffix[:,2]; h=suffix[:,0].clone()
    elif s.layout=="suggestion_last":
        present=suffix[:,0]!=n
        x=torch.where(present,suffix[:,0],suffix[:,2]);h=suffix[:,2].clone()
    else:
        present=suffix[:,2]!=n;x=suffix[:,0];h=suffix[:,2].clone()
        if s.layout=="fixed_answer" and not (suffix[:,3]==n+4).all():
            raise ValueError("Invalid answer token")
    h[~present]=0
    if (x>=n).any() or (h>=n).any():
        raise ValueError("Invalid query/hint")
    w=World(f.clone(),g.clone(),x.clone(),two.clone(),h,present.clone())
    if not torch.equal(encode(w,s),a):
        raise ValueError("Encoding is not canonical for this layout")
    return w


def equal_pairs(s: Setting, size: int=4096, seed: int=20260919) -> tuple[World,World,World,np.ndarray]:
    """Appendix D.2 restrictions, NOT the author's unrecovered input array.

    A new NumPy PCG64 stream is used for diagnostics. Base f uses NEITHER r nor h.
    Rejection enforces distinct true and suggested countries. All arrays retained.
    """
    if size<1:
        raise ValueError("size must be positive")
    rng=np.random.default_rng(seed)
    rows=[]
    while len(rows)<size:
        h,r=rng.choice(s.n,2,replace=False)
        pool=np.setdiff1d(np.arange(s.n),[h,r])
        f=rng.choice(pool,s.n)
        g=rng.integers(s.n,size=s.n)
        g[r]=g[h]
        x,j=rng.choice(s.n,2,replace=False)
        if g[f[x]]==g[h]:
            continue
        rows.append((f,g,x,j,h,r))
    f=torch.tensor(np.stack([row[0] for row in rows]))
    g=torch.tensor(np.stack([row[1] for row in rows]))
    x=torch.tensor([row[2] for row in rows])
    j=np.array([row[3] for row in rows],dtype=np.int64)
    h=torch.tensor([row[4] for row in rows])
    r=torch.tensor([row[5] for row in rows])
    base=World(f,g,x,torch.ones(size,dtype=torch.bool),h,torch.ones(size,dtype=torch.bool))
    a,b=base.copy(),base.copy()
    a.f[torch.arange(size),j]=r
    b.f[torch.arange(size),j]=h
    validate_equal_pair(encode(a,s).numpy(),encode(b,s).numpy(),s,require_d2=True)
    return base,a,b,j


def validate_equal_pair(control: np.ndarray, witness: np.ndarray, s: Setting, require_d2: bool=True) -> None:
    a,b=decode(control,s),decode(witness,s)
    if len(a.x)!=len(b.x) or not a.two.all() or not b.two.all() or not a.present.all() or not b.present.all():
        raise ValueError("Expected paired two-step, hint-present inputs")
    for key in ("g","x","hint","two","present"):
        if not torch.equal(getattr(a,key),getattr(b,key)):
            raise ValueError(f"D.2 requires unchanged {key}")
    changed=a.f!=b.f
    if not (changed.sum(1)==1).all():
        raise ValueError("D.2 requires exactly one employee edit")
    j=changed.long().argmax(1);i=torch.arange(len(a.x))
    if (j==a.x).any() or not torch.equal(a.answers(),b.answers()):
        raise ValueError("Equal-function invariant failed")
    r=a.f[i,j]
    if not torch.equal(b.f[i,j],a.hint) or (a.labels()==a.g[i,a.hint]).any():
        raise ValueError("Wrong-witness/decisive-answer invariant failed")
    if require_d2:
        other=~changed
        if (((a.f==a.hint[:,None])|(a.f==r[:,None]))&other).any():
            raise ValueError("D.2 excludes BOTH r and h from every other row")


@torch.no_grad()
def logits_for(model: Model, tokens: np.ndarray | torch.Tensor, batch_size: int=256, manual: bool=False) -> np.ndarray:
    if batch_size<1:
        raise ValueError("batch_size must be positive")
    device=next(model.parameters()).device
    x=torch.as_tensor(tokens,dtype=torch.long)
    model.eval()
    out=[]
    for start in range(0,len(x),batch_size):
        xb=x[start:start+batch_size].to(device)
        out.append((model.manual(xb) if manual else model(xb)).cpu().numpy())
    if not out:
        raise ValueError("Empty evaluation")
    return np.concatenate(out)


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(state: dict[str,torch.Tensor]) -> str:
    h=hashlib.sha256()
    for key in sorted(state):
        a=state[key].detach().cpu().contiguous()
        h.update(key.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def dump_json(path: str | Path, data: Any) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
