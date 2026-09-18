"""Two-layer pre-LN transformer with explicit, patchable Q/K/V.

Width 64, four heads, FF width 128, exact GELU, dropout zero. Causality is
an explicit reconstruction assumption, not a verified upstream code detail.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .data import Task


@dataclass(frozen=True)
class Architecture:
    width: int = 64
    heads: int = 4
    hidden: int = 128
    layers: int = 2
    causal: bool = True


class Block(nn.Module):
    def __init__(self, cfg: Architecture):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.out = nn.Linear(cfg.width, cfg.width)
        self.ln2 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.hidden), nn.GELU(approximate="none"),
                                nn.Linear(cfg.hidden, cfg.width))

    def forward(self, x, capture=False, patch=None):
        B, L, D = x.shape
        H = self.cfg.heads
        q, k, v = self.qkv(self.ln1(x)).reshape(B, L, 3, H, D//H).permute(2, 0, 3, 1, 4)
        if patch:
            q = patch.get("q", q)
            k = patch.get("k", k)
            v = patch.get("v", v)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=self.cfg.causal)
        x = x + self.out(z.transpose(1, 2).reshape(B, L, D))
        x = x + self.ff(self.ln2(x))
        cache = None
        if capture:
            scores = q[:, :, -1:] @ k.transpose(-2, -1) / math.sqrt(D//H)
            cache = {"q": q, "k": k, "v": v, "scores": scores.squeeze(2)}
        return x, cache


class Transformer(nn.Module):
    def __init__(self, task: Task, cfg: Architecture = Architecture()):
        super().__init__()
        if cfg.width % cfg.heads:
            raise ValueError("width must be divisible by heads")
        self.task, self.cfg = task, cfg
        self.token = nn.Embedding(task.vocab, cfg.width)
        self.position = nn.Parameter(torch.empty(task.length, cfg.width))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, task.n)

    def forward(self, tokens, capture=False, patch=None):
        if tokens.shape[1] != self.task.length:
            raise ValueError("Unexpected sequence length")
        x = self.token(tokens) + self.position
        caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, capture, None if patch is None else patch.get(i))
            caches.append(cache)
        logits = self.head(self.norm(x[:, -1]))
        return (logits, caches) if capture else logits

    def metadata(self):
        return {"task": asdict(self.task), "architecture": asdict(self.cfg),
                "parameters": sum(p.numel() for p in self.parameters())}
