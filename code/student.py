"""Student model: baseline GPT generalised to grouped-query / multi-query attention.

Key mechanism (MP1, feature 001-param-realloc-gqa)
--------------------------------------------------
Multi-query attention (MQA) shares a single key/value projection across all query
heads instead of one per head.  For width `d` and `h` query heads this saves
`2*d*d*(1 - kv_heads/h) + 2*d*(1 - kv_heads/h)` parameters per block, which is
re-invested into additional depth at a roughly constant parameter budget.

Note this compresses the key/value *projections* only: keys and values are broadcast
back to the query-head count before scoring, so the quadratic attention core keeps
its size and total multiply-adds still grow with depth -- about +8% for the
configurations used here (see research.md R2 for the full accounting).

`kv_heads == heads` reproduces standard multi-head attention exactly, which is what
the baseline uses; `kv_heads == 1` is MQA.  The equivalence of the `kv_heads == heads`
path with the classroom baseline is guarded by `tests/test_equivalence.py`.

Contract (see specs/001-param-realloc-gqa/contracts/model-interface.md)
----------------------------------------------------------------------
- `build_model(config)` returns a module with `.context == 256`.
- `forward(ids)` -> unnormalised logits `[batch, time, 2048]`, causal.
- `predict_log_probs(ids)` -> finite, normalised natural-log probabilities,
  causal, with no state carried across calls.
- No future tokens, no cross-window state, no cached targets.

This module is deliberately self-contained: it does not import `model.py`, so the
submission bundle only needs this file.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Causal self-attention with a configurable number of key/value heads."""

    def __init__(self, width, heads, kv_heads):
        super().__init__()
        if width % heads != 0:
            raise ValueError('width must be divisible by heads')
        if not 1 <= kv_heads <= heads:
            raise ValueError('kv_heads must satisfy 1 <= kv_heads <= heads')
        if heads % kv_heads != 0:
            # Without this, heads=4, kv_heads=3 would broadcast to only 3 heads and
            # fail later with an opaque shape error inside scaled_dot_product_attention.
            raise ValueError('heads must be divisible by kv_heads')
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.q = nn.Linear(width, heads * self.head_dim)
        self.k = nn.Linear(width, kv_heads * self.head_dim)
        self.v = nn.Linear(width, kv_heads * self.head_dim)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_heads != self.heads:
            # Broadcast each shared key/value head to the query heads that use it.
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        # Each position attends only to itself and earlier input tokens.  The
        # expansion above happens before scoring, so nothing beyond the causal
        # mask can be observed.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.proj(attended)


class Block(nn.Module):
    def __init__(self, width, heads, kv_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads, kv_heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        # Missing kv_heads means standard multi-head attention, which keeps the
        # supplied configs/baseline.json valid for this implementation.
        kv_heads = config.get('kv_heads', heads)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            [Block(width, heads, kv_heads) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalised next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalised log probabilities, with no access to targets.

        A prediction at position t uses ids[:, :t+1] and nothing later.  This model
        holds no temporary state, so every call starts fresh by construction.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)