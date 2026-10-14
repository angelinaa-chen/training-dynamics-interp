"""Small decoder-only transformer, written to be easy to intervene on.

Interventions used later in the project are built in:
  * `head_mask` [n_layer, n_head] multiplies each head's output (0 = ablate).
  * `resid_hooks` {i: fn(resid) -> resid} edits the residual stream at position i,
    where 0 is the embedding output and i is the output of block i-1.
  * `return_resid` returns the residual stream at every position, for probing.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        z = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )  # [B, H, T, dh]
        if head_mask is not None:
            z = z * head_mask.view(1, -1, 1, 1).to(z.dtype)
        return self.proj(z.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), head_mask))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, block_size: int):
        super().__init__()
        self.cfg, self.vocab_size, self.block_size = cfg, vocab_size, block_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init on residual projections
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_resid: bool = False,
        head_mask: torch.Tensor | None = None,
        resid_hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        hooks = resid_hooks or {}
        resid = []
        if 0 in hooks:
            x = hooks[0](x)
        resid.append(x)
        for i, block in enumerate(self.blocks):
            x = block(x, None if head_mask is None else head_mask[i])
            if i + 1 in hooks:
                x = hooks[i + 1](x)
            resid.append(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_resid:
            return logits, loss, resid
        return logits, loss
