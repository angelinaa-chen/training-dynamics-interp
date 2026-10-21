from __future__ import annotations

import numpy as np
import torch

from ..model import GPT
from .common import mean_ld


def head_mask(n_layer: int, n_head: int, off: list[tuple[int, int]]) -> torch.Tensor:
    m = torch.ones(n_layer, n_head)
    for l, h in off:
        m[l, h] = 0.0
    return m


def head_ablation_effects(model: GPT, items, device) -> np.ndarray:
    """[n_layer, n_head] drop in mean agreement logit difference when one head is zeroed."""
    L, H = model.cfg.n_layer, model.cfg.n_head
    base = mean_ld(model, items, device)
    out = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            out[l, h] = base - mean_ld(model, items, device, head_mask=head_mask(L, H, [(l, h)]).to(device))
    return out


def top_heads(effects: np.ndarray, k: int) -> list[tuple[int, int]]:
    flat = np.argsort(effects, axis=None)[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(i, effects.shape)) for i in flat]


def random_head_sets(n_layer: int, n_head: int, k: int, n: int, seed: int) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    allh = [(l, h) for l in range(n_layer) for h in range(n_head)]
    return [[allh[i] for i in rng.choice(len(allh), size=k, replace=False)] for _ in range(n)]
