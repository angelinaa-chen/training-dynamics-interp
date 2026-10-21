from __future__ import annotations

import torch

from ..encoding import EncodedExample
from ..model import GPT


@torch.no_grad()
def collect_residuals(
    model: GPT, items: list[EncodedExample], device, batch_size: int = 256
) -> torch.Tensor:
    """Residual stream at every position: float32 tensor [n_layer + 1, N, T_max, d_model] on CPU.

    Index 0 is the embedding output, index i the output of block i-1. Sequences are right-padded;
    padded positions hold garbage and are never indexed by the probing code."""
    was_training = model.training
    model.eval()
    t_max = max(len(it.prefix_ids) for it in items)
    out = None
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        idx = torch.zeros(len(chunk), t_max, dtype=torch.long)
        for j, it in enumerate(chunk):
            idx[j, : len(it.prefix_ids)] = torch.tensor(it.prefix_ids)
        _, _, resid = model(idx.to(device), return_resid=True)
        stacked = torch.stack([r.float().cpu() for r in resid])  # [L+1, B, T, D]
        if out is None:
            out = torch.empty(stacked.shape[0], len(items), t_max, stacked.shape[-1])
        out[:, i : i + len(chunk)] = stacked
    model.train(was_training)
    return out
