from __future__ import annotations

import numpy as np
import torch

from ..data import TokenData
from ..evals.agreement import logit_diffs
from ..model import GPT


def mean_ld(model: GPT, items, device, per_batch=None, **fwd) -> float:
    """Mean log P(correct verb) - log P(wrong verb) over `items`, optionally under an intervention."""
    return float(logit_diffs(model, items, device, per_batch=per_batch, **fwd).mean())


@torch.no_grad()
def heldout_loss(model: GPT, data: TokenData, batch_size: int, block_size: int, n_batches: int,
                 device, **fwd) -> float:
    """Validation cross-entropy on fixed batches (same ones as train.val_loss), under an intervention."""
    was = model.training
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = data.batch("val", batch_size, block_size, rng)
        _, loss = model(x.to(device), y.to(device), **fwd)
        losses.append(loss.item())
    model.train(was)
    return float(np.mean(losses))
