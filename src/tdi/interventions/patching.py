"""Activation patching on counterfactual minimal pairs.

Run the corrupted sentence (subject number flipped), then re-run the clean sentence with the
residual stream at one (layer, position) replaced by the corrupted run's. The normalized effect
    (LD_patched - LD_clean) / (LD_corrupt - LD_clean)
is 1 when the patch flips the model's preference completely (the site fully carries the subject's
number) and 0 when it changes nothing (LD = log P(clean-correct verb) - log P(clean-wrong verb)).
"""
from __future__ import annotations

import numpy as np
import torch

from ..encoding import pad_batch
from ..model import GPT

SITES = ("subject", "last")


@torch.no_grad()
def patching_effects(model: GPT, clean, corrupt, device, batch_size: int = 256) -> np.ndarray:
    """Array [n_layer + 1, 2] (rows: residual index, columns: SITES) of normalized effects."""
    was = model.training
    model.eval()
    n_pos = model.cfg.n_layer + 1
    ld_clean, ld_corrupt = [], []
    ld_patched = {(l, s): [] for l in range(n_pos) for s in SITES}
    for i in range(0, len(clean), batch_size):
        c, k = clean[i : i + batch_size], corrupt[i : i + batch_size]
        idx_c, lens = pad_batch(c)
        idx_k, _ = pad_batch(k)
        idx_c, idx_k = idx_c.to(device), idx_k.to(device)
        r = torch.arange(len(c), device=device)
        cid = torch.tensor([it.correct_id for it in c], device=device)
        wid = torch.tensor([it.wrong_id for it in c], device=device)
        pos = {"subject": torch.tensor([it.word_end[1] for it in c], device=device),
               "last": (lens - 1).to(device)}

        def ld(logits):
            last = torch.log_softmax(logits[r, (lens - 1).to(device)].float(), -1)
            return (last[r, cid] - last[r, wid]).cpu()

        lg_k, _, cache = model(idx_k, return_resid=True)
        ld_corrupt.append(ld(lg_k))
        lg_c, _ = model(idx_c)
        ld_clean.append(ld(lg_c))
        for l in range(n_pos):
            for s in SITES:
                p = pos[s]

                def hook(x, l=l, p=p):
                    out = x.clone()
                    out[r, p] = cache[l][r, p]
                    return out

                lg, _ = model(idx_c, resid_hooks={l: hook})
                ld_patched[(l, s)].append(ld(lg))
    model.train(was)
    base, cor = torch.cat(ld_clean).mean(), torch.cat(ld_corrupt).mean()
    denom = float(cor - base)
    out = np.zeros((n_pos, len(SITES)))
    for l in range(n_pos):
        for j, s in enumerate(SITES):
            out[l, j] = float(torch.cat(ld_patched[(l, s)]).mean() - base) / denom if abs(denom) > 1e-6 else 0.0
    return out
