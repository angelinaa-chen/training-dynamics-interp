"""Subject-verb agreement minimal-pair evaluation.

For each example the model sees `<eos> prefix` and we compare the next-token log-probability of
the correct and incorrect verb form. `logit_diff` > 0 means the model prefers the correct form.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from ..data import EOS_ID
from ..model import GPT
from ..synthetic import Example


@dataclass
class EncodedPair:
    example: Example
    prefix_ids: list[int]
    correct_id: int
    wrong_id: int


def encode_pairs(tok: Tokenizer, examples: list[Example]) -> list[EncodedPair]:
    """Encode examples, dropping any whose verb forms are not single tokens in this tokenizer."""
    out = []
    for ex in examples:
        c, w = tok.encode(" " + ex.correct).ids, tok.encode(" " + ex.wrong).ids
        if len(c) != 1 or len(w) != 1:
            continue
        out.append(EncodedPair(ex, [EOS_ID] + tok.encode(ex.prefix).ids, c[0], w[0]))
    return out


@torch.no_grad()
def logit_diffs(
    model: GPT, pairs: list[EncodedPair], device, batch_size: int = 256, **forward_kwargs
) -> torch.Tensor:
    """Log-prob(correct) - log-prob(wrong) for each pair. `forward_kwargs` pass interventions
    (head_mask, resid_hooks) through to the model."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        lens = torch.tensor([len(p.prefix_ids) for p in chunk])
        idx = torch.zeros(len(chunk), int(lens.max()), dtype=torch.long)
        for j, p in enumerate(chunk):  # right-pad: causal attention keeps earlier positions unaffected
            idx[j, : len(p.prefix_ids)] = torch.tensor(p.prefix_ids)
        logits, _ = model(idx.to(device), **forward_kwargs)
        last = logits[torch.arange(len(chunk)), (lens - 1).to(device)]
        logp = torch.log_softmax(last.float(), dim=-1)
        c = torch.tensor([p.correct_id for p in chunk], device=device)
        w = torch.tensor([p.wrong_id for p in chunk], device=device)
        r = torch.arange(len(chunk), device=device)
        out.append((logp[r, c] - logp[r, w]).cpu())
    model.train(was_training)
    return torch.cat(out)


def summarize(pairs: list[EncodedPair], diffs: torch.Tensor) -> dict[str, float]:
    """Accuracy and mean logit difference, overall and per condition."""
    by_cond: dict[str, list[float]] = defaultdict(list)
    for p, d in zip(pairs, diffs.tolist()):
        by_cond[p.example.condition].append(d)
    res = {"agree_acc": float((diffs > 0).float().mean()), "agree_logit_diff": float(diffs.mean())}
    for cond, ds in by_cond.items():
        t = torch.tensor(ds)
        res[f"acc/{cond}"] = float((t > 0).float().mean())
        res[f"ld/{cond}"] = float(t.mean())
    return res
