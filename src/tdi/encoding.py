"""Tokenize synthetic examples, keeping track of which token ends each word."""
from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer

from .data import EOS_ID
from .synthetic import Example


@dataclass
class EncodedExample:
    example: Example
    prefix_ids: list[int]  # [<eos>] + prefix tokens
    word_end: list[int]  # index into prefix_ids of the last token of each word
    correct_id: int
    wrong_id: int


def encode_examples(tok: Tokenizer, examples: list[Example]) -> list[EncodedExample]:
    """Encode examples. Examples whose verb forms are not single tokens are dropped, and so are
    any whose words would tokenize differently when joined (checked once per example)."""
    out = []
    for ex in examples:
        c, w = tok.encode(" " + ex.correct).ids, tok.encode(" " + ex.wrong).ids
        if len(c) != 1 or len(w) != 1:
            continue
        ids, ends = [EOS_ID], []
        for i, word in enumerate(ex.words):
            ids += tok.encode(word if i == 0 else " " + word).ids
            ends.append(len(ids) - 1)
        if ids[1:] != tok.encode(ex.prefix).ids:  # byte-level BPE splits at spaces; verify anyway
            continue
        out.append(EncodedExample(ex, ids, ends, c[0], w[0]))
    return out


def pad_batch(items: list[EncodedExample]):
    """Right-pad to a [B, T] LongTensor; also returns the true lengths. With causal attention,
    padding on the right never changes the activations at earlier positions."""
    import torch

    lens = torch.tensor([len(it.prefix_ids) for it in items])
    idx = torch.zeros(len(items), int(lens.max()), dtype=torch.long)
    for j, it in enumerate(items):
        idx[j, : len(it.prefix_ids)] = torch.tensor(it.prefix_ids)
    return idx, lens


def counterfactual_pairs(
    tok: Tokenizer, examples: list[Example]
) -> tuple[list[EncodedExample], list[EncodedExample]]:
    """(clean, corrupt) lists, index-aligned: corrupt flips the subject's number and nothing else.
    Pairs are kept only if both encode and have the same token length (so positions line up)."""
    clean, corrupt = [], []
    for ex in examples:
        a = encode_examples(tok, [ex])
        b = encode_examples(tok, [ex.flipped()])
        if a and b and len(a[0].prefix_ids) == len(b[0].prefix_ids):
            clean.append(a[0])
            corrupt.append(b[0])
    return clean, corrupt
