import random

import torch

from tdi import synthetic
from tdi.checkpoints import checkpoint_steps
from tdi.config import CheckpointConfig, ModelConfig
from tdi.model import GPT


def tiny() -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(n_layer=2, n_head=4, d_model=32), vocab_size=50, block_size=16).eval()


def test_model_is_causal():
    m = tiny()
    a = torch.randint(0, 50, (1, 10))
    b = a.clone()
    b[0, 7:] = (b[0, 7:] + 1) % 50  # change only future tokens
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)


def test_head_mask_ablation_changes_output_and_identity_does_not():
    m = tiny()
    x = torch.randint(0, 50, (2, 8))
    base, _ = m(x)
    ones = torch.ones(2, 4)
    same, _ = m(x, head_mask=ones)
    assert torch.allclose(base, same, atol=1e-6)
    mask = ones.clone()
    mask[0, 1] = 0
    ablated, _ = m(x, head_mask=mask)
    assert not torch.allclose(base, ablated, atol=1e-4)


def test_resid_hook_and_return_resid():
    m = tiny()
    x = torch.randint(0, 50, (2, 8))
    logits, _, resid = m(x, return_resid=True)
    assert len(resid) == 3 and resid[0].shape == (2, 8, 32)
    zeroed, _ = m(x, resid_hooks={1: lambda r: torch.zeros_like(r)})
    assert not torch.allclose(logits, zeroed, atol=1e-4)


def test_checkpoint_schedule():
    c = CheckpointConfig(dense_until=100, dense_every=25, n_log=10, late_every=500)
    s = checkpoint_steps(c, 2000)
    assert s[0] == 0 and s[-1] == 2000 and s == sorted(set(s))
    assert {25, 50, 75, 100, 500, 1000, 1500}.issubset(s)


def test_synthetic_verb_agrees_with_subject_number():
    lex = synthetic.make_lexicon()
    rng = random.Random(0)
    for _ in range(300):
        ex = synthetic.sample(rng, lex, "train")
        verb = next(v for v in synthetic.VERBS if ex.correct in v)
        assert (verb[0] if ex.subject_number == "sg" else verb[1]) == ex.correct
        assert ex.wrong != ex.correct


def test_attractor_conditions():
    lex = synthetic.make_lexicon()
    rng = random.Random(1)
    for _ in range(100):
        mis = synthetic.sample(rng, lex, "test", "pp_mismatch")
        mat = synthetic.sample(rng, lex, "test", "pp_match")
        assert mis.attractor_number != mis.subject_number
        assert mat.attractor_number == mat.subject_number


def test_lexicon_split_is_disjoint():
    lex = synthetic.make_lexicon()
    assert not set(lex.train) & set(lex.test)
