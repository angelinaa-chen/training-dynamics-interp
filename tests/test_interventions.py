import numpy as np
import torch

from tdi import synthetic
from tdi.config import ModelConfig
from tdi.data import train_tokenizer
from tdi.encoding import counterfactual_pairs, encode_examples
from tdi.interventions.ablation import head_ablation_effects, head_mask, random_head_sets, top_heads
from tdi.interventions.erasure import class_mean_direction, fit_leace, random_eraser, steer_hook
from tdi.interventions.patching import patching_effects
from tdi.model import GPT
from tdi.probing.probes import probe_accuracy


def toy_model_and_items():
    lex = synthetic.make_lexicon()
    tok = train_tokenizer(iter(synthetic.generate_corpus(3000, 1, lex)), 1024)
    exs = synthetic.build_pairs(40, 5, lex, "train")
    torch.manual_seed(0)
    m = GPT(ModelConfig(n_layer=2, n_head=2, d_model=16), vocab_size=1024, block_size=32).eval()
    return tok, m, encode_examples(tok, exs), exs


def test_leace_removes_linear_information_and_random_does_not():
    rng = np.random.default_rng(0)
    n, d = 2000, 12
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))  # correlated features
    X[:, 2] += 2.0 * y
    X[:, 5] -= 1.5 * y
    test = np.arange(n) >= 1200
    before = probe_accuracy(X, y, test)
    er = fit_leace(X[:1200], y[:1200])
    after = probe_accuracy(er(torch.tensor(X, dtype=torch.float32)).numpy(), y, test)
    rand = probe_accuracy(random_eraser(er, 0)(torch.tensor(X, dtype=torch.float32)).numpy(), y, test)
    assert er.rank == 1
    assert before > 0.88 and after < 0.6
    assert rand > after + 0.1  # a random same-rank subspace removes far less than LEACE


def test_leace_leaves_directions_uncorrelated_with_label_alone():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    X = rng.normal(size=(3000, 8))
    X[:, 0] += 3 * y
    er = fit_leace(X, y)
    Xe = er(torch.tensor(X, dtype=torch.float32)).numpy()
    assert np.abs(Xe[y == 1].mean(0) - Xe[y == 0].mean(0)).max() < 0.05
    # an unrelated coordinate is untouched
    assert np.allclose(Xe[:, 4], X[:, 4], atol=0.1)


def test_steering_adds_direction_at_position():
    x = torch.zeros(2, 4, 3)
    out = steer_hook(torch.tensor([1.0, 0, 0]), 2.0, pos=torch.tensor([1, 3]))(x)
    assert out[0, 1, 0] == 2 and out[1, 3, 0] == 2 and out.abs().sum() == 4
    d = class_mean_direction(np.array([[0.0, 0], [2, 0]]), np.array([0, 1]))
    assert np.allclose(d, [1, 0])


def test_head_ablation_and_random_sets():
    _, m, items, _ = toy_model_and_items()
    eff = head_ablation_effects(m, items[:60], "cpu")
    assert eff.shape == (2, 2)
    assert len(top_heads(eff, 3)) == 3
    sets = random_head_sets(2, 2, 2, 5, seed=0)
    assert all(len(set(s)) == 2 for s in sets)
    assert head_mask(2, 2, [(0, 1)])[0, 1] == 0


def test_patching_identity_and_full_patch():
    tok, m, items, exs = toy_model_and_items()
    clean, corrupt = counterfactual_pairs(tok, exs[:80])
    assert len(clean) > 20
    for c, k in zip(clean[:5], corrupt[:5]):
        assert c.example.subject_number != k.example.subject_number
        assert len(c.prefix_ids) == len(k.prefix_ids)
    eff = patching_effects(m, clean, corrupt, "cpu")
    assert eff.shape == (3, 2)
    # Patching the last position in the final residual copies the corrupt run's prediction exactly.
    assert abs(eff[-1, 1] - 1.0) < 1e-3


def test_local_hook_only_touches_selected_positions():
    from tdi.interventions.erasure import local_hook

    x = torch.ones(3, 4, 2)
    out = local_hook(lambda v: v * 0, torch.tensor([1, -1, 3]))(x)
    assert out[0, 1].sum() == 0 and out[2, 3].sum() == 0
    assert out.sum() == x.sum() - 4  # only two positions (2 dims each) zeroed
    assert torch.equal(out[1], x[1])
