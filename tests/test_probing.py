import numpy as np
import torch

from tdi import synthetic
from tdi.config import ModelConfig
from tdi.data import train_tokenizer
from tdi.encoding import encode_examples
from tdi.model import GPT
from tdi.probing.activations import collect_residuals
from tdi.probing.features import FEATURES
from tdi.probing.probes import control_labels, probe_accuracy, split_by_group


def small_setup():
    lex = synthetic.make_lexicon()
    texts = synthetic.generate_corpus(3000, 1, lex)
    tok = train_tokenizer(iter(texts), 1024)
    exs = synthetic.build_pairs(40, 5, lex, "train")
    return tok, encode_examples(tok, exs)


def test_word_positions_match_tokens():
    tok, items = small_setup()
    assert len(items) > 200
    for it in items[:50]:
        assert len(it.word_end) == len(it.example.words)
        # the token ending the subject word decodes to the subject (plus leading space)
        subj_tok = tok.decode([it.prefix_ids[it.word_end[1]]]).strip()
        assert subj_tok == it.example.words[1]


def test_features_labels_and_positions():
    _, items = small_setup()
    for name, fn in FEATURES.items():
        s = fn(items)
        assert len(s) > 50 and s.y.min() >= 0 and s.y.max() < s.n_classes
    s = FEATURES["subj_number@subj"](items)
    for k in range(20):
        it = items[s.ex[k]]
        assert s.pos[k] == it.word_end[1]
        assert s.y[k] == (0 if it.example.subject_number == "sg" else 1)
    a = FEATURES["attr_number@attr"](items)
    assert all(items[e].example.attractor_positions for e in a.ex)  # only attractor conditions


def test_group_split_is_disjoint_and_control_is_unlearnable_by_content():
    _, items = small_setup()
    s = FEATURES["subj_number@subj"](items)
    test = split_by_group(s.group, seed=0)
    assert not set(s.group[test]) & set(s.group[~test])
    # A control label is a function of the token only, so identical tokens share a label.
    c = control_labels(s, seed=0)
    for t in np.unique(s.tok)[:10]:
        assert len(set(c[s.tok == t])) == 1


def test_probe_finds_planted_signal_and_not_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    X = rng.normal(size=(600, 16))
    test = np.arange(600) >= 400
    assert probe_accuracy(X, y, test) < 0.65  # no signal
    X[:, 3] += 6 * y
    assert probe_accuracy(X, y, test) > 0.95  # planted signal


def test_collect_residuals_shapes_and_matches_forward():
    _, items = small_setup()
    torch.manual_seed(0)
    m = GPT(ModelConfig(n_layer=2, n_head=2, d_model=16), vocab_size=1024, block_size=32).eval()
    r = collect_residuals(m, items[:10], "cpu")
    assert r.shape[0] == 3 and r.shape[1] == 10 and r.shape[-1] == 16
    it = items[0]
    _, _, resid = m(torch.tensor([it.prefix_ids]), return_resid=True)
    assert torch.allclose(r[2, 0, : len(it.prefix_ids)], resid[2][0], atol=1e-5)
