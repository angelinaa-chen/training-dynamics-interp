"""python -m tdi.interventions.run --run runs/synthetic_debug_seed0

For every checkpoint measures how much agreement depends on (a) attention heads, (b) linearly
erasable feature directions at each layer, (c) the residual stream at each patching site. Every
targeted intervention is paired with random controls of matched size, and the effect on held-out
language-modeling loss is recorded so a general breakdown can be told from a specific one.

Writes <run>/causal_results.csv, long format: one row per (step, kind, feature, layer, site).
`effect` is the drop in mean agreement logit difference (nats); `adj_effect` subtracts the mean
random-control drop. Patching rows report the normalized flip fraction in `effect`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import synthetic
from ..checkpoints import list_checkpoints, load_checkpoint
from ..config import Config
from ..data import TokenData, load_tokenizer
from ..encoding import counterfactual_pairs, encode_examples
from ..probing.activations import collect_residuals
from ..probing.features import FEATURES
from ..train import pick_device
from .ablation import head_ablation_effects, head_mask, random_head_sets, top_heads
from .common import heldout_loss, mean_ld
import torch

from .erasure import fit_leace, local_hook, random_eraser
from .patching import SITES, patching_effects

ERASE_FEATURES = ["subj_number@subj", "subj_number@last", "attr_number@attr"]


def _examples(cfg: Config, n: int, seed: int):
    split = "train" if cfg.data.corpus == "synthetic" else "all"
    return synthetic.build_pairs(max(1, n // len(synthetic.CONDITIONS)), seed, synthetic.make_lexicon(), split)


def causal_rows(model, cfg, data, fit_items, eval_items, clean, corrupt, device, *,
                k_heads: int = 2, n_random: int = 6, loss_batches: int = 5) -> list[dict]:
    L, H = cfg.model.n_layer, cfg.model.n_head
    bs, bl = cfg.train.batch_size, cfg.data.block_size
    base_ld = mean_ld(model, eval_items, device)
    base_loss = heldout_loss(model, data, bs, bl, loss_batches, device)
    rows = []

    def row(kind, feature, layer, site, effect, rand_mean, rand_sd, loss_t, loss_r):
        rows.append({"kind": kind, "feature": feature, "layer": layer, "site": site,
                     "effect": effect, "random_mean": rand_mean, "random_sd": rand_sd,
                     "adj_effect": effect - rand_mean, "loss_delta": loss_t,
                     "loss_delta_random": loss_r, "baseline_ld": base_ld})

    # (a) heads: choose the k most important heads on the fit set, evaluate on the eval set
    eff_fit = head_ablation_effects(model, fit_items, device)
    chosen = top_heads(eff_fit, k_heads)
    mask = head_mask(L, H, chosen).to(device)
    t_eff = base_ld - mean_ld(model, eval_items, device, head_mask=mask)
    t_loss = heldout_loss(model, data, bs, bl, loss_batches, device, head_mask=mask) - base_loss
    r_eff, r_loss = [], []
    for k, hs in enumerate(random_head_sets(L, H, k_heads, n_random, seed=k_heads)):
        m = head_mask(L, H, hs).to(device)
        r_eff.append(base_ld - mean_ld(model, eval_items, device, head_mask=m))
        if k < 2:
            r_loss.append(heldout_loss(model, data, bs, bl, loss_batches, device, head_mask=m) - base_loss)
    row("head_ablation", "agreement", -1, f"top{k_heads}", t_eff, float(np.mean(r_eff)),
        float(np.std(r_eff)), t_loss, float(np.mean(r_loss)))

    # (b) linear erasure of each feature, vs random same-rank subspaces. "global" applies the
    # eraser at every token position (and reports the language-modeling side effect); "local"
    # applies it only at the position where the feature is read (the subject noun, the last
    # prefix token, or the attractor noun), which tests whether the information *at that
    # position* is used.
    resid = collect_residuals(model, fit_items, device)
    for name in ERASE_FEATURES:
        s = FEATURES[name](fit_items)
        se = FEATURES[name](eval_items)
        pos_eval = torch.full((len(eval_items),), -1, dtype=torch.long)
        pos_eval[torch.tensor(se.ex)] = torch.tensor(se.pos)
        for layer in range(L + 1):
            X = resid[layer, s.ex, s.pos].numpy()
            er = fit_leace(X, s.y)
            hook = {layer: er.hook()}
            t_eff = base_ld - mean_ld(model, eval_items, device, resid_hooks=hook)
            t_loss = heldout_loss(model, data, bs, bl, loss_batches, device, resid_hooks=hook) - base_loss
            r_eff, r_loss = [], []
            for k in range(n_random):
                rh = {layer: random_eraser(er, seed=k).hook()}
                r_eff.append(base_ld - mean_ld(model, eval_items, device, resid_hooks=rh))
                if k < 2:
                    r_loss.append(heldout_loss(model, data, bs, bl, loss_batches, device, resid_hooks=rh) - base_loss)
            row("leace_global", name, layer, "all", t_eff, float(np.mean(r_eff)), float(np.std(r_eff)),
                t_loss, float(np.mean(r_loss)))

            def local(eraser, layer=layer):
                return lambda a, b: {"resid_hooks": {layer: local_hook(eraser.hook(), pos_eval[a:b])}}

            t_eff = base_ld - mean_ld(model, eval_items, device, per_batch=local(er))
            r_eff = [base_ld - mean_ld(model, eval_items, device, per_batch=local(random_eraser(er, seed=k)))
                     for k in range(n_random)]
            row("leace_local", name, layer, "at_feature", t_eff, float(np.mean(r_eff)),
                float(np.std(r_eff)), float("nan"), float("nan"))

    # (c) activation patching on counterfactual minimal pairs
    eff = patching_effects(model, clean, corrupt, device)
    for layer in range(eff.shape[0]):
        for j, site in enumerate(SITES):
            row("patching", "subj_number", layer, site, float(eff[layer, j]), 0.0, 0.0, 0.0, 0.0)
    return rows


def run(run_dir: str | Path, device: str = "cpu", every: int = 1, n_examples: int = 1200,
        n_random: int = 6, steps: list[int] | None = None) -> pd.DataFrame:
    run_dir = Path(run_dir)
    cfg = Config.from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    dev = pick_device(device)
    tok, data = load_tokenizer(cfg.data.dir), TokenData(cfg.data.dir)
    fit_items = encode_examples(tok, _examples(cfg, n_examples, 777))
    eval_ex = _examples(cfg, n_examples, 888)
    eval_items = encode_examples(tok, eval_ex)
    clean, corrupt = counterfactual_pairs(tok, eval_ex)
    ckpts = list_checkpoints(run_dir)
    ckpts = [c for c in ckpts if c[0] in set(steps)] if steps else ckpts[::every]
    out = []
    for k, (step, path) in enumerate(ckpts):
        model, _, _ = load_checkpoint(path, dev)
        for r in causal_rows(model, cfg, data, fit_items, eval_items, clean, corrupt, dev,
                            n_random=n_random):
            out.append({"step": step, **r})
        if k % 10 == 0:
            print(f"intervened on checkpoint {k + 1}/{len(ckpts)} (step {step})", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(run_dir / "causal_results.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    ap.add_argument("--n-examples", type=int, default=1200,
                    help="examples per set; use more for wider models so the erasure fit is stable")
    ap.add_argument("--n-random", type=int, default=6, help="random controls per intervention")
    ap.add_argument("--steps", type=int, nargs="*", help="only these checkpoint steps")
    args = ap.parse_args()
    df = run(args.run, args.device, args.every, args.n_examples, args.n_random, args.steps)
    print(f"wrote {args.run}/causal_results.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
