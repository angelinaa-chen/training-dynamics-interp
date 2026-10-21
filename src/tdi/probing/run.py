"""python -m tdi.probing.run --run runs/synthetic_debug_seed0

Trains linear probes for every feature, layer and checkpoint of a run and writes
<run>/probe_results.csv with real and control-task accuracy (selectivity = real - control).
Step 0 is the untrained model, so it doubles as the untrained-model baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .. import synthetic
from ..checkpoints import list_checkpoints, load_checkpoint
from ..config import Config
from ..data import load_tokenizer
from ..encoding import encode_examples
from ..train import pick_device
from .activations import collect_residuals
from .features import FEATURES
from .probes import control_labels, probe_accuracy, split_by_group


def build_probe_items(cfg: Config, n_examples: int, seed: int = 777):
    tok = load_tokenizer(cfg.data.dir)
    lex = synthetic.make_lexicon()
    split = "train" if cfg.data.corpus == "synthetic" else "all"
    exs = synthetic.build_pairs(max(1, n_examples // len(synthetic.CONDITIONS)), seed, lex, split)
    return encode_examples(tok, exs)


def probe_model(model, items, samples: dict, splits: dict, controls: dict, device) -> list[dict]:
    resid = collect_residuals(model, items, device)
    rows = []
    for name, s in samples.items():
        for layer in range(resid.shape[0]):
            X = resid[layer, s.ex, s.pos].numpy()
            acc = probe_accuracy(X, s.y, splits[name])
            ctl = probe_accuracy(X, controls[name], splits[name])
            majority = float(np.bincount(s.y[splits[name]]).max() / splits[name].sum())
            rows.append(
                {"feature": name, "layer": layer, "acc": acc, "control_acc": ctl,
                 "selectivity": acc - ctl, "majority": majority}
            )
    return rows


def run(run_dir: str | Path, n_examples: int = 1800, seed: int = 0, device: str = "cpu",
        features: list[str] | None = None, every: int = 1) -> pd.DataFrame:
    run_dir = Path(run_dir)
    cfg = Config.from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    dev = pick_device(device)
    items = build_probe_items(cfg, n_examples)
    samples = {n: FEATURES[n](items) for n in (features or FEATURES)}
    splits = {n: split_by_group(s.group, seed) for n, s in samples.items()}
    controls = {n: control_labels(s, seed) for n, s in samples.items()}
    ckpts = list_checkpoints(run_dir)[::every]
    all_rows = []
    for k, (step, path) in enumerate(ckpts):
        model, _, _ = load_checkpoint(path, dev)
        for r in probe_model(model, items, samples, splits, controls, dev):
            all_rows.append({"step": step, **r})
        if k % 10 == 0:
            print(f"probed checkpoint {k + 1}/{len(ckpts)} (step {step})", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "probe_results.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-examples", type=int, default=1800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=1, help="probe every k-th checkpoint")
    args = ap.parse_args()
    df = run(args.run, args.n_examples, device=args.device, every=args.every)
    print(f"wrote {args.run}/probe_results.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
