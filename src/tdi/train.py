"""python -m tdi.train --config configs/base.yaml --seed 0

Trains one model, saving weights-only checkpoints on the schedule in `checkpoints` and appending
one JSON line of metrics per checkpoint to <run_dir>/metrics.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import yaml

from . import synthetic
from .checkpoints import checkpoint_steps, save_checkpoint
from .config import Config, load_config
from .data import TokenData, load_tokenizer
from .evals.agreement import encode_pairs, logit_diffs, summarize
from .model import GPT


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, t) -> float:
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model: GPT, t) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t.lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def val_loss(model: GPT, data: TokenData, cfg: Config, device) -> float:
    model.eval()
    rng = np.random.default_rng(1234)  # same validation batches at every checkpoint and seed
    losses = []
    for _ in range(cfg.train.eval_batches):
        x, y = data.batch("val", cfg.train.batch_size, cfg.data.block_size, rng)
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def run(cfg: Config) -> None:
    device = pick_device(cfg.train.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    data = TokenData(cfg.data.dir)
    tok = load_tokenizer(cfg.data.dir)
    pairs = encode_pairs(
        tok,
        synthetic.build_pairs(
            cfg.eval.n_agreement_pairs // len(synthetic.CONDITIONS) or 1, seed=0,
            lex=synthetic.make_lexicon(),
            # Held-out nouns are never seen in the synthetic corpus, so the model cannot know their
            # number. Evaluate on nouns it has seen (WikiText models have seen them all).
            split="train" if cfg.data.corpus == "synthetic" else "all",
        ),
    )
    if not pairs:
        raise SystemExit("no agreement pairs are single-token in this tokenizer; increase data.vocab_size")
    print(f"{len(pairs)} agreement pairs usable with this tokenizer")

    model = GPT(cfg.model, data.vocab_size, cfg.data.block_size).to(device)
    print(f"{model.n_params() / 1e6:.2f}M parameters on {device}")
    opt = make_optimizer(model, cfg.train)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")
    ckpt_steps = set(checkpoint_steps(cfg.checkpoints, cfg.train.max_steps))

    def checkpoint(step: int, train_loss: float) -> None:
        save_checkpoint(run_dir, model, step, cfg)
        row = {
            "step": step,
            "tokens": step * cfg.train.batch_size * cfg.data.block_size,
            "train_loss": train_loss,
            "val_loss": val_loss(model, data, cfg, device),
        }
        row["val_ppl"] = math.exp(row["val_loss"])
        row.update(summarize(pairs, logit_diffs(model, pairs, device)))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"step {step:6d} | train {train_loss:.3f} | val {row['val_loss']:.3f} | "
            f"agree {row['agree_acc']:.3f}"
        )

    model.train()
    ema, t0 = float("nan"), time.time()
    if 0 in ckpt_steps:
        checkpoint(0, float("nan"))
    for step in range(1, cfg.train.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, cfg.train)
        x, y = data.batch("train", cfg.train.batch_size, cfg.data.block_size, rng)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        ema = loss.item() if math.isnan(ema) else 0.98 * ema + 0.02 * loss.item()
        if step in ckpt_steps:
            checkpoint(step, ema)
    print(f"done in {time.time() - t0:.0f}s -> {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = load_config(args.config, seed=args.seed)
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.device:
        cfg.train.device = args.device
    run(cfg)


if __name__ == "__main__":
    main()
