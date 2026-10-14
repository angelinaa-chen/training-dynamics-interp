from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import CheckpointConfig, Config, ModelConfig
from .model import GPT


def checkpoint_steps(c: CheckpointConfig, max_steps: int) -> list[int]:
    """Dense early, log-spaced through the middle, linear late. Always includes 0 and max_steps."""
    steps = {0, max_steps}
    steps.update(range(c.dense_every, min(c.dense_until, max_steps) + 1, c.dense_every))
    if max_steps > c.dense_until:
        log = np.geomspace(c.dense_until, max_steps, c.n_log)
        steps.update(int(round(s)) for s in log)
    steps.update(range(c.late_every, max_steps + 1, c.late_every))
    return sorted(s for s in steps if 0 <= s <= max_steps)


def ckpt_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:07d}.pt"


def save_checkpoint(run_dir: Path, model: GPT, step: int, cfg: Config) -> Path:
    path = ckpt_path(run_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": step, "model": model.state_dict(), "config": cfg.to_dict(), "vocab_size": model.vocab_size},
        path,
    )
    return path


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    paths = sorted(Path(run_dir, "checkpoints").glob("step_*.pt"))
    return [(int(p.stem.split("_")[1]), p) for p in paths]


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[GPT, Config, int]:
    blob = torch.load(path, map_location=device)
    cfg = Config.from_dict(blob["config"])
    model = GPT(cfg.model, blob["vocab_size"], cfg.data.block_size)
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), cfg, blob["step"]
