from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    corpus: str = "wikitext103"
    dir: str = "data/wikitext103"
    vocab_size: int = 8192
    block_size: int = 256
    n_synthetic_sentences: int = 60000


@dataclass
class ModelConfig:
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_steps: int = 20000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_batches: int = 20
    device: str = "auto"


@dataclass
class CheckpointConfig:
    dense_until: int = 500
    dense_every: int = 25
    n_log: int = 40
    late_every: int = 2000


@dataclass
class EvalConfig:
    n_agreement_pairs: int = 2000
    out_dir: str = "runs"


@dataclass
class Config:
    name: str = "base"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.eval.out_dir) / f"{self.name}_seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            if f.name in ("name", "seed"):
                kwargs[f.name] = d[f.name]
            else:
                kwargs[f.name] = f.default_factory(**d[f.name])  # type: ignore[misc]
        return cls(**kwargs)


def load_config(path: str | Path, **overrides) -> Config:
    """Load a YAML config. `overrides` may set top-level keys, e.g. seed=1."""
    with open(path) as f:
        d = yaml.safe_load(f)
    d.update({k: v for k, v in overrides.items() if v is not None})
    return Config.from_dict(d)
