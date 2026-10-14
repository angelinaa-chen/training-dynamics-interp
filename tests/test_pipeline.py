"""End-to-end smoke test: prepare synthetic data, train a few steps, reload a checkpoint."""
import json

import torch

from tdi.checkpoints import list_checkpoints, load_checkpoint
from tdi.config import Config
from tdi.data import prepare
from tdi.train import run


def test_prepare_train_reload(tmp_path):
    cfg = Config.from_dict(
        {
            "name": "smoke",
            "seed": 0,
            "data": {"corpus": "synthetic", "dir": str(tmp_path / "data"), "vocab_size": 1024,
                     "block_size": 32, "n_synthetic_sentences": 2000},
            "model": {"n_layer": 1, "n_head": 2, "d_model": 32, "dropout": 0.0},
            "train": {"batch_size": 8, "max_steps": 30, "warmup_steps": 5, "eval_batches": 2,
                      "device": "cpu"},
            "checkpoints": {"dense_until": 10, "dense_every": 10, "n_log": 3, "late_every": 30},
            "eval": {"n_agreement_pairs": 60, "out_dir": str(tmp_path / "runs")},
        }
    )
    prepare(cfg)
    run(cfg)

    ckpts = list_checkpoints(cfg.run_dir)
    assert [s for s, _ in ckpts][0] == 0 and [s for s, _ in ckpts][-1] == 30
    rows = [json.loads(l) for l in (cfg.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == len(ckpts)
    assert rows[-1]["val_loss"] < rows[0]["val_loss"]

    model, loaded_cfg, step = load_checkpoint(ckpts[-1][1])
    assert step == 30 and loaded_cfg.model.d_model == 32
    logits, _ = model(torch.randint(0, model.vocab_size, (1, 8)))
    assert logits.shape == (1, 8, model.vocab_size)
