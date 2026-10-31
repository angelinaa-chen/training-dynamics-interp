"""Corpus preparation, tokenization and batching.

`prepare(cfg)` writes to `cfg.data.dir`:
    tokenizer.json   byte-level BPE shared by every run on this corpus (id 0 = <eos>)
    train.bin        uint16 token ids
    val.bin          uint16 token ids
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import Config
from . import synthetic

EOS = "<eos>"
EOS_ID = 0


# ---------------------------------------------------------------- tokenizer

def train_tokenizer(texts: Iterator[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    assert tok.token_to_id(EOS) == EOS_ID
    return tok


def load_tokenizer(data_dir: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))


# ---------------------------------------------------------------- corpora

# Pinned commit of the Salesforce/wikitext dataset repo, so every run (and teammate) reads
# byte-identical files. Loading the repo by name with older `datasets` versions picks up every
# parquet file in it (non-raw and WikiText-2 included), so the raw files are read directly.
_WIKITEXT_COMMIT = "b08601e04326c79dfdd32d625aee71d232d685c3"
_WIKITEXT_FILES = {  # config -> {split: number of parquet shards}
    "wikitext-103-raw-v1": {"train": 2, "validation": 1},
    "wikitext-2-raw-v1": {"train": 1, "validation": 1},
}
_WIKITEXT_ROWS = {  # published sizes, used to catch a wrong or partial download
    "wikitext-103-raw-v1": {"train": 1_801_350, "validation": 3_760},
    "wikitext-2-raw-v1": {"train": 36_718, "validation": 3_760},
}


def _wikitext_lines(name: str) -> tuple[list[str], list[str]]:
    from datasets import DownloadConfig, load_dataset  # imported lazily: needs network on first use

    base = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{_WIKITEXT_COMMIT}/{name}"
    files = {
        split: [f"{base}/{split}-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        for split, n in _WIKITEXT_FILES[name].items()
    }
    # The corpus is large and the CDN sometimes stalls: retry and resume instead of failing.
    ds = load_dataset("parquet", data_files=files,
                      download_config=DownloadConfig(max_retries=20, resume_download=True))
    for split, n in _WIKITEXT_ROWS[name].items():
        if len(ds[split]) != n:
            raise RuntimeError(f"{name} {split}: got {len(ds[split])} rows, expected {n}")
    return list(ds["train"]["text"]), list(ds["validation"]["text"])


def _corpus_texts(cfg: Config) -> tuple[list[str], list[str]]:
    c = cfg.data.corpus
    if c == "wikitext103":
        return _wikitext_lines("wikitext-103-raw-v1")
    if c == "wikitext2":
        return _wikitext_lines("wikitext-2-raw-v1")
    if c == "synthetic":
        lex = synthetic.make_lexicon()
        n = cfg.data.n_synthetic_sentences
        train = synthetic.generate_corpus(n, seed=1, lex=lex, split="train")
        val = synthetic.generate_corpus(max(n // 20, 100), seed=2, lex=lex, split="train")
        return train, val
    raise ValueError(f"unknown corpus {c!r}")


def _clean(lines: list[str]) -> list[str]:
    """Drop blank lines and trailing newlines; each remaining line becomes one EOS-delimited unit."""
    return [l.rstrip("\n") for l in lines if l.strip()]


def _encode(tok: Tokenizer, lines: list[str], chunk: int = 20000) -> np.ndarray:
    """Encode lines as `<eos> line <eos> line ...`, so every unit starts after an EOS token.
    The agreement evals prompt the model the same way."""
    ids: list[np.ndarray] = [np.array([EOS_ID], dtype=np.uint16)]
    for i in range(0, len(lines), chunk):
        enc = tok.encode_batch(lines[i : i + chunk])
        ids.append(np.fromiter((t for e in enc for t in (*e.ids, EOS_ID)), dtype=np.uint16))
    return np.concatenate(ids)


def prepare(cfg: Config) -> Path:
    out = Path(cfg.data.dir)
    out.mkdir(parents=True, exist_ok=True)
    train_lines, val_lines = (_clean(x) for x in _corpus_texts(cfg))
    # Tokenizer is trained on (at most) the first 300k training lines, then frozen.
    tok = train_tokenizer(iter(train_lines[:300_000]), cfg.data.vocab_size)
    tok.save(str(out / "tokenizer.json"))
    train_ids, val_ids = _encode(tok, train_lines), _encode(tok, val_lines)
    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "corpus": cfg.data.corpus,
                "vocab_size": tok.get_vocab_size(),
                "train_tokens": int(len(train_ids)),
                "val_tokens": int(len(val_ids)),
            },
            indent=2,
        )
    )
    return out


# ---------------------------------------------------------------- batching

class TokenData:
    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.splits = {s: np.memmap(d / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
        self.meta = json.loads((d / "meta.json").read_text())
        self.vocab_size: int = self.meta["vocab_size"]

    def batch(
        self, split: str, batch_size: int, block_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[s : s + block_size] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
