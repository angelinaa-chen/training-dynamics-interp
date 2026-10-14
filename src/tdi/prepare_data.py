"""python -m tdi.prepare_data --config configs/base.yaml

Downloads the corpus if needed (WikiText needs network on first run), trains the tokenizer and
writes token files into `data.dir`. Run once per corpus; all seeds share the result.
"""
import argparse

from .config import load_config
from .data import prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    out = prepare(load_config(args.config))
    print(f"wrote {out}")
    print((out / "meta.json").read_text())


if __name__ == "__main__":
    main()
