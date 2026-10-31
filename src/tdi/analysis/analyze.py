"""python -m tdi.analysis.analyze --runs runs/base_seed0 runs/base_seed1 ... [--out analysis_out]

For each run: onsets of behavior, decodability and causal effect per feature, and the lags between
them. Across runs (seeds): mean and bootstrap interval of each lag, with every seed's value shown.
Writes lag_per_run.csv, lag_summary.csv and a curves figure per run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .onset import bootstrap_ci, onset, run_curves
from .plots import plot_run

FEATURES = ["subj_number@subj", "subj_number@last", "attr_number@attr"]


CAUSAL_MIN_EFFECT = 1.0  # nats of logit difference: smaller effects are treated as "not used"
DECOD_MIN_GAIN = 0.1  # probe accuracy must rise by at least this much over the untrained model


def lags_for_run(run_dir, decod_metric: str = "acc", **kw) -> pd.DataFrame:
    curves = run_curves(run_dir, decod_metric)
    t_beh = onset(curves["behavior:agree_logit_diff"].step, curves["behavior:agree_logit_diff"].value, **kw)
    rows = []
    for f in FEATURES:
        def t(key, **extra):
            c = curves.get(key)
            return float("nan") if c is None else onset(c.step, c.value, **kw, **extra)

        causal = {"min_change": CAUSAL_MIN_EFFECT}
        t_dec = t(f"decodability:{f}", min_change=DECOD_MIN_GAIN)
        t_loc, t_glo = t(f"causal:leace_local:{f}", **causal), t(f"causal:leace_global:{f}", **causal)
        rows.append({
            "run": Path(run_dir).name, "feature": f, "t_beh": t_beh, "t_dec": t_dec,
            "t_use_local": t_loc, "t_use_global": t_glo,
            "lag_dec_to_use": t_loc - t_dec, "lag_dec_to_beh": t_beh - t_dec,
        })
    return pd.DataFrame(rows)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, g in per_run.groupby("feature"):
        for col in ("t_beh", "t_dec", "t_use_local", "lag_dec_to_use", "lag_dec_to_beh"):
            m, lo, hi = bootstrap_ci(g[col].tolist())
            rows.append({"feature": f, "quantity": col, "mean": m, "ci_low": lo, "ci_high": hi,
                         "n_runs": int(g[col].notna().sum()), "per_run": g[col].round(1).tolist()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--decod-metric", default="acc", choices=["acc", "selectivity"])
    ap.add_argument("--frac", type=float, default=0.5, help="onset threshold (fraction of total change)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_run = pd.concat([lags_for_run(r, args.decod_metric, frac=args.frac) for r in args.runs], ignore_index=True)
    per_run.to_csv(out / "lag_per_run.csv", index=False)
    summarize(per_run).to_csv(out / "lag_summary.csv", index=False)
    for r in args.runs:
        plot_run(run_curves(r, args.decod_metric), FEATURES, out / f"curves_{Path(r).name}.png", title=Path(r).name)
    pd.set_option("display.width", 200)
    print(per_run.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
