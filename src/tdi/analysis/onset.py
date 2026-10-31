"""Onset detection and lag estimation.

An onset is the first step at which a curve has covered `frac` (default 50%) of its total change
(final value minus initial value) and stays at least that far along for the next `hold` points.
The same definition is used for behavior, decodability and causal effect, and is fixed before
looking at results. Thresholds are swept in `onset_sweep` to show how sensitive the lag is to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def onset(steps, values, frac: float = 0.5, hold: int = 3, init_k: int = 3, final_k: int = 3,
          min_change: float = 0.0) -> float:
    """Step of onset, or nan if the curve never changes by a meaningful amount.

    `init` and `final` are medians of the first/last `k` points, which keeps them robust to noise.
    `min_change` is a floor on |final - init|: a curve that never moves that far has no onset
    (nan), so noise around zero is not reported as an emergence time.
    Works for increasing and decreasing curves. If the threshold is crossed but not held for `hold`
    further points the crossing is ignored and the search continues."""
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    order = np.argsort(steps)
    steps, values = steps[order], values[order]
    ok = ~np.isnan(values)
    steps, values = steps[ok], values[ok]
    if len(values) < init_k + final_k:
        return float("nan")
    init, final = np.median(values[:init_k]), np.median(values[-final_k:])
    total = final - init
    if abs(total) < max(1e-12, min_change):
        return float("nan")
    progress = (values - init) / total  # 0 at the start, 1 at the end, whatever the sign
    reached = progress >= frac
    for i in range(len(values)):
        if reached[i] and reached[i : i + hold + 1].all():
            # interpolate linearly between the last point below threshold and this one
            if i == 0:
                return float(steps[0])
            p0, p1 = progress[i - 1], progress[i]
            if p1 == p0:
                return float(steps[i])
            w = (frac - p0) / (p1 - p0)
            return float(steps[i - 1] + np.clip(w, 0, 1) * (steps[i] - steps[i - 1]))
    return float("nan")


def onset_sweep(steps, values, fracs=(0.25, 0.5, 0.75), **kw) -> dict[float, float]:
    return {f: onset(steps, values, frac=f, **kw) for f in fracs}


def lag(t_dec: float, t_use: float) -> float:
    return t_use - t_dec


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval, ignoring nans. With very few values (for example 3
    seeds) the interval is wide and the individual points should be shown alongside it."""
    v = np.asarray([x for x in values if not np.isnan(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ---------------------------------------------------------------- curves from a run's result files

def best_layer_curve(df: pd.DataFrame, value: str, feature: str) -> pd.DataFrame:
    """Per step, the maximum of `value` over layers for one feature."""
    d = df[df.feature == feature]
    return d.groupby("step")[value].max().reset_index().sort_values("step")


def run_curves(run_dir, decod_metric: str = "acc") -> dict[str, pd.DataFrame]:
    """Behavior, decodability and causal curves for one run, keyed by name. Each is a DataFrame
    with columns step, value. Decodability uses `decod_metric` (probe accuracy by default, whose
    baseline is the untrained model at step 0; "selectivity" subtracts a control-task probe, which
    is not informative when the control is as easy as the real task)."""
    from pathlib import Path

    run_dir = Path(run_dir)
    out: dict[str, pd.DataFrame] = {}
    m = pd.read_json(run_dir / "metrics.jsonl", lines=True).sort_values("step")
    out["behavior:agree_logit_diff"] = m[["step", "agree_logit_diff"]].rename(columns={"agree_logit_diff": "value"})
    out["behavior:val_loss"] = m[["step", "val_loss"]].rename(columns={"val_loss": "value"})
    p = run_dir / "probe_results.csv"
    if p.exists():
        d = pd.read_csv(p)
        for f in d.feature.unique():
            out[f"decodability:{f}"] = best_layer_curve(d, decod_metric, f).rename(columns={decod_metric: "value"})
    c = run_dir / "causal_results.csv"
    if c.exists():
        d = pd.read_csv(c)
        for (kind, f), g in d[d.kind.isin(["leace_local", "leace_global"])].groupby(["kind", "feature"]):
            out[f"causal:{kind}:{f}"] = best_layer_curve(g, "adj_effect", f).rename(columns={"adj_effect": "value"})
        h = d[d.kind == "head_ablation"][["step", "adj_effect"]].rename(columns={"adj_effect": "value"})
        if len(h):
            out["causal:head_ablation"] = h.sort_values("step")
    return out


def run_onsets(run_dir, **kw) -> pd.DataFrame:
    rows = []
    for name, c in run_curves(run_dir).items():
        kind, _, rest = name.partition(":")
        rows.append({"curve": name, "kind": kind, "feature": rest, "onset": onset(c.step, c.value, **kw)})
    return pd.DataFrame(rows)
