"""Curve plots: behavior, decodability and causal effect over training, with onsets marked.

Colors are the first three slots of a palette validated for color-vision deficiency
(blue, orange, aqua); every series is also labeled directly and linestyles differ, so identity
never depends on color alone. Each curve is rescaled to progress (0 at its start, 1 at its end)
so curves with different units share one axis; raw values are in the CSV files.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .onset import onset

COLORS = {"behavior": "#2a78d6", "decodability": "#eb6834", "causal": "#1baf7a"}
STYLES = {"behavior": "-", "decodability": "--", "causal": "-."}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def progress(steps, values, k: int = 3):
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    o = np.argsort(steps)
    steps, values = steps[o], values[o]
    init, final = np.median(values[:k]), np.median(values[-k:])
    if abs(final - init) < 1e-12:
        return steps, np.zeros_like(values)
    return steps, (values - init) / (final - init)


def plot_run(curves: dict, features: list[str], path, title: str | None = None,
             min_effect: float = 1.0) -> None:
    """`curves` is the output of onset.run_curves. One panel per feature."""
    fig, axes = plt.subplots(1, len(features), figsize=(4.6 * len(features), 3.9), squeeze=False, sharey=True)
    beh = curves["behavior:agree_logit_diff"]
    for ax, feat in zip(axes[0], features):
        series = {
            "behavior": ("agreement", beh),
            "decodability": ("probe", curves.get(f"decodability:{feat}")),
            "causal": ("erasure", curves.get(f"causal:leace_local:{feat}")),
        }
        for kind, (label, c) in series.items():
            if c is None:
                continue
            if kind == "causal" and abs(np.median(c.value.values[-3:]) - np.median(c.value.values[:3])) < min_effect:
                # Normalizing a near-zero effect would just plot noise as if it were an emergence.
                ax.text(0.97, 0.25, f"erasure effect < {min_effect:g} nat:\nfeature not used", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=9, color=COLORS["causal"])
                continue
            x, y = progress(c.step, c.value)
            ax.plot(x, y, STYLES[kind], color=COLORS[kind], lw=2, label=label)
            t = onset(c.step, c.value)
            if not np.isnan(t):
                ax.axvline(t, color=COLORS[kind], lw=1, alpha=0.6, ls=":")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, float(beh.step.max()))
        ax.set_ylim(-0.25, 1.25)
        ax.set_title(feat, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training step", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    axes[0][0].set_ylabel("progress (0 = start, 1 = end)", color=MUTED)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
