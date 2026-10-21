import numpy as np

from tdi.analysis.onset import bootstrap_ci, lag, onset, onset_sweep


def sigmoid_curve(t0, steps, width=20.0, lo=0.0, hi=1.0):
    steps = np.asarray(steps, float)
    return lo + (hi - lo) / (1 + np.exp(-(steps - t0) / width))


def test_onset_recovers_known_midpoint():
    steps = np.arange(0, 2000, 10)
    for t0 in (300, 800, 1400):
        assert abs(onset(steps, sigmoid_curve(t0, steps)) - t0) < 15


def test_onset_handles_decreasing_curves_and_unsorted_input():
    steps = np.arange(0, 1000, 10)
    v = sigmoid_curve(500, steps, lo=6.0, hi=2.0)  # a loss-like curve going down
    perm = np.random.default_rng(0).permutation(len(steps))
    assert abs(onset(steps[perm], v[perm]) - 500) < 15


def test_onset_ignores_transient_spike_and_flat_curves():
    steps = np.arange(0, 1000, 10)
    v = sigmoid_curve(700, steps)
    v[20] = 1.0  # one-point spike long before the real rise
    assert abs(onset(steps, v) - 700) < 15
    assert np.isnan(onset(steps, np.ones_like(steps, float)))


def test_lag_is_difference_of_onsets_and_sweep_is_monotone():
    steps = np.arange(0, 2000, 10)
    dec, use = sigmoid_curve(500, steps), sigmoid_curve(800, steps)
    assert abs(lag(onset(steps, dec), onset(steps, use)) - 300) < 20
    sw = onset_sweep(steps, use)
    assert sw[0.25] < sw[0.5] < sw[0.75]


def test_bootstrap_ci_contains_mean_and_skips_nans():
    m, lo, hi = bootstrap_ci([100.0, 120.0, np.nan, 110.0])
    assert lo <= m <= hi and abs(m - 110) < 1e-9


def test_min_change_suppresses_noise_only_curves():
    steps = np.arange(0, 1000, 10)
    small = sigmoid_curve(500, steps, lo=0.0, hi=0.3)
    assert abs(onset(steps, small) - 500) < 15
    assert np.isnan(onset(steps, small, min_change=1.0))
