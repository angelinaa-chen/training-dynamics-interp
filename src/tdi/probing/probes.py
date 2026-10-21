from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import Samples


def split_by_group(group: np.ndarray, seed: int, test_frac: float = 0.3) -> np.ndarray:
    """Boolean mask, True = test. Whole groups go to one side."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(group)
    test_groups = rng.choice(uniq, size=max(1, int(round(len(uniq) * test_frac))), replace=False)
    return np.isin(group, test_groups)


def control_labels(s: Samples, seed: int) -> np.ndarray:
    """Hewitt and Liang (2019) control task: every token type gets a fixed random label drawn
    from the empirical label distribution. A probe can only solve it by memorizing token types."""
    rng = np.random.default_rng(seed)
    mapping = {int(t): int(rng.choice(s.y)) for t in np.unique(s.tok)}
    return np.array([mapping[int(t)] for t in s.tok])


def probe_accuracy(X: np.ndarray, y: np.ndarray, test: np.ndarray, C: float = 1.0) -> float:
    """Test accuracy of a regularized logistic-regression probe. Hyperparameters are fixed
    everywhere so curves are comparable across layers and checkpoints."""
    ytr, yte = y[~test], y[test]
    if len(np.unique(ytr)) < 2:
        return float((yte == ytr[0]).mean())
    scaler = StandardScaler().fit(X[~test])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, max_iter=300).fit(scaler.transform(X[~test]), ytr)
    return float((clf.predict(scaler.transform(X[test])) == yte).mean())
