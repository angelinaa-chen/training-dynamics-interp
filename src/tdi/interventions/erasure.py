"""Linear concept erasure (LEACE, Belrose et al. 2023) and matched random controls.

LEACE removes, with the smallest change to the representation, all linearly available information
about a categorical label: after erasure no linear probe can beat chance. For labels z and
features x with mean mu, covariance S_xx and cross-covariance S_xz:
    W = S_xx^(-1/2),  P = projector onto the column space of W S_xz,
    x' = x - S_xx^(1/2) P W (x - mu)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class Eraser:
    mu: torch.Tensor  # [d]
    A: torch.Tensor  # [d, d]; x' = x - (x - mu) A^T
    rank: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mu, A = self.mu.to(x), self.A.to(x)
        return x - (x - mu) @ A.T

    def hook(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.__call__


def fit_leace(X: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> Eraser:
    X = X.astype(np.float64)
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    classes = np.unique(y)
    Z = (y[:, None] == classes[None, :]).astype(np.float64)
    Z = Z[:, :-1] if len(classes) > 1 else Z  # drop one column: one-hot columns sum to 1
    Zc = Z - Z.mean(0)
    sxx = Xc.T @ Xc / n
    sxz = Xc.T @ Zc / n
    evals, evecs = np.linalg.eigh(sxx + eps * np.trace(sxx) / d * np.eye(d))
    evals = np.clip(evals, 1e-12, None)
    W = (evecs / np.sqrt(evals)) @ evecs.T
    Winv = (evecs * np.sqrt(evals)) @ evecs.T
    U, s, _ = np.linalg.svd(W @ sxz, full_matrices=False)
    keep = s > 1e-8 * max(s.max(), 1e-30)
    U = U[:, keep]
    A = Winv @ U @ U.T @ W
    return Eraser(torch.tensor(mu, dtype=torch.float32), torch.tensor(A, dtype=torch.float32), int(keep.sum()))


def random_eraser(like: Eraser, seed: int) -> Eraser:
    """Projects out a random subspace of the same rank (orthogonal projection, same centering)."""
    d = like.mu.numel()
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, like.rank, generator=g))
    return Eraser(like.mu.clone(), Q @ Q.T, like.rank)


def class_mean_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit vector from the class-0 mean to the class-1 mean (binary labels)."""
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    return v / (np.linalg.norm(v) + 1e-12)


def steer_hook(direction: torch.Tensor, alpha: float, pos: torch.Tensor | None = None):
    """Adds alpha * direction to the residual, at positions `pos` ([B] long) or everywhere."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        v = direction.to(x) * alpha
        if pos is None:
            return x + v
        out = x.clone()
        out[torch.arange(x.shape[0]), pos.to(x.device)] += v
        return out
    return fn


def local_hook(fn: Callable[[torch.Tensor], torch.Tensor], pos: torch.Tensor):
    """Applies `fn` to the residual only at one position per example (`pos` [B] long; -1 = skip)."""
    def hook(x: torch.Tensor) -> torch.Tensor:
        p = pos.to(x.device)
        b = (p >= 0).nonzero().squeeze(1)
        if len(b) == 0:
            return x
        out = x.clone()
        out[b, p[b]] = fn(x[b, p[b]])
        return out
    return hook
