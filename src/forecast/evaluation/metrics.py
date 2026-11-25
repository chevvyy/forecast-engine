"""Quantile-forecast metrics and the Diebold-Mariano test."""
from __future__ import annotations

import math

import numpy as np


def pinball_rows(y: np.ndarray, q_pred: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """y [n,H,2], q_pred [n,H,2,Q] -> per-row loss [n,H,2] averaged over quantiles."""
    q = np.asarray(quantiles, dtype=np.float32)
    d = y[..., None] - q_pred
    return np.maximum(q * d, (q - 1.0) * d).mean(-1)


def coverage(y: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    """Share of outcomes inside [lowest, highest] predicted quantile, per (H, target)."""
    return ((y >= q_pred[..., 0]) & (y <= q_pred[..., -1])).mean(0)


def rank_ic(y_ret: np.ndarray, med: np.ndarray, t: np.ndarray) -> float:
    """Mean per-timestamp Spearman correlation between predicted median return and outcome."""
    ics = []
    for ti in np.unique(t):
        m = t == ti
        if m.sum() < 5:
            continue
        a = np.argsort(np.argsort(med[m])).astype(float)
        b = np.argsort(np.argsort(y_ret[m])).astype(float)
        if a.std() > 0 and b.std() > 0:
            ics.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, t: np.ndarray, lag: int) -> tuple[float, float]:
    """DM test on per-timestamp mean loss differences (a - b) with a Newey-West variance.
    Negative statistic = model a has lower loss. Returns (stat, two-sided p)."""
    ut, inv = np.unique(t, return_inverse=True)
    cnt = np.bincount(inv)
    d = np.bincount(inv, weights=loss_a - loss_b) / cnt
    n = len(d)
    if n < 10:
        return 0.0, 1.0
    dm = d - d.mean()
    var = float(dm @ dm) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(dm[k:] @ dm[:-k]) / n
    if var <= 0:
        return 0.0, 1.0
    stat = float(d.mean() / math.sqrt(var / n))
    return stat, float(math.erfc(abs(stat) / math.sqrt(2.0)))
