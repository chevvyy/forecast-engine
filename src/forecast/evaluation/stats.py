"""Diebold-Mariano (HAC, HLN-corrected) and Deflated Sharpe Ratio."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = float(d @ d) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * float(d[k:] @ d[:-k]) / n
    return max(v, 1e-300)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1,
                    alternative: str = "less") -> tuple[float, float]:
    """Test H0: E[loss_a - loss_b] = 0. alternative='less' -> p small when model A is better.

    HAC variance with max(h-1, n^(1/3)) Newey-West lags (overlapping labels need more than h-1),
    Harvey-Leybourne-Newbold small-sample correction, Student-t reference.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    lags = max(h - 1, int(round(n ** (1 / 3))))
    dm = d.mean() / math.sqrt(newey_west_var(d, lags) / n)
    hln = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    t = dm * hln
    if alternative == "less":
        p = stats.t.cdf(t, n - 1)
    elif alternative == "greater":
        p = stats.t.sf(t, n - 1)
    else:
        p = 2 * stats.t.sf(abs(t), n - 1)
    return float(t), float(p)


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_var_trials: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability the observed (per-period) Sharpe beats the expected max of n_trials null Sharpes
    (Bailey & Lopez de Prado 2014)."""
    if n_trials < 1 or n_obs < 2:
        return float("nan")
    g = 0.5772156649
    z = stats.norm.ppf
    sr0 = math.sqrt(max(sr_var_trials, 1e-12)) * (
        (1 - g) * z(1 - 1 / max(n_trials, 2)) + g * z(1 - 1 / (max(n_trials, 2) * math.e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom))
