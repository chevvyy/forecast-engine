"""Forecast metrics. Quantile arrays are shape [n, q]."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


def pinball(y: np.ndarray, q: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Per-sample mean pinball loss across quantiles -> shape [n]."""
    d = y[:, None] - q
    return np.mean(np.maximum(qs * d, (qs - 1) * d), axis=1)


def crps_q(y: np.ndarray, q: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """CRPS approximated as 2 x mean pinball over the quantile grid (per sample)."""
    return 2.0 * pinball(y, q, qs)


def coverage(y: np.ndarray, q: np.ndarray, qs: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    i, j = int(np.argmin(np.abs(qs - lo))), int(np.argmin(np.abs(qs - hi)))
    return float(np.mean((y >= q[:, i]) & (y <= q[:, j])))


def qlike(rv: np.ndarray, pred_vol: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """QLIKE on variances (per sample). Robust to noisy RV proxies (Patton 2011)."""
    r = (rv**2 + eps) / (pred_vol**2 + eps)
    return r - np.log(r) - 1


def rank_ic(ts: np.ndarray, pred: np.ndarray, y: np.ndarray, min_names: int = 5) -> tuple[float, float]:
    """Mean per-timestamp Spearman IC and its IR (mean/std)."""
    df = pl.DataFrame({"ts": ts, "p": pred, "y": y})
    ics = (df.with_columns(rp=pl.col("p").rank().over("ts"), ry=pl.col("y").rank().over("ts"),
                           n=pl.len().over("ts"))
           .filter(pl.col("n") >= min_names)
           .group_by("ts").agg(ic=pl.corr("rp", "ry")))["ic"].drop_nans().drop_nulls().to_numpy()
    if len(ics) < 2:
        return float("nan"), float("nan")
    return float(ics.mean()), float(ics.mean() / (ics.std() + 1e-12))


def hit_rate(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Directional hit rate (zeros excluded) with a one-sided binomial p-value vs 50%."""
    m = (pred != 0) & (y != 0)
    n = int(m.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    hits = int((np.sign(pred[m]) == np.sign(y[m])).sum())
    p = stats.binomtest(hits, n, 0.5, alternative="greater").pvalue
    return hits / n, float(p), n


def per_timestamp(ts: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a per-sample loss within each timestamp -> (sorted unique ts, mean loss).
    DM tests run on this series so cross-sectional correlation doesn't fake significance."""
    df = pl.DataFrame({"ts": ts, "l": loss}).group_by("ts").agg(pl.col("l").mean()).sort("ts")
    return df["ts"].to_numpy(), df["l"].to_numpy()
