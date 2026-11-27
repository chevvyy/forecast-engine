"""Rung-0 baselines every model has to beat on the same folds.

- ``Unconditional``: train-set empirical quantiles of the vol-scaled targets. For returns this is the
  zero-return / random-walk baseline with honest spread; it is very hard to beat.
- ``HarRV``: HAR-style linear model for the log-vol target on (rv5/rv20, rv60/rv20), quantiles from
  train residuals. Returns fall back to the unconditional quantiles. It reads the first ``n_har``
  columns of ``tab`` only, so it never sees the clock.
- ``HarTod`` (intraday only): the same regression plus one level per time-of-day bin (15 minutes of
  the session by default), per horizon. In log space that is a multiplicative diurnal factor on the
  forward / trailing volatility ratio: the baseline that knows the U-shape of the session. Without it a
  model given the clock could "beat HAR" by learning the time of day and nothing about the market.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Unconditional:
    def __init__(self, quantiles: list[float]):
        self.q = quantiles
        self.table: np.ndarray | None = None   # [H, 2, Q]

    def fit(self, train: dict, val: dict | None = None, config: dict | None = None) -> dict:
        self.table = np.quantile(train["y"], self.q, axis=0).transpose(1, 2, 0).astype(np.float32)
        return {}

    def predict(self, X: np.ndarray, tab: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.table, (len(tab), *self.table.shape)).copy()

    def save(self, path: Path) -> None:
        np.save(path, self.table)

    def load(self, path: Path) -> None:
        self.table = np.load(path)

    def vram_estimate(self, config: dict | None = None) -> float:
        return 0.0


class HarRV(Unconditional):
    def __init__(self, quantiles: list[float], n_har: int = 2):
        super().__init__(quantiles)
        self.n_har = n_har            # leading columns of ``tab`` that are the HAR regressors

    def design(self, tab: np.ndarray) -> np.ndarray:
        return np.concatenate([tab[:, :self.n_har], np.ones((len(tab), 1), np.float32)], axis=1)

    def fit(self, train: dict, val: dict | None = None, config: dict | None = None) -> dict:
        super().fit(train)
        A = self.design(train["tab"])
        yv = train["y"][:, :, 1]                                   # [n, H]
        self.beta, *_ = np.linalg.lstsq(A, yv, rcond=None)         # [T+1, H]
        self.resid_q = np.quantile(yv - A @ self.beta, self.q, axis=0).T.astype(np.float32)  # [H, Q]
        return {}

    def predict(self, X: np.ndarray, tab: np.ndarray) -> np.ndarray:
        out = super().predict(X, tab)
        out[:, :, 1, :] = (self.design(tab) @ self.beta)[:, :, None] + self.resid_q[None]
        return out


class HarTod(HarRV):
    """HAR plus a level per time-of-day bin. ``tod_col`` is the column of ``tab`` holding the share of
    the session elapsed (0..1). The intercept stays in, and ``lstsq`` returns the minimum-norm solution
    of the collinear system, so a bin never seen in training predicts close to the overall level."""

    def __init__(self, quantiles: list[float], tod_col: int, n_bins: int = 26, n_har: int = 2):
        super().__init__(quantiles, n_har)
        self.tod_col, self.n_bins = tod_col, n_bins

    def design(self, tab: np.ndarray) -> np.ndarray:
        b = np.clip((tab[:, self.tod_col] * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        onehot = np.zeros((len(tab), self.n_bins), np.float32)
        onehot[np.arange(len(tab)), b] = 1.0
        return np.concatenate([super().design(tab), onehot], axis=1)
