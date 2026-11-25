"""Rung-0 baselines every model has to beat on the same folds.

- ``Unconditional``: train-set empirical quantiles of the vol-scaled targets. For returns this is the
  zero-return / random-walk baseline with honest spread; it is very hard to beat.
- ``HarRV``: HAR-style linear model for the log-vol target on (rv5/rv20, rv60/rv20), quantiles from
  train residuals. Returns fall back to the unconditional quantiles.
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
    def fit(self, train: dict, val: dict | None = None, config: dict | None = None) -> dict:
        super().fit(train)
        A = np.concatenate([train["tab"], np.ones((len(train["tab"]), 1), np.float32)], axis=1)
        yv = train["y"][:, :, 1]                                   # [n, H]
        self.beta, *_ = np.linalg.lstsq(A, yv, rcond=None)         # [T+1, H]
        self.resid_q = np.quantile(yv - A @ self.beta, self.q, axis=0).T.astype(np.float32)  # [H, Q]
        return {}

    def predict(self, X: np.ndarray, tab: np.ndarray) -> np.ndarray:
        out = super().predict(X, tab)
        A = np.concatenate([tab, np.ones((len(tab), 1), np.float32)], axis=1)
        out[:, :, 1, :] = (A @ self.beta)[:, :, None] + self.resid_q[None]
        return out
