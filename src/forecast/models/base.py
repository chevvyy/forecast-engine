"""Common model interface (docs/ARCHITECTURE.md section 2). Everything downstream of training talks to
models only through this, so baselines and deep models are interchangeable."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Model(Protocol):
    def fit(self, train: dict, val: dict, config: dict) -> dict: ...
    def predict(self, X: np.ndarray, tab: np.ndarray) -> np.ndarray:
        """-> quantiles [n, horizons, targets(2: return, log-vol), Q], non-decreasing along Q."""
        ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def vram_estimate(self, config: dict) -> float: ...
