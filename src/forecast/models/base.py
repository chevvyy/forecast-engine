"""Common model interface. Every model predicts quantiles for one task: predict(X) -> [n, q]."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Task:
    name: str            # e.g. "rv_30"
    kind: str            # "ret" | "rv"
    horizon_min: int
    horizon_bars: int
    label_col: str
    quantiles: tuple[float, ...]

    @property
    def qs(self) -> np.ndarray:
        return np.asarray(self.quantiles)


class QuantileModel(Protocol):
    name: str

    def fit(self, X: pl.DataFrame, y: np.ndarray, X_es: pl.DataFrame, y_es: np.ndarray,
            task: Task) -> None: ...

    def predict(self, X: pl.DataFrame) -> np.ndarray: ...

    def vram_estimate(self) -> float: ...


class PickleMixin:
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def vram_estimate(self) -> float:
        return 0.0


def to_target_space(y: np.ndarray, task: Task) -> np.ndarray:
    return np.log(np.maximum(y, 1e-8)) if task.kind == "rv" else y


def from_target_space(z: np.ndarray, task: Task) -> np.ndarray:
    return np.exp(z) if task.kind == "rv" else z


def monotone(q: np.ndarray) -> np.ndarray:
    return np.sort(q, axis=1)
