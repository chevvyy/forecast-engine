"""LightGBM quantile model: one booster per quantile, early-stopped on an inner (purged) split."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from forecast.models.base import PickleMixin, Task, from_target_space, monotone, to_target_space

DEFAULTS = dict(num_leaves=63, learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, min_data_in_leaf=500, lambda_l2=1.0, max_bin=255)


class LGBMQuantile(PickleMixin):
    name = "lightgbm"

    def __init__(self, params: dict | None = None, features: list[str] | None = None,
                 device: str = "cpu", num_boost_round: int = 2000, early_stopping: int = 200,
                 seed: int = 1337, n_jobs: int = -1):
        self.params = {**DEFAULTS, **(params or {})}
        self.features = features
        self.device = device
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.seed = seed
        self.n_jobs = n_jobs
        self.boosters: list[lgb.Booster] = []

    def _X(self, X: pl.DataFrame) -> np.ndarray:
        return X.select(self.features).to_numpy().astype(np.float32)

    def fit(self, X, y, X_es, y_es, task: Task) -> None:
        self.task = task
        self.features = self.features or list(X.columns)
        z, z_es = to_target_space(y, task), to_target_space(y_es, task)
        m, m_es = np.isfinite(z), np.isfinite(z_es)
        dtr = lgb.Dataset(self._X(X)[m], z[m], free_raw_data=False)
        des = lgb.Dataset(self._X(X_es)[m_es], z_es[m_es], reference=dtr)
        self.boosters, self.best_iters = [], []
        for a in task.quantiles:
            p = {**self.params, "objective": "quantile", "alpha": a, "verbosity": -1,
                 "seed": self.seed, "deterministic": True, "num_threads": self.n_jobs,
                 "device_type": self.device}
            try:
                b = lgb.train(p, dtr, self.num_boost_round, valid_sets=[des],
                              callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            except lgb.basic.LightGBMError as e:
                if self.device == "cpu":
                    raise
                warnings.warn(f"LightGBM device={self.device} failed ({e}); falling back to CPU",
                              stacklevel=2)
                self.device = p["device_type"] = "cpu"
                b = lgb.train(p, dtr, self.num_boost_round, valid_sets=[des],
                              callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            self.boosters.append(b)
            self.best_iters.append(b.best_iteration)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        A = self._X(X)
        q = np.column_stack([b.predict(A, num_iteration=b.best_iteration) for b in self.boosters])
        return monotone(from_target_space(q, self.task))

    def feature_importance(self) -> dict[str, float]:
        imp = np.mean([b.feature_importance("gain") for b in self.boosters], axis=0)
        return dict(sorted(zip(self.features, imp, strict=True), key=lambda t: -t[1]))

    def vram_estimate(self) -> float:
        return 2.0 if self.device != "cpu" else 0.0
