"""Rung-0 baselines. Point forecast + empirical residual quantiles from the fit set."""
from __future__ import annotations

import numpy as np
import polars as pl

from forecast.models.base import PickleMixin, Task, from_target_space, monotone, to_target_space


class _Residual(PickleMixin):
    name = "base"

    def point(self, X: pl.DataFrame) -> np.ndarray:  # in target space
        raise NotImplementedError

    def _prep(self, X: pl.DataFrame, y: np.ndarray, task: Task) -> None:
        pass

    def fit(self, X, y, X_es, y_es, task: Task) -> None:
        self.task = task
        self._prep(X, y, task)
        z = to_target_space(y, task)
        res = z - self.point(X)
        res = res[np.isfinite(res)]
        self.res_q = np.quantile(res, task.qs)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return monotone(from_target_space(self.point(X)[:, None] + self.res_q[None, :], self.task))


class ZeroReturn(_Residual):
    """Returns: median 0; quantile spread from training distribution (climatology)."""
    name = "zero_return"

    def point(self, X):
        return np.zeros(X.height)


class RandomWalk(_Residual):
    """Returns: last h-bar return persists. Vol: last realized vol over ~h persists."""
    name = "random_walk"

    def _prep(self, X, y, task):
        pre = "ret_" if task.kind == "ret" else "rv_"
        cands = sorted((int(c[len(pre):]), c) for c in X.columns
                       if c.startswith(pre) and c[len(pre):].isdigit())
        self.col = min(cands, key=lambda t: abs(t[0] - task.horizon_min))[1]
        self.scale = task.horizon_min / int(self.col[len(pre):])

    def point(self, X):
        x = X[self.col].to_numpy().astype(float)
        if self.task.kind == "ret":
            return np.nan_to_num(x * self.scale)
        return to_target_space(np.nan_to_num(x * np.sqrt(self.scale), nan=1e-4), self.task)


class RollingMean(_Residual):
    """Returns: mean 1-bar return over the last session, scaled to horizon."""
    name = "rolling_mean_390"

    def _prep(self, X, y, task):
        self.col = "ret_390"

    def point(self, X):
        if self.col not in X.columns:
            return np.zeros(X.height)
        return np.nan_to_num(X[self.col].to_numpy().astype(float) / 390 * self.task.horizon_min)


class HARRV(_Residual):
    """HAR-style vol model: OLS of log forward RV on log RV at short/day/week windows."""
    name = "har_rv"

    def _prep(self, X, y, task):
        cols = [c for c in X.columns if c.startswith("rv_") and c[3:].isdigit()]
        fin = {c: float(np.isfinite(X[c].to_numpy().astype(float)).mean()) for c in cols}
        self.cols = [c for c in cols if fin[c] > 0.5] or cols[:1]
        self.med = None
        A = self._design(X)
        self.med = np.nanmedian(np.where(np.isfinite(A), A, np.nan), axis=0)
        z = to_target_space(y, task)
        m = np.isfinite(A).all(1) & np.isfinite(z)
        self.beta, *_ = np.linalg.lstsq(A[m], z[m], rcond=None)

    def _design(self, X):
        L = np.log(np.maximum(X.select(self.cols).to_numpy().astype(float), 1e-8))
        return np.column_stack([np.ones(len(L)), L])

    def point(self, X):
        A = self._design(X)
        A = np.where(np.isfinite(A), A, self.med)
        return A @ self.beta


class ZeroReturnVolScaled(PickleMixin):
    """Returns: median 0, spread = standardized-residual quantiles x HAR-predicted vol.
    The honest return baseline: a model beating ZeroReturn only by forecasting the spread is a vol
    model, not a return model. Beating this one requires directional/shape skill."""
    name = "zero_return_volscaled"

    def fit(self, X, y, X_es, y_es, task: Task) -> None:
        self.task = task
        vt = Task(task.name, "rv", task.horizon_min, task.horizon_bars, task.label_col, task.quantiles)
        self.har = HARRV()
        self.har.task = vt
        self.har._prep(X, np.abs(y) + 1e-8, vt)   # HAR on |return| as a vol proxy
        sig = np.exp(self.har.point(X))
        z = y / sig
        self.zq = np.quantile(z[np.isfinite(z)], task.qs)

    def predict(self, X):
        sig = np.exp(self.har.point(X))
        return monotone(sig[:, None] * self.zq[None, :])


def baselines_for(task: Task) -> list:
    if task.kind == "ret":
        return [ZeroReturn(), ZeroReturnVolScaled(), RandomWalk(), RollingMean()]
    return [HARRV(), RandomWalk()]
