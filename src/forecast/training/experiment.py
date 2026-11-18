"""Data preparation, walk-forward evaluation and scoring shared by every command."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from forecast.data.align import session_align
from forecast.evaluation import metrics as M
from forecast.evaluation.splits import Fold, inner_split, walk_forward
from forecast.evaluation.stats import diebold_mariano
from forecast.features.build import FeatureSet, build, subsample
from forecast.models.base import Task
from forecast.utils import env
from forecast.utils.config import config_hash, load_yaml
from forecast.utils.timeutil import TIMEFRAME_MINUTES, parse_duration, utc


# ----------------------------------------------------------------------------- data
def load_run_config(path: str | Path = "configs/run/default.yaml") -> dict:
    cfg = load_yaml(path)
    cfg["features_cfg"] = load_yaml(cfg["data"]["features"])
    cfg["universe_cfg"] = load_yaml(cfg["data"]["universe"])
    return cfg


def universe_symbols(u: dict) -> list[str]:
    return list(dict.fromkeys(u["benchmarks"] + u["sectors"] + u["singles"]))


def load_bars(cfg: dict) -> tuple[pl.DataFrame, str]:
    d = cfg["data"]
    if d["source"] == "synthetic":
        from forecast.data.synthetic import make_bars
        s = d["synthetic"]
        syms = ["SPY", "QQQ"] + [f"S{i:02d}" for i in range(s["n_symbols"])]
        bars = make_bars(syms, utc(s["start"]), utc(s["end"]), signal=s.get("signal", 0.0),
                         seed=s.get("seed", 0))
        return bars, "synthetic-" + config_hash(s)
    from forecast.data.store import read_bars, snapshot_hash
    syms = universe_symbols(cfg["universe_cfg"])
    bars = read_bars(d.get("layer", "adjusted"), d["timeframe"], syms)
    if d.get("start"):
        bars = bars.filter(pl.col("ts") >= utc(str(d["start"])))
    h = snapshot_hash(d.get("layer", "adjusted"), d["timeframe"])
    return bars, config_hash([h, str(d.get("start"))])


def prepare(cfg: dict, use_cache: bool = True) -> tuple[FeatureSet, str]:
    """Bars -> session-aligned -> features/labels -> subsampled. Cached by data + feature hash."""
    bars, data_hash = load_bars(cfg)
    tf = cfg["data"]["timeframe"]
    key = config_hash([data_hash, cfg["features_cfg"], cfg["data"].get("subsample_every", 1), tf])
    cache = env.data_root() / "derived" / "features" / f"{key}.parquet"
    aligned = session_align(bars, tf)
    fs = build(aligned, cfg["features_cfg"], tf)
    fs = subsample(fs, cfg["data"].get("subsample_every", 1))
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fs.df.write_parquet(cache)
    return fs, data_hash


def make_tasks(fs: FeatureSet, cfg: dict) -> dict[str, Task]:
    bm = TIMEFRAME_MINUTES[cfg["data"]["timeframe"]]
    qs = tuple(cfg["features_cfg"]["labels"]["quantiles"])
    out = {}
    for name, col in fs.label_cols.items():
        kind, h = name.split("_")
        out[name] = Task(name, kind, int(h), max(1, int(h) // bm), col, qs)
    return out


def folds(fs: FeatureSet, cfg: dict, max_folds: int | None = None) -> list[Fold]:
    e = cfg["eval"]
    fl = list(walk_forward(fs.df, utc(e["first_val_start"]), parse_duration(e["val_block"]),
                           parse_duration(e["step"]), timedelta(minutes=e["embargo_min"]),
                           parse_duration(e["final_holdout"]), e.get("min_train_rows", 1000)))
    if not fl:
        raise RuntimeError("No folds: check eval.first_val_start against the data range")
    return fl[-max_folds:] if max_folds else fl


# ----------------------------------------------------------------------------- evaluation
@dataclass
class OOF:
    """Out-of-fold predictions for one model on one task, in fold order."""
    model: str
    task: Task
    ts: np.ndarray
    symbol: np.ndarray
    y: np.ndarray
    q: np.ndarray
    fold: np.ndarray
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def loss(self) -> np.ndarray:
        return primary_loss(self.task, self.y, self.q)


def primary_loss(task: Task, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    if task.kind == "rv":
        return M.qlike(y, q[:, len(task.quantiles) // 2])
    return M.crps_q(y, q, task.qs)


def run_oof(factory, fs: FeatureSet, task: Task, fold_list: list[Fold], features: list[str],
            on_fold=None) -> OOF:
    """Fit a fresh model per fold (inner purged split for early stopping), predict the val block.
    `on_fold(k, fold_loss)` may raise to prune (Optuna)."""
    t0 = time.time()
    df = fs.df
    y_all = df[task.label_col].to_numpy()
    parts = []
    for f in fold_list:
        tr = f.train_idx[np.isfinite(y_all[f.train_idx])]
        va = f.val_idx[np.isfinite(y_all[f.val_idx])]
        fit_idx, es_idx = inner_split(df, tr)
        X = df.select(features)
        m = factory()
        m.fit(X[fit_idx], y_all[fit_idx], X[es_idx], y_all[es_idx], task)
        q = m.predict(X[va])
        parts.append((va, q, f.k))
        if on_fold is not None:
            on_fold(f.k, float(np.nanmean(primary_loss(task, y_all[va], q))), m)
    idx = np.concatenate([p[0] for p in parts])
    return OOF(getattr(m, "name", "model"), task, df["ts"].to_numpy()[idx], df["symbol"].to_numpy()[idx],
               y_all[idx], np.vstack([p[1] for p in parts]),
               np.concatenate([np.full(len(p[0]), p[2]) for p in parts]), time.time() - t0)


def score(o: OOF) -> dict:
    t, y, q = o.task, o.y, o.q
    med = q[:, len(t.quantiles) // 2]
    out = {"n": int(len(y)), "crps": float(np.mean(M.crps_q(y, q, t.qs))),
           "pinball": float(np.mean(M.pinball(y, q, t.qs))), "coverage_80": M.coverage(y, q, t.qs)}
    if t.kind == "rv":
        out["qlike"] = float(np.mean(M.qlike(y, med)))
        out["mse_log_rv"] = float(np.mean((np.log(y + 1e-10) - np.log(med + 1e-10)) ** 2))
    else:
        out["rank_ic"], out["rank_ic_ir"] = M.rank_ic(o.ts, med, y)
        out["hit_rate"], out["hit_p"], out["hit_n"] = M.hit_rate(med, y)
        up = float(np.mean(y[y != 0] > 0))
        out["always_up_rate"] = up
    out["per_fold"] = {int(k): float(np.mean(primary_loss(t, y[o.fold == k], q[o.fold == k])))
                       for k in np.unique(o.fold)}
    return out


def ts_loss(o: OOF) -> tuple[np.ndarray, np.ndarray]:
    return M.per_timestamp(o.ts, o.loss())


def compare_series(ta, la, tb, lb, h: int) -> dict:
    """DM test on per-timestamp averaged primary loss. p small => series a better than b."""
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    stat, p = diebold_mariano(la[ia], lb[ib], h=h)
    return {"dm_stat": stat, "dm_p": p, "loss_a": float(la[ia].mean()), "loss_b": float(lb[ib].mean()),
            "rel": float(la[ia].mean() / lb[ib].mean() - 1), "n_ts": int(len(common))}


def compare(a: OOF, b: OOF) -> dict:
    return compare_series(*ts_loss(a), *ts_loss(b), h=a.task.horizon_bars)
