"""Purged, embargoed expanding walk-forward splits plus a final holdout that loop jobs never see.

Timeline (unique bar timestamps): [ ........ development ........ | final holdout ]
Development is cut into ``n_folds`` consecutive validation blocks at its end. For fold k:
  train      rows whose labels end before (val_start - embargo), minus the early-stopping tail
  early-stop the last ``es_frac`` of the train timeline, purged from the train core
  val        rows inside the block whose labels also end inside development
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train: np.ndarray
    es: np.ndarray
    val: np.ndarray


def holdout_start(t_unique: np.ndarray, holdout_frac: float) -> int:
    """Epoch second of the first holdout bar."""
    k = int(len(t_unique) * (1.0 - holdout_frac))
    return int(t_unique[min(max(k, 1), len(t_unique) - 1)])


def walk_forward(t: np.ndarray, t_end: np.ndarray, n_folds: int = 3, val_frac: float = 0.08,
                 embargo_bars: int = 5, es_frac: float = 0.1) -> list[Fold]:
    """``t``/``t_end`` are per-row label start/end times (epoch s). Rows are assumed to already
    exclude the final holdout (build(..., max_t_end=holdout_start))."""
    ut = np.unique(t)
    vlen = max(int(len(ut) * val_frac), 1)
    folds = []
    for k in range(n_folds):
        ve = len(ut) - (n_folds - 1 - k) * vlen          # exclusive index of val end
        vs = ve - vlen
        cut = ut[max(vs - embargo_bars, 0)]               # train labels must end before this
        val_start = ut[vs]
        val_last = ut[ve - 1]
        train_all = t_end < cut
        tt = np.unique(t[train_all])
        if len(tt) < 50:
            continue
        es_start = tt[int(len(tt) * (1.0 - es_frac))]
        es = train_all & (t >= es_start)
        train = train_all & (t_end < es_start)
        val = (t >= val_start) & (t <= val_last)
        folds.append(Fold(np.flatnonzero(train), np.flatnonzero(es), np.flatnonzero(val)))
    return folds
