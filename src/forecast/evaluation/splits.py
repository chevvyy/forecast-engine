"""Expanding walk-forward with purge (by label interval) + embargo, and a final holdout carve-out."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Fold:
    k: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_start: datetime
    val_end: datetime


def holdout_cutoff(meta: pl.DataFrame, holdout: timedelta) -> datetime:
    return meta["label_start"].max() - holdout


def research_mask(meta: pl.DataFrame, holdout: timedelta, purge_gap: timedelta) -> np.ndarray:
    """Rows usable for research: decision before the holdout and label finishing before it (purged)."""
    cut = holdout_cutoff(meta, holdout)
    return ((meta["label_start"] < cut - purge_gap) & (meta["label_end"] <= cut)).to_numpy()


def walk_forward(meta: pl.DataFrame, first_val_start: datetime, val_block: timedelta,
                 step: timedelta, embargo: timedelta, holdout: timedelta,
                 min_train_rows: int = 1000) -> Iterator[Fold]:
    """Train = rows whose label interval ends at or before (val_start - embargo): purge by label
    interval plus a safety gap. In expanding walk-forward no training row comes after a val block,
    so the classic post-val embargo is applied as a pre-val gap instead (strictly more conservative).
    Holdout rows never appear in train or val."""
    ls = meta["label_start"].to_numpy()
    le = meta["label_end"].to_numpy()
    cut = np.datetime64(holdout_cutoff(meta, holdout).replace(tzinfo=None), "us")
    ls = ls.astype("datetime64[us]")
    le = le.astype("datetime64[us]")
    to64 = lambda d: np.datetime64(d.replace(tzinfo=None), "us")  # noqa: E731
    vs = first_val_start
    k = 0
    while True:
        ve = vs + val_block
        vs64, ve64 = to64(vs), min(to64(ve), cut)
        if vs64 >= cut:
            return
        val = (ls >= vs64) & (ls < ve64) & (le <= cut)
        train = (ls < vs64) & (le <= to64(vs - embargo))      # purge + gap
        tr_idx, va_idx = np.flatnonzero(train), np.flatnonzero(val)
        if len(tr_idx) >= min_train_rows and len(va_idx) > 0:
            yield Fold(k, tr_idx, va_idx, vs, ve)
            k += 1
        vs = vs + step


def inner_split(meta: pl.DataFrame, train_idx: np.ndarray, frac: float = 0.15
                ) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered inner split of a training set for early stopping / tuning, purged."""
    ls = meta["label_start"].to_numpy()[train_idx].astype("datetime64[us]")
    le = meta["label_end"].to_numpy()[train_idx].astype("datetime64[us]")
    cut = np.quantile(ls.astype("int64"), 1 - frac).astype("datetime64[us]")
    fit = train_idx[(ls < cut) & (le <= cut)]
    es = train_idx[ls >= cut]
    return fit, es
