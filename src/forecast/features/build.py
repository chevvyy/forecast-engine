"""Point-in-time features and labels.

Row ``i`` of a symbol is "the close of bar i". Inputs use bars ``<= i`` only; labels cover bars
``i+1 .. i+h``. Nothing here may look past ``i`` on the input side (tests/test_leakage.py).

Targets (ADR-0001: returns and realized volatility, never price), both scaled by trailing volatility so
one model works across symbols and regimes:
  y[..., 0] = forward log return over h bars / (rv20 * sqrt(h))
  y[..., 1] = log( forward realized vol over h bars / (rv20 * sqrt(h)) )
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

CHANNELS = ["ret", "abs_ret", "range", "vol_z", "mkt_ret", "gap"]
TAB = ["log_rv5_rv20", "log_rv60_rv20"]  # HAR-style regressors for the baseline
WARMUP = 60
EPS = 1e-8


@dataclass
class Dataset:
    X: np.ndarray        # [n, C, L] float32 channel windows ending at bar i
    tab: np.ndarray      # [n, len(TAB)] float32
    y: np.ndarray        # [n, H, 2] float32
    t: np.ndarray        # [n] int64 epoch seconds of bar i
    t_end: np.ndarray    # [n] int64 epoch seconds of the last bar any label of row i touches
    sym: np.ndarray      # [n] int32 index into symbols
    scale: np.ndarray    # [n] float32 rv20 at i (multiply by sqrt(h) to undo the target scaling)
    symbols: list[str]
    horizons: list[int]


def _roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last ``w`` values including the current one (shorter at the start)."""
    cs = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(idx - w, 0)
    return (cs[idx] - cs[lo]) / (idx - lo)


def symbol_arrays(o, h, low, c, v, mkt_ret) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar channels [n, C], tab features [n, T], log returns r [n], rv20 [n]. All causal."""
    r = np.zeros(len(c))
    r[1:] = np.log(c[1:] / c[:-1])
    gap = np.zeros(len(c))
    gap[1:] = np.log(o[1:] / c[:-1])
    rng = np.log(np.maximum(h, low + EPS) / low)
    lv = np.log1p(np.maximum(v, 0.0))
    m = _roll_mean(lv, 20)
    sd = np.sqrt(np.maximum(_roll_mean(lv * lv, 20) - m * m, 0.0)) + 1e-3
    vol_z = np.clip((lv - m) / sd, -6, 6)
    r2 = r * r
    rv5, rv20, rv60 = (np.sqrt(_roll_mean(r2, w)) + EPS for w in (5, 20, 60))
    ch = np.stack([r, np.abs(r), rng, vol_z, mkt_ret, gap], axis=1)
    tab = np.stack([np.log(rv5 / rv20), np.log(rv60 / rv20)], axis=1)
    return ch, tab, r, rv20


def build(bars: pl.DataFrame, horizons: list[int], lookback: int, benchmark: str = "SPY",
          max_t_end: int | None = None, first_index: int = 0) -> Dataset:
    """Build the modelling table. ``max_t_end`` drops every row whose label would touch a bar at or
    after that epoch second; the loop uses it to keep the final holdout physically out of reach.
    ``first_index`` pins the first usable bar so jobs with different lookbacks share identical folds."""
    hmax = max(horizons)
    symbols = sorted(bars["symbol"].unique().to_list())
    bench = bars.filter(pl.col("symbol") == benchmark).sort("t")
    if bench.height:
        bc = bench["close"].to_numpy()
        br = np.zeros(len(bc))
        br[1:] = np.log(bc[1:] / bc[:-1])
        bench = pl.DataFrame({"t": bench["t"], "mkt_ret": br})
    Xs, tabs, ys, ts, tes, syms, scs = [], [], [], [], [], [], []
    for si, s in enumerate(symbols):
        d = bars.filter(pl.col("symbol") == s).sort("t")
        if bench.height:
            d = d.join(bench, on="t", how="left").with_columns(pl.col("mkt_ret").fill_null(0.0))
        else:
            d = d.with_columns(pl.lit(0.0).alias("mkt_ret"))
        n = d.height
        first = max(WARMUP, lookback - 1, first_index)
        if n < first + hmax + 2:
            continue
        ch, tab, r, rv20 = symbol_arrays(
            d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(),
            d["volume"].to_numpy(), d["mkt_ret"].to_numpy(),
        )
        tsec = d["t"].dt.epoch("s").to_numpy()
        idx = np.arange(first, n - hmax)
        if max_t_end is not None:
            idx = idx[tsec[idx + hmax] < max_t_end]
        if len(idx) == 0:
            continue
        cs, cs2 = np.cumsum(r), np.cumsum(r * r)
        y = np.empty((len(idx), len(horizons), 2), dtype=np.float32)
        for hi, hz in enumerate(horizons):
            sc = rv20[idx] * np.sqrt(hz)
            y[:, hi, 0] = (cs[idx + hz] - cs[idx]) / sc
            y[:, hi, 1] = np.log((np.sqrt(cs2[idx + hz] - cs2[idx]) + EPS) / sc)
        win = np.lib.stride_tricks.sliding_window_view(ch, lookback, axis=0)  # [n-L+1, C, L]
        Xs.append(win[idx - lookback + 1].astype(np.float32))
        tabs.append(tab[idx].astype(np.float32))
        ys.append(np.clip(y, -8, 8))
        ts.append(tsec[idx])
        tes.append(tsec[idx + hmax])
        syms.append(np.full(len(idx), si, dtype=np.int32))
        scs.append(rv20[idx].astype(np.float32))
    if not Xs:
        raise ValueError("no usable rows: not enough history for the lookback/horizons requested")
    return Dataset(np.concatenate(Xs), np.concatenate(tabs), np.concatenate(ys), np.concatenate(ts),
                   np.concatenate(tes), np.concatenate(syms), np.concatenate(scs), symbols, list(horizons))
