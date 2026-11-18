"""Synthetic minute bars on the real XNYS calendar, for tests and offline development.

Vol clustering (GARCH-like) + intraday U-shape + a market factor. Optionally plants a weak, known
mean-reversion signal in returns so tests can check the pipeline finds real signal and does NOT find
signal in pure noise.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from forecast.data.align import session_minutes


def make_bars(symbols: list[str], start: datetime, end: datetime, signal: float = 0.0,
              seed: int = 0, drop_frac: float = 0.01) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    grid = session_minutes(start, end)
    n, k = grid.height, len(symbols)
    et = grid["ts"].dt.convert_time_zone("America/New_York")
    tod = (et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)).to_numpy()
    x = (tod - 570) / 390.0
    ushape = 0.7 + 6.4 * (x - 0.5) ** 2
    new_sess = np.r_[True, grid["session"].to_numpy()[1:] != grid["session"].to_numpy()[:-1]]

    base = 0.0006
    omega, alpha, beta = 0.02, 0.08, 0.90
    h_m = np.ones(1)
    h_i = np.ones(k)
    r = np.zeros((n, k))
    hist = np.zeros((15, k))
    for t in range(n):
        zm, zi = rng.standard_normal(), rng.standard_normal(k)
        em = np.sqrt(h_m) * zm
        ei = np.sqrt(h_i) * zi
        sig = base * ushape[t] * (3.0 if new_sess[t] else 1.0)
        mu = -signal * hist.mean(axis=0)
        r[t] = mu + sig * (0.5 * em + 0.85 * ei)
        h_m = omega + alpha * em**2 + beta * h_m
        h_i = omega + alpha * ei**2 + beta * h_i
        hist = np.roll(hist, -1, axis=0)
        hist[-1] = r[t]
    close = 100 * np.exp(np.cumsum(r, axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.00005, (n, k)))
    wig = np.abs(rng.normal(0, 0.0003, (n, k)))
    high = np.maximum(open_, close) * np.exp(wig)
    low = np.minimum(open_, close) * np.exp(-wig)
    vol = 1e4 * ushape[:, None] * np.exp(rng.normal(0, 0.4, (n, k))) * (1 + 200 * np.abs(r))
    frames = []
    for j, s in enumerate(symbols):
        frames.append(pl.DataFrame({
            "symbol": s, "ts": grid["ts"], "open": open_[:, j], "high": high[:, j], "low": low[:, j],
            "close": close[:, j], "volume": vol[:, j], "trade_count": vol[:, j] / 50,
            "vwap": (high[:, j] + low[:, j] + close[:, j]) / 3,
        }))
    df = pl.concat(frames)
    if drop_frac > 0:
        keep = rng.random(df.height) > drop_frac
        df = df.filter(pl.Series(keep))
    return df.sort(["symbol", "ts"])
