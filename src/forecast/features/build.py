"""Point-in-time features and labels.

Row ``i`` of a symbol is "the close of bar i". Inputs use bars ``<= i`` only; labels cover bars
``i+1 .. i+h``. Nothing here may look past ``i`` on the input side (tests/test_leakage.py).

Targets (ADR-0001: returns and realized volatility, never price), both scaled by trailing volatility so
one model works across symbols and regimes:
  y[..., 0] = forward log return over h bars / (rv20 * sqrt(h))
  y[..., 1] = log( forward realized vol over h bars / (rv20 * sqrt(h)) )

Intraday bars (``build(..., bar_seconds=60)``). A bar's ``t`` is its START, so bar i is only known at
``t`` + one bar; ``Dataset.known`` carries that time and is the earliest moment row i may be used.
A row exists only if its ``lookback`` window bars (or ``first_index + 1`` bars, if that is more) and its ``max(horizons)`` label bars sit on one unbroken
bar grid, so no window and no label crosses the overnight gap, a weekend or a missing bar. Where a bar
does not directly follow the previous one (session open, or after a hole), its return is open-to-close
instead of close-to-close and its gap channel is 0, so the overnight move enters nothing. The trailing
statistics (rv20 scale, volume z-score, HAR regressors) do run across sessions, on those intraday
returns only: a session is too short to warm them up from scratch, and they are trailing, so causal.
Intraday rows also carry the time of day (``TAB_TOD``, appended to ``tab``): where in the regular New York
session the forecast is made, taken from the clock at ``known`` and nothing else. Volatility has a
strong shape over the session (high at the open, low at midday, rising into the close), so a model or
a baseline without the clock cannot tell a quiet lunch hour from a calm market. Daily rows have no such
columns: the daily ``tab`` and everything computed from it are unchanged.
On an early-close day (``forecast.data.calendar``) the session is 09:30-13:00, and the time of day is
the share of THAT session: 12:45 on such a day is "15 minutes before the close", as 15:45 is on a full
day. The calendar is published in advance, so this is still the clock alone.
Known gap: horizons are counted in bars (the grid rule makes that equal to clock time).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from forecast.data.calendar import session_seconds_of
from forecast.utils.common import DAY

CHANNELS = ["ret", "abs_ret", "range", "vol_z", "mkt_ret", "gap"]
TAB = ["log_rv5_rv20", "log_rv60_rv20"]  # HAR-style regressors for the baseline
TAB_TOD = ["tod", "tod_u"]  # intraday only: share of the session elapsed, and its U-shape (2 tod - 1)^2
SESSION_OPEN, SESSION_SECONDS = 9 * 3600 + 30 * 60, 23400   # regular New York session, 09:30-16:00
WARMUP = 60
EPS = 1e-8


def grid_runs(tsec: np.ndarray, bar_seconds: int) -> np.ndarray:
    """``run[i]`` = how many consecutive bars, each exactly ``bar_seconds`` after the one before, end
    at bar i (1 at a session open or right after a missing bar)."""
    brk = np.ones(len(tsec), dtype=bool)
    brk[1:] = np.diff(tsec) != bar_seconds
    pos = np.arange(len(tsec))
    return pos - np.maximum.accumulate(np.where(brk, pos, 0)) + 1


def time_of_day(known: np.ndarray) -> np.ndarray:
    """Share of the regular New York session elapsed at epoch second ``known``, in [0, 1]: of 09:30-16:00,
    or of 09:30-13:00 on an early-close day. A function of the clock and the published exchange
    calendar alone (daylight saving included), so it cannot carry anything from a bar."""
    ny = pl.Series("t", np.asarray(known, dtype=np.int64) * 1_000_000).cast(pl.Datetime("us", "UTC"))
    ny = ny.dt.convert_time_zone("America/New_York")
    sec = (ny.dt.hour().cast(pl.Int64) * 3600 + ny.dt.minute().cast(pl.Int64) * 60
           + ny.dt.second().cast(pl.Int64)).to_numpy()
    length = session_seconds_of(ny.dt.date().cast(pl.Int32).to_numpy())
    return np.clip((sec - SESSION_OPEN) / length, 0.0, 1.0)


class Windows:
    """Lazy ``[n, C, L]`` float32 channel windows.

    Holds every symbol's per-bar channels once (``ch`` [N, C], symbols stacked end to end) plus, per
    row, the position in ``ch`` of the window's last bar (``end``). Row ``k`` is
    ``ch[end[k]-L+1 : end[k]+1].T``: bars up to and including bar i, never later ones. Indexing gives
    exactly the array an eager builder would have produced, but only for the rows asked for, so a
    500-symbol universe costs tens of MB instead of ~15 GB. A window never crosses into another symbol
    because every row has at least ``L-1`` bars of its own symbol before it (``build`` guarantees it).
    """

    dtype = np.dtype(np.float32)

    def __init__(self, ch: np.ndarray, end: np.ndarray, lookback: int):
        self.ch, self.end, self.L = ch, end, int(lookback)
        self._off = np.arange(-self.L + 1, 1)

    def __len__(self) -> int:
        return len(self.end)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.end), self.ch.shape[1], self.L)

    @property
    def nbytes(self) -> int:
        """Size if fully materialised (what callers budget memory against)."""
        return int(np.prod(self.shape)) * 4

    def subset(self, idx: np.ndarray) -> Windows:
        """Rows ``idx`` as another lazy view (no window is copied)."""
        return Windows(self.ch, self.end[idx], self.L)

    def take(self, rows: np.ndarray) -> np.ndarray:
        e = np.atleast_1d(self.end[rows])
        return np.ascontiguousarray(self.ch[e[:, None] + self._off].transpose(0, 2, 1))

    def __getitem__(self, key):
        if isinstance(key, tuple):          # e.g. X[..., -32:]: materialise, then plain numpy rules
            return np.asarray(self)[key]
        if isinstance(key, (int, np.integer)):
            return self.take(np.array([key]))[0]
        return self.take(key)

    def __array__(self, dtype=None, copy=None):
        a = self.take(slice(None))
        return a if dtype is None else a.astype(dtype, copy=False)


@dataclass
class Dataset:
    X: Windows           # [n, C, L] float32 channel windows ending at bar i, built on demand
    tab: np.ndarray      # [n, len(tab_names)] float32: TAB, then TAB_TOD on intraday bars
    y: np.ndarray        # [n, H, 2] float32
    t: np.ndarray        # [n] int64 epoch seconds of bar i
    t_end: np.ndarray    # [n] int64 epoch seconds of the last bar any label of row i touches
    sym: np.ndarray      # [n] int32 index into symbols
    scale: np.ndarray    # [n] float32 rv20 at i (multiply by sqrt(h) to undo the target scaling)
    symbols: list[str]
    horizons: list[int]
    known: np.ndarray | None = None  # [n] int64 epoch seconds from which row i may be used: t + one bar
    tab_names: tuple[str, ...] = tuple(TAB)


def _roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last ``w`` values including the current one (shorter at the start)."""
    cs = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(idx - w, 0)
    return (cs[idx] - cs[lo]) / (idx - lo)


def _returns(o: np.ndarray, c: np.ndarray, brk: np.ndarray | None) -> np.ndarray:
    """Close-to-close log returns; at ``brk`` bars (no bar directly before) open-to-close instead."""
    r = np.zeros(len(c))
    r[1:] = np.log(c[1:] / c[:-1])
    if brk is not None:
        r[brk] = np.log(c[brk] / o[brk])
    return r


def symbol_arrays(o, h, low, c, v, mkt_ret,
                  brk: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar channels [n, C], tab features [n, T], log returns r [n], rv20 [n]. All causal.
    ``brk`` (intraday only) marks bars that do not directly follow the previous bar."""
    r = _returns(o, c, brk)
    gap = np.zeros(len(c))
    gap[1:] = np.log(o[1:] / c[:-1])
    if brk is not None:
        gap[brk] = 0.0
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
          max_t_end: int | None = None, first_index: int = 0, bar_seconds: int | None = None) -> Dataset:
    """Build the modelling table. ``max_t_end`` drops every row whose label would touch a bar at or
    after that epoch second; the loop uses it to keep the final holdout physically out of reach.
    ``first_index`` pins the first usable bar so jobs with different lookbacks share identical folds.
    ``bar_seconds`` (intraday bars only; leave None for daily) switches on the unbroken-grid rule and
    the session-open handling described in the module docstring. The grid rule asks for
    ``max(lookback, first_index + 1)`` unbroken bars, so there too ``first_index`` keeps the rows
    identical across lookbacks."""
    hmax = max(horizons)
    intraday = bar_seconds is not None
    symbols = sorted(bars["symbol"].unique().to_list())
    bench = bars.filter(pl.col("symbol") == benchmark).sort("t")
    if bench.height:
        bt = bench["t"].dt.epoch("s").to_numpy()
        br = _returns(bench["open"].to_numpy(), bench["close"].to_numpy(),
                      grid_runs(bt, bar_seconds) == 1 if intraday else None)
        bench = pl.DataFrame({"t": bench["t"], "mkt_ret": br})
    chs, ends, tabs, ys, ts, tes, syms, scs = [], [], [], [], [], [], [], []
    base = 0  # rows of ``ch`` already taken by earlier symbols
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
        tsec = d["t"].dt.epoch("s").to_numpy()
        run = grid_runs(tsec, bar_seconds) if intraday else None
        ch, tab, r, rv20 = symbol_arrays(
            d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(),
            d["volume"].to_numpy(), d["mkt_ret"].to_numpy(), brk=None if run is None else run == 1,
        )
        idx = np.arange(first, n - hmax)
        if run is not None:  # window bars i-L+1..i and label bars i+1..i+hmax on one unbroken grid
            idx = idx[(run[idx] >= max(lookback, first_index + 1)) & (run[idx + hmax] > hmax)]
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
        chs.append(ch.astype(np.float32))
        ends.append(base + idx)
        base += n
        tabs.append(tab[idx].astype(np.float32))
        ys.append(np.clip(y, -8, 8))
        ts.append(tsec[idx])
        tes.append(tsec[idx + hmax])
        syms.append(np.full(len(idx), si, dtype=np.int32))
        scs.append(rv20[idx].astype(np.float32))
    if not chs:
        raise ValueError("no usable rows: not enough history for the lookback/horizons requested")
    X = Windows(np.concatenate(chs), np.concatenate(ends).astype(np.int64), lookback)
    t = np.concatenate(ts)
    # A daily bar's stamp is its date or session open, so "one day later" is a safe time for its close.
    known = t + (bar_seconds or DAY)
    tab, names = np.concatenate(tabs), tuple(TAB)
    if intraday:  # the clock at the moment the row becomes usable; see the module docstring
        tod = time_of_day(known)
        tab = np.concatenate([tab, np.stack([tod, (2.0 * tod - 1.0) ** 2], axis=1).astype(np.float32)], axis=1)
        names = (*TAB, *TAB_TOD)
    return Dataset(X, tab, np.concatenate(ys), t,
                   np.concatenate(tes), np.concatenate(syms), np.concatenate(scs), symbols, list(horizons),
                   known=known, tab_names=names)
