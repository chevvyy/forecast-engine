"""Bar sources. Every source returns a Polars frame with columns
``symbol, t (UTC datetime), open, high, low, close, volume`` sorted by (symbol, t).

- ``synthetic``: seeded GARCH-like bars for tests and smoke jobs. Never used for reportable results.
- ``yahoo``: keyless split/dividend-adjusted bars (daily back decades, 1h back ~730 days).
  Good enough to start the loop without accounts.
- ``auto``: daily bars from Alpaca when this machine has keys, else from Yahoo.
- ``alpaca``: minute bars (``forecast.data.alpaca``); needs ALPACA_API_KEY / ALPACA_SECRET_KEY in ``.env``.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from forecast.utils.common import REPO, data_root, interval_seconds

COLS = ["symbol", "t", "open", "high", "low", "close", "volume"]
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def _session_starts(n_bars: int, bar_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Bar start times (epoch s) on weekday 14:30-21:00 UTC sessions, and a mask of session-open bars."""
    per = max(23400 // bar_seconds, 1)
    day = int(datetime(2024, 1, 1, 14, 30, tzinfo=UTC).timestamp())  # a Monday
    out: list[int] = []
    while len(out) < n_bars:
        if datetime.fromtimestamp(day, tz=UTC).weekday() < 5:
            out.extend(day + bar_seconds * k for k in range(per))
        day += 86400
    t = np.array(out[:n_bars], dtype=np.int64)
    return t, np.arange(n_bars) % per == 0


def synthetic(symbols: list[str], n_bars: int = 1500, seed: int = 0, bar_seconds: int = 86400) -> pl.DataFrame:
    """GARCH(1,1)-style returns with a shared market factor. Volatility clusters, so realized vol is
    partly predictable while returns are (almost) not: the same shape the real problem has.
    With ``bar_seconds`` below a day the bars sit on weekday sessions and every session opens with an
    overnight jump, so intraday code paths meet the gaps they must not cross."""
    rng = np.random.default_rng(seed)
    start = int(datetime(2015, 1, 1, tzinfo=UTC).timestamp())
    intraday = bar_seconds < 86400
    if intraday:
        tsec, opens = _session_starts(n_bars, bar_seconds)
        ts = [datetime.fromtimestamp(int(x), tz=UTC) for x in tsec]
    else:
        ts = [datetime.fromtimestamp(start + 86400 * i, tz=UTC) for i in range(n_bars)]
    mkt = rng.standard_normal(n_bars)
    frames = []
    for s in symbols:
        eps = 0.6 * mkt + 0.8 * rng.standard_normal(n_bars)
        var = np.empty(n_bars)
        r = np.empty(n_bars)
        v = 1e-4
        for i in range(n_bars):
            var[i] = v
            r[i] = np.sqrt(v) * eps[i]
            v = 4e-6 + 0.10 * r[i] ** 2 + 0.86 * v
        if intraday:  # overnight jump before each session's first bar, several bars' worth of move
            jump = np.where(opens, 0.01 * rng.standard_normal(n_bars), 0.0)
            close = 100.0 * np.exp(np.cumsum(r + jump))
            open_ = close / np.exp(r)
        else:
            close = 100.0 * np.exp(np.cumsum(r))
            open_ = np.concatenate([[100.0], close[:-1]])
        spread = np.abs(rng.standard_normal(n_bars)) * np.sqrt(var) * 0.5
        high = np.maximum(open_, close) * np.exp(spread)
        low = np.minimum(open_, close) * np.exp(-spread)
        volume = np.exp(13 + 0.5 * rng.standard_normal(n_bars) + 20 * np.sqrt(var))
        frames.append(
            pl.DataFrame(
                {"symbol": s, "t": ts, "open": open_, "high": high, "low": low, "close": close,
                 "volume": volume}
            )
        )
    return pl.concat(frames).with_columns(pl.col("t").cast(pl.Datetime("us", "UTC")))


def parse_yahoo_chart(symbol: str, payload: dict) -> pl.DataFrame:
    """Turn a Yahoo v8 chart response into adjusted bars. Rows with any missing OHLC are dropped."""
    res = payload["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pl.DataFrame(
        {
            "ts": ts,
            "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
            "volume": q["volume"],
            "adj": adj if adj is not None else q["close"],
        },
        schema_overrides={c: pl.Float64 for c in ["open", "high", "low", "close", "volume", "adj"]},
    ).drop_nulls(["open", "high", "low", "close", "adj"])
    df = df.filter((pl.col("close") > 0) & (pl.col("low") > 0))
    f = pl.col("adj") / pl.col("close")
    return df.select(
        pl.lit(symbol).alias("symbol"),
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC")).alias("t"),
        (pl.col("open") * f).alias("open"),
        (pl.col("high") * f).alias("high"),
        (pl.col("low") * f).alias("low"),
        pl.col("adj").alias("close"),
        pl.col("volume").fill_null(0.0),
    ).unique(subset=["t"], keep="last").sort("t")


class RateLimitedError(RuntimeError):
    """Yahoo kept answering 429 after backing off."""


class UnknownSymbolError(RuntimeError):
    """Yahoo has no such symbol (renamed or delisted). Not worth retrying."""


_SYMBOL_RE = re.compile(r"[A-Z0-9^][A-Z0-9.\-=^]{0,11}")


def read_symbols(path: str | Path) -> list[str]:
    """Read a symbol list: one symbol per line, ``#`` comments, blanks ignored, duplicates dropped
    (first occurrence wins). A relative path is resolved against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    out: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.split("#", 1)[0].strip().upper()
        if not s:
            continue
        if not _SYMBOL_RE.fullmatch(s):
            raise ValueError(f"{p.name}:{n}: not a symbol: {s!r}")
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve_symbols(cfg: dict) -> list[str]:
    """``symbols`` (inline) first, then ``symbols_file``; either or both may be given."""
    out = [str(s).upper() for s in (cfg.get("symbols") or [])]
    if cfg.get("symbols_file"):
        out += read_symbols(cfg["symbols_file"])
    out = list(dict.fromkeys(out))
    if not out:
        raise ValueError("data config names no symbols (symbols / symbols_file)")
    return out


def _yahoo_fetch(symbol: str, interval: str) -> dict:
    rng = "max" if interval == "1d" else "730d"
    last: Exception | None = None
    limited = 0
    for attempt in range(5):
        host = "query1" if attempt % 2 == 0 else "query2"
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?range={rng}&interval={interval}&includePrePost=false&events=div%2Csplits")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                raise UnknownSymbolError(f"yahoo has no symbol {symbol}") from e
            if e.code == 429:  # rate limited: wait much longer than for a network blip
                limited += 1
                try:
                    wait = float(e.headers.get("Retry-After", "")) if e.headers else 0.0
                except ValueError:
                    wait = 0.0
                time.sleep(min(max(wait, 15.0 * 2 ** (limited - 1)), 120.0))
            else:
                time.sleep(2.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    if limited >= 3:
        raise RateLimitedError(f"yahoo rate limit for {symbol}: {last}")
    raise RuntimeError(f"yahoo fetch failed for {symbol}: {last}")


def yahoo(symbols: list[str], interval: str = "1d", max_age_h: float = 20.0) -> pl.DataFrame:
    """Download (or reuse a fresh cached copy of) adjusted bars for each symbol."""
    cache = data_root() / "yahoo" / interval
    cache.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []
    blocked = False  # once Yahoo keeps answering 429, stop asking for the rest of this call
    for s in symbols:
        p: Path = cache / f"{s}.parquet"
        fresh = p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600
        if not fresh and not blocked:
            try:
                parse_yahoo_chart(s, _yahoo_fetch(s, interval)).write_parquet(p)
                time.sleep(0.4)  # stay polite; this endpoint rate-limits bursts
            except RateLimitedError as e:  # the remaining symbols use whatever the cache has
                blocked = True
                failed.append(f"{s}: {e}")
            except Exception as e:  # keep going with the stale copy if there is one
                failed.append(f"{s}: {e}")
        if p.exists():
            frames.append(pl.read_parquet(p))
    if len(frames) < max(2, len(symbols) // 2):
        raise RuntimeError(f"too few symbols loaded ({len(frames)}/{len(symbols)}); errors: {failed[:3]}")
    return pl.concat(frames).select(COLS).sort(["symbol", "t"])


def end_exclusive(cfg: dict) -> datetime | None:
    """``data.end`` (a date, inclusive) as the UTC midnight after it, or None when the data is not pinned.
    A daily bar is stamped with its own date (Alpaca: New York midnight, Yahoo: the session open), so
    ``t < end_exclusive`` keeps the bar of ``data.end`` and nothing later."""
    if not cfg.get("end"):
        return None
    return datetime.fromisoformat(str(cfg["end"])).replace(tzinfo=UTC, hour=0, minute=0, second=0,
                                                           microsecond=0) + timedelta(days=1)


def load_bars(cfg: dict) -> pl.DataFrame:
    """Bars for ``cfg`` (the ``data`` block of a loop config). ``end`` pins the last bar for every source,
    so new bars arriving day by day cannot move the folds under a running comparison."""
    end = end_exclusive(cfg)
    df = _load_bars(cfg, end)
    return df if end is None else df.filter(pl.col("t") < end)


def _load_bars(cfg: dict, end: datetime | None) -> pl.DataFrame:
    src = cfg.get("source", "yahoo")
    symbols = resolve_symbols(cfg)
    if src == "synthetic":
        return synthetic(symbols, int(cfg.get("n_bars", 1500)), int(cfg.get("seed", 0)),
                         interval_seconds(cfg.get("interval", "1d")))
    start = datetime.fromisoformat(str(cfg["start"])).replace(tzinfo=UTC) if cfg.get("start") else None
    if src == "auto":
        # Daily bars from whichever source this machine can use: Alpaca when keys are present (a keyed
        # API meant for programs), otherwise keyless Yahoo, which may refuse scripted clients with 429.
        from forecast.data import alpaca

        if interval_seconds(str(cfg.get("interval", "1d"))) != 86400:
            raise ValueError("data.source auto is for daily bars; use alpaca for minute bars")
        if alpaca.available():
            return alpaca.daily(symbols, start or alpaca.DAILY_FLOOR, end=end)
        src = "yahoo"
    if src == "yahoo":
        df = yahoo(symbols, cfg.get("interval", "1d"))
        if cfg.get("start"):
            df = df.filter(pl.col("t") >= datetime.fromisoformat(str(cfg["start"])).replace(tzinfo=UTC))
        return df
    if src == "alpaca":
        from forecast.data import alpaca

        if not cfg.get("start"):
            raise ValueError("data.start is required for the alpaca source (minute history is large)")
        return alpaca.load(symbols, datetime.fromisoformat(str(cfg["start"])).replace(tzinfo=UTC), end=end)
    raise ValueError(f"unknown data source {src!r}")
