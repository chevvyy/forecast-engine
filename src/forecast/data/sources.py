"""Bar sources. Every source returns a Polars frame with columns
``symbol, t (UTC datetime), open, high, low, close, volume`` sorted by (symbol, t).

- ``synthetic``: seeded GARCH-like bars for tests and smoke jobs. Never used for reportable results.
- ``yahoo``: keyless split/dividend-adjusted bars (daily back decades, 1h back ~730 days).
  Good enough to start the loop without accounts. Minute history needs Alpaca (see docs/DATA_SOURCES.md);
  that ingestor is on the loop backlog and switches on when ALPACA_API_KEY is present.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from forecast.utils.common import data_root

COLS = ["symbol", "t", "open", "high", "low", "close", "volume"]
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def synthetic(symbols: list[str], n_bars: int = 1500, seed: int = 0) -> pl.DataFrame:
    """GARCH(1,1)-style returns with a shared market factor. Volatility clusters, so realized vol is
    partly predictable while returns are (almost) not: the same shape the real problem has."""
    rng = np.random.default_rng(seed)
    start = int(datetime(2015, 1, 1, tzinfo=UTC).timestamp())
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


def _yahoo_fetch(symbol: str, interval: str) -> dict:
    rng = "max" if interval == "1d" else "730d"
    last: Exception | None = None
    for attempt in range(5):
        host = "query1" if attempt % 2 == 0 else "query2"
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?range={rng}&interval={interval}&includePrePost=false&events=div%2Csplits")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yahoo fetch failed for {symbol}: {last}")


def yahoo(symbols: list[str], interval: str = "1d", max_age_h: float = 20.0) -> pl.DataFrame:
    """Download (or reuse a fresh cached copy of) adjusted bars for each symbol."""
    cache = data_root() / "yahoo" / interval
    cache.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []
    for s in symbols:
        p: Path = cache / f"{s}.parquet"
        fresh = p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600
        if not fresh:
            try:
                parse_yahoo_chart(s, _yahoo_fetch(s, interval)).write_parquet(p)
                time.sleep(0.4)  # stay polite; this endpoint rate-limits bursts
            except Exception as e:  # keep going with the stale copy if there is one
                failed.append(f"{s}: {e}")
        if p.exists():
            frames.append(pl.read_parquet(p))
    if len(frames) < max(2, len(symbols) // 2):
        raise RuntimeError(f"too few symbols loaded ({len(frames)}/{len(symbols)}); errors: {failed[:3]}")
    return pl.concat(frames).select(COLS).sort(["symbol", "t"])


def load_bars(cfg: dict) -> pl.DataFrame:
    src = cfg.get("source", "yahoo")
    symbols = list(cfg["symbols"])
    if src == "synthetic":
        return synthetic(symbols, int(cfg.get("n_bars", 1500)), int(cfg.get("seed", 0)))
    if src == "yahoo":
        df = yahoo(symbols, cfg.get("interval", "1d"))
        if cfg.get("start"):
            df = df.filter(pl.col("t") >= datetime.fromisoformat(str(cfg["start"])).replace(tzinfo=UTC))
        return df
    raise ValueError(f"unknown data source {src!r}")
