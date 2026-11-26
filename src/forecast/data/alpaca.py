"""Alpaca minute-bar ingestor (historical REST, free plan). Switched on by ``ALPACA_API_KEY`` and
``ALPACA_SECRET_KEY`` in the environment or in the repo's git-ignored ``.env``; without them every
entry point raises ``MissingKeysError`` and the loop keeps using the keyless Yahoo source.

What is on disk (``<DATA_ROOT>/alpaca/1Min/<SYMBOL>/``):

- ``YYYY-MM.parquet``: one finished calendar month of RAW (unadjusted) regular-session bars. Written
  once, atomically, and never requested again. A month with no bars is stored as an empty file so it
  is not asked for twice. These files are the resume manifest: a crash loses at most the month in flight.
- ``YYYY-MM.partial.parquet``: the month that is still open. Each run fetches only what is newer than
  its last bar and rewrites it; it is replaced by the final file once the month has ended.

Raw bars are stored on purpose (docs/DATA_SOURCES.md, pitfalls 1 and 2): a provider-adjusted month goes
stale at the next split or dividend, and a cache mixing months adjusted on different days shows fake
jumps at the seams. ``load`` applies split and dividend adjustment when reading, from one pair of daily
series per symbol (adjusted close / raw close), so the whole history is adjusted as of the same moment.

Conventions and limits, so nobody has to guess:

- ``t`` is the START of the minute (Alpaca's convention), in UTC. The 09:30 ET bar covers
  09:30:00-09:30:59 and is only known at 09:31. Features must treat a bar as known at ``t`` + 1 minute.
- Regular session only: bars starting 09:30 <= New York time < 16:00, Monday to Friday. No exchange
  calendar yet, so on half-days the few after-hours bars between 13:00 and 16:00 are kept. Missing
  minutes are left missing (no forward fill here).
- Free plan: the newest 15 minutes of SIP data are refused, so nothing newer than 16 minutes is asked for.
- Free plan: 200 requests/min; requests are spaced to stay near 170/min, and a 429 backs off.
- ``load`` returns everything in memory: fine for tens of symbols, not for hundreds times ten years.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import polars as pl

from forecast.utils.common import REPO, data_root

BASE = "https://data.alpaca.markets/v2/stocks"
COLS = ["symbol", "t", "open", "high", "low", "close", "volume"]
SCHEMA = {"symbol": pl.String, "t": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64,
          "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
MIN_INTERVAL_S = 0.35          # ~170 requests/min, under the free plan's 200
DELAY = timedelta(minutes=16)  # free plan refuses SIP data newer than 15 minutes
_last_request = 0.0


class MissingKeysError(RuntimeError):
    """No Alpaca keys on this machine."""


def credentials(env_file: Path | None = None) -> tuple[str, str] | None:
    """Keys from the environment, else from ``.env`` in the repo root. Never logged, never returned
    anywhere but the request headers."""
    vals = {k: os.environ.get(k, "") for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
    p = env_file or REPO / ".env"
    if not all(vals.values()) and p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.split("#", 1)[0].partition("=")
            if sep and k.strip() in vals and not vals[k.strip()]:
                vals[k.strip()] = v.strip().strip("\"'")
    if all(vals.values()):
        return vals["ALPACA_API_KEY"], vals["ALPACA_SECRET_KEY"]
    return None


def available() -> bool:
    return credentials() is not None


def _get(path: str, params: dict, creds: tuple[str, str]) -> dict:
    global _last_request
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    headers = {"APCA-API-KEY-ID": creds[0], "APCA-API-SECRET-KEY": creds[1], "Accept": "application/json"}
    last: Exception | None = None
    for attempt in range(6):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403):   # bad keys or a plan limit: retrying cannot help
                raise RuntimeError(f"alpaca refused the request (HTTP {e.code}) for {path}") from e
            if e.code in (400, 404, 422):   # unknown symbol or bad parameters: the answer will not change
                raise ValueError(f"alpaca rejected the request for {path} (HTTP {e.code})") from e
            time.sleep(min(20.0 * (attempt + 1), 65.0) if e.code == 429 else 2.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"alpaca fetch failed for {path}: {last}")


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_bars(symbol: str, rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    df = pl.DataFrame({k: [r[k] for r in rows] for k in ("t", "o", "h", "l", "c", "v")},
                      schema_overrides={k: pl.Float64 for k in ("o", "h", "l", "c", "v")})
    return df.select(
        pl.lit(symbol).alias("symbol"),
        pl.col("t").str.to_datetime(time_zone="UTC", time_unit="us"),
        pl.col("o").alias("open"), pl.col("h").alias("high"), pl.col("l").alias("low"),
        pl.col("c").alias("close"), pl.col("v").alias("volume"),
    ).filter((pl.col("close") > 0) & (pl.col("low") > 0)).unique(subset=["t"], keep="last").sort("t")


def fetch_bars(symbol: str, start: datetime, end: datetime, creds: tuple[str, str],
               timeframe: str = "1Min", adjustment: str = "raw") -> pl.DataFrame:
    """All bars with ``start <= t <= end``, following ``next_page_token`` until it runs out."""
    rows: list[dict] = []
    token = None
    while True:
        params = {"timeframe": timeframe, "start": _iso(start), "end": _iso(end), "limit": 10000,
                  "adjustment": adjustment, "feed": "sip", "sort": "asc"}
        if token:
            params["page_token"] = token
        page = _get(f"{symbol}/bars", params, creds)
        rows += page.get("bars") or []
        token = page.get("next_page_token")
        if not token:
            return parse_bars(symbol, rows)


def regular_session(df: pl.DataFrame) -> pl.DataFrame:
    """Keep bars that start inside 09:30-16:00 New York time on a weekday (see module docstring)."""
    ny = pl.col("t").dt.convert_time_zone("America/New_York")
    return df.filter((ny.dt.time() >= dtime(9, 30)) & (ny.dt.time() < dtime(16, 0)) & (ny.dt.weekday() <= 5))


def month_starts(start: datetime, end: datetime) -> list[datetime]:
    out, cur = [], datetime(start.year, start.month, 1, tzinfo=UTC)
    while cur <= end:
        out.append(cur)
        cur = datetime(cur.year + cur.month // 12, cur.month % 12 + 1, 1, tzinfo=UTC)
    return out


def _write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.select(COLS).write_parquet(tmp)
    os.replace(tmp, path)


def ingest(symbols: list[str], start: datetime, now: datetime | None = None,
           creds: tuple[str, str] | None = None) -> dict:
    """Bring the raw minute store up to date. Safe to interrupt and re-run: finished months are skipped,
    the open month is extended from its last bar. Returns counts for the log (no data, no keys)."""
    creds = creds or credentials()
    if creds is None:
        raise MissingKeysError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (environment or .env)")
    cutoff = ((now or datetime.now(UTC)) - DELAY).replace(second=0, microsecond=0)
    start = start.astimezone(UTC)
    stats = {"months_fetched": 0, "months_skipped": 0, "bars": 0, "failed": []}
    for s in symbols:
        folder = data_root() / "alpaca" / "1Min" / s
        folder.mkdir(parents=True, exist_ok=True)
        for m0 in month_starts(start, cutoff):
            m1 = datetime(m0.year + m0.month // 12, m0.month % 12 + 1, 1, tzinfo=UTC)
            final, partial = folder / f"{m0:%Y-%m}.parquet", folder / f"{m0:%Y-%m}.partial.parquet"
            if final.exists():
                stats["months_skipped"] += 1
                continue
            have = pl.read_parquet(partial) if partial.exists() else pl.DataFrame(schema=SCHEMA)
            lo = max(m0, start)
            if have.height:
                lo = max(lo, have["t"].max() + timedelta(minutes=1))
            hi = min(m1 - timedelta(minutes=1), cutoff)
            try:
                new = regular_session(fetch_bars(s, lo, hi, creds)) if lo <= hi else have.clear()
            except Exception as e:  # keep what is on disk, carry on with the next symbol
                stats["failed"].append(f"{s} {m0:%Y-%m}: {type(e).__name__}: {e}")
                break
            df = pl.concat([have, new]).unique(subset=["t"], keep="last").sort("t")
            if m1 - timedelta(minutes=1) <= cutoff:   # the month is over: freeze it
                _write(df, final)
                partial.unlink(missing_ok=True)
            else:
                _write(df, partial)
            stats["months_fetched"] += 1
            stats["bars"] += new.height
    return stats


def adjustment_factors(symbol: str, start: datetime, end: datetime, creds: tuple[str, str]) -> pl.DataFrame:
    """Per New York calendar day: adjusted close / raw close, both from Alpaca daily bars fetched now."""
    def daily(adjustment: str, name: str) -> pl.DataFrame:
        d = fetch_bars(symbol, start, end, creds, timeframe="1Day", adjustment=adjustment)
        return d.select(pl.col("t").dt.convert_time_zone("America/New_York").dt.date().alias("day"),
                        pl.col("close").alias(name))
    j = daily("raw", "raw").join(daily("all", "adj"), on="day", how="inner")
    return j.select("day", (pl.col("adj") / pl.col("raw")).alias("factor")).sort("day")


def adjust(bars: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """Scale raw OHLC by that day's factor. Days newer than the last daily bar use the latest factor
    (it is 1.0 unless an ex-date falls in between); volume is left as traded."""
    if not bars.height:
        return bars
    day = pl.col("t").dt.convert_time_zone("America/New_York").dt.date()
    out = bars.with_columns(day.alias("day")).sort("day").join_asof(factors, on="day", strategy="backward")
    f = pl.col("factor").fill_null(1.0)
    return out.with_columns([(pl.col(c) * f).alias(c) for c in ("open", "high", "low", "close")]
                            ).select(COLS).sort(["symbol", "t"])


def stored(symbol: str, start: date | None = None) -> pl.DataFrame:
    """Raw bars already on disk for one symbol (finished months and the open one)."""
    folder = data_root() / "alpaca" / "1Min" / symbol
    files = sorted(p for p in folder.glob("*.parquet") if start is None or p.name[:7] >= f"{start:%Y-%m}")
    frames = [pl.read_parquet(p) for p in files]
    frames = [f for f in frames if f.height]
    if not frames:
        return pl.DataFrame(schema=SCHEMA)
    return pl.concat(frames).unique(subset=["t"], keep="last").sort("t")


def load(symbols: list[str], start: datetime, now: datetime | None = None) -> pl.DataFrame:
    """Update the store, then return adjusted regular-session minute bars from ``start`` on."""
    creds = credentials()
    if creds is None:
        raise MissingKeysError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (environment or .env)")
    now = now or datetime.now(UTC)
    ingest(symbols, start, now, creds)
    frames = []
    for s in symbols:
        raw = stored(s, start.date()).filter(pl.col("t") >= start)
        if raw.height:
            frames.append(adjust(raw, adjustment_factors(s, start - timedelta(days=7), now - DELAY, creds)))
    if len(frames) < max(2, len(symbols) // 2):
        raise RuntimeError(f"too few symbols with alpaca minute bars ({len(frames)}/{len(symbols)})")
    return pl.concat(frames).select(COLS).sort(["symbol", "t"])


DAILY_FLOOR = datetime(2016, 1, 4, tzinfo=UTC)   # Alpaca's history starts in 2016


def daily(symbols: list[str], start: datetime, now: datetime | None = None, max_age_h: float = 20.0,
          end: datetime | None = None) -> pl.DataFrame:
    """Split/dividend-adjusted daily bars, one parquet per symbol under ``<DATA_ROOT>/alpaca/1Day/``.

    Unlike the minute store, a stale file is refetched whole with ``adjustment=all``: daily history is a
    few thousand rows, and one fetch keeps the whole series adjusted as of the same moment.

    ``end`` (exclusive; the loop passes the midnight after ``data.end``) pins the data: later bars are
    dropped, and a file written after ``end`` is complete and is never refetched, so every job sees the
    same bars and the same folds until the pin is moved. Without it a file is refetched after
    ``max_age_h`` and the folds drift by a bar a day.

    Symbols are asked for in Alpaca's spelling (``BRK.B`` for the list's ``BRK-B``) and returned under the
    list's name. A symbol Alpaca does not know, or has no bars for, gets a ``<symbol>.none`` marker next
    to the parquet files and is not asked for again while a parquet written at that time would still be
    in date; otherwise every job would spend its first seconds re-asking for the same dead symbols. A
    network error or rate limit leaves no marker: that symbol is tried again by the next job."""
    creds = credentials()
    if creds is None:
        raise MissingKeysError("no ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment or .env")
    now = now or datetime.now(UTC)
    start = max(start, DAILY_FLOOR)
    pinned = end is not None and end <= now - DELAY
    cache = data_root() / "alpaca" / "1Day"
    cache.mkdir(parents=True, exist_ok=True)

    def in_date(path: Path) -> bool:
        if not path.exists():
            return False
        mtime = path.stat().st_mtime
        return mtime >= end.timestamp() if pinned else (time.time() - mtime) < max_age_h * 3600

    frames, failed = [], []
    for s in symbols:
        p, none = cache / f"{s}.parquet", cache / f"{s}.none"
        if not in_date(p) and not in_date(none):
            try:
                df = fetch_bars(s.replace("-", "."), DAILY_FLOOR, now - DELAY, creds, timeframe="1Day",
                                adjustment="all").with_columns(pl.lit(s).alias("symbol"))
                if df.height:
                    _write(df, p)
                else:
                    none.touch()
            except ValueError as e:  # Alpaca does not know the symbol
                none.touch()
                failed.append(f"{s}: {e}")
            except Exception as e:  # keep going with the stale copy if there is one
                failed.append(f"{s}: {e}")
        if p.exists():
            frames.append(pl.read_parquet(p))
    if len(frames) < max(2, len(symbols) // 2):
        raise RuntimeError(f"too few symbols loaded ({len(frames)}/{len(symbols)}); errors: {failed[:3]}")
    out = pl.concat(frames).select(COLS).filter(pl.col("t") >= start)
    if end is not None:
        out = out.filter(pl.col("t") < end)
    return out.sort(["symbol", "t"])
