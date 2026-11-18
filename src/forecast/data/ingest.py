"""Resumable, idempotent historical bar ingest from Alpaca, chunked by (symbol, month).

Two layers are pulled: `raw` (adjustment=raw) and `adjusted` (adjustment=all). Features use `adjusted`;
`raw` is kept so quality.py can cross-check split/dividend handling (Alpaca has known CA errors).
Alpaca bar timestamps are the bar OPEN time (asserted in tests/data against a provider sample).
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from forecast.data.store import BAR_SCHEMA, partition_path, write_partition
from forecast.utils import env
from forecast.utils.timeutil import now_utc

LAYERS = {"raw": "raw", "adjusted": "all"}
MIN_SECONDS_BETWEEN_CALLS = 0.35  # ~170 calls/min, under the 200/min free-tier limit


def month_chunks(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    cur = datetime(start.year, start.month, 1, tzinfo=UTC)
    while cur < end:
        nxt = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=UTC)
        yield max(cur, start), min(nxt, end)
        cur = nxt


class Manifest:
    """Append-only JSON-lines manifest of completed chunks."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.done.add(json.loads(line)["key"])

    @staticmethod
    def key(layer: str, tf: str, sym: str, y: int, m: int) -> str:
        return f"{layer}|{tf}|{sym}|{y}-{m:02d}"

    def mark(self, key: str, rows: int) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "rows": rows, "at": now_utc().isoformat()}) + "\n")
        self.done.add(key)


def _alpaca_fetch(client, symbol: str, tf: str, start: datetime, end: datetime,
                  adjustment: str, feed: str) -> pl.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tfmap = {"1Min": TimeFrame(1, TimeFrameUnit.Minute), "5Min": TimeFrame(5, TimeFrameUnit.Minute),
             "1Hour": TimeFrame(1, TimeFrameUnit.Hour), "1Day": TimeFrame(1, TimeFrameUnit.Day)}
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tfmap[tf], start=start, end=end,
                           adjustment=Adjustment(adjustment), feed=DataFeed(feed))
    bars = client.get_stock_bars(req)
    pdf = bars.df
    if pdf is None or len(pdf) == 0:
        return pl.DataFrame(schema=BAR_SCHEMA)
    pdf = pdf.reset_index()
    df = pl.from_pandas(pdf).rename({"timestamp": "ts"})
    return df.select([pl.col(c).cast(t) if c in df.columns else pl.lit(None, t).alias(c)
                      for c, t in BAR_SCHEMA.items()])


def ingest(symbols: list[str], start: datetime, end: datetime | None, timeframe: str,
           feed: str = "sip", layers: tuple[str, ...] = ("raw", "adjusted"),
           root: Path | None = None, fetch=None, client=None) -> dict[str, int]:
    """Pull every (layer, symbol, month) chunk not already in the manifest.

    The current (incomplete) month is always re-pulled. `fetch` is injectable for tests.
    """
    root = root or env.data_root()
    end = end or now_utc()
    manifest = Manifest(root / "manifests" / "ingest_manifest.jsonl")
    if fetch is None:
        from alpaca.data.historical import StockHistoricalDataClient
        client = client or StockHistoricalDataClient(
            env.get("ALPACA_API_KEY", required=True), env.get("ALPACA_SECRET_KEY", required=True))
        fetch = lambda *a: _alpaca_fetch(client, *a)  # noqa: E731
    stats = {"pulled": 0, "skipped": 0, "rows": 0}
    this_month = (now_utc().year, now_utc().month)
    last_call = 0.0
    for sym in symbols:
        for c0, c1 in month_chunks(start, end):
            for layer in layers:
                key = Manifest.key(layer, timeframe, sym, c0.year, c0.month)
                if key in manifest.done and (c0.year, c0.month) != this_month:
                    stats["skipped"] += 1
                    continue
                wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - last_call)
                if wait > 0:
                    time.sleep(wait)
                last_call = time.monotonic()
                df = fetch(sym, timeframe, c0, c1, LAYERS[layer], feed)
                write_partition(df, partition_path(layer, timeframe, sym, c0.year, c0.month, root))
                manifest.mark(key, df.height)
                stats["pulled"] += 1
                stats["rows"] += df.height
            print(f"[ingest] {sym} {c0:%Y-%m} done", flush=True)
    return stats
