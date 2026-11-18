"""Partitioned Parquet store: <root>/<layer>/bars/timeframe=/symbol=/year=/month=/part-0.parquet."""
from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from forecast.utils.env import data_root

BAR_SCHEMA = {
    "symbol": pl.Utf8,
    "ts": pl.Datetime("ns", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trade_count": pl.Float64,
    "vwap": pl.Float64,
}


def bars_dir(layer: str, timeframe: str, root: Path | None = None) -> Path:
    return (root or data_root()) / layer / "bars" / f"timeframe={timeframe}"


def partition_path(layer: str, timeframe: str, symbol: str, year: int, month: int,
                   root: Path | None = None) -> Path:
    return (bars_dir(layer, timeframe, root) / f"symbol={symbol}" / f"year={year}"
            / f"month={month:02d}" / "part-0.parquet")


def write_partition(df: pl.DataFrame, path: Path) -> None:
    """Atomic write (tmp + rename) so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.sort("ts").write_parquet(tmp)
    tmp.replace(path)


def read_bars(layer: str, timeframe: str, symbols: list[str] | None = None,
              root: Path | None = None) -> pl.DataFrame:
    base = bars_dir(layer, timeframe, root)
    files = sorted(base.glob("symbol=*/year=*/month=*/part-0.parquet"))
    if symbols is not None:
        keep = {f"symbol={s}" for s in symbols}
        files = [f for f in files if f.parts[-4] in keep]
    if not files:
        raise FileNotFoundError(f"No bars under {base} (did you run fe-ingest?)")
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.cast({k: v for k, v in BAR_SCHEMA.items() if k in df.columns}).sort(["symbol", "ts"])


def snapshot_hash(layer: str, timeframe: str, root: Path | None = None) -> str:
    """Hash of (path, size, mtime) of every partition: cheap and changes when data changes."""
    h = hashlib.sha256()
    for f in sorted(bars_dir(layer, timeframe, root).rglob("*.parquet")):
        st = f.stat()
        h.update(f"{f.relative_to(data_root() if root is None else root)}|{st.st_size}".encode())
    return h.hexdigest()[:12]
