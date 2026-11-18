"""Data-quality report."""
from __future__ import annotations

import polars as pl


def dq_report(bars: pl.DataFrame, raw: pl.DataFrame | None = None, jump_sigma: float = 15.0) -> pl.DataFrame:
    r = pl.col("close").log().diff().over("symbol")
    df = bars.with_columns(r=r)
    rep = df.group_by("symbol").agg(
        rows=pl.len(),
        start=pl.col("ts").min(),
        end=pl.col("ts").max(),
        ohlc_violations=((pl.col("low") > pl.min_horizontal("open", "close"))
                         | (pl.col("high") < pl.max_horizontal("open", "close"))).sum(),
        nonpositive_prices=(pl.col("close") <= 0).sum(),
        filled_frac=(pl.col("is_filled").mean() if "is_filled" in df.columns else pl.lit(None)),
        extreme_returns=(pl.col("r").abs() > jump_sigma * pl.col("r").std()).sum(),
    )
    if raw is not None:
        # Days where raw vs adjusted ratio jumps = corporate action days; listed so they can be checked.
        j = bars.select("symbol", "ts", adj="close").join(raw.select("symbol", "ts", raw="close"),
                                                         on=["symbol", "ts"])
        ca = (j.with_columns(ratio=(pl.col("adj") / pl.col("raw")).log())
              .with_columns(d=pl.col("ratio").diff().abs().over("symbol"))
              .group_by("symbol").agg(ca_events=(pl.col("d") > 0.01).sum()))
        rep = rep.join(ca, on="symbol", how="left")
    return rep.sort("symbol")
