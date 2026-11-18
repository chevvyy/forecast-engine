"""Long/short top-k backtest. Signal at bar close t, fill at OPEN of t+1, exit at close of t+h.
Non-overlapping rebalances every h bars. Costs charged per side on traded weight."""
from __future__ import annotations

import math

import numpy as np
import polars as pl


def backtest(pred: pl.DataFrame, bars: pl.DataFrame, h: int, top_k: int = 5,
             cost_bps: float = 2.0, bars_per_year: float = 252 * 390) -> dict:
    """pred: symbol, ts, score. bars: symbol, ts, open, close (aligned grid)."""
    px = bars.sort(["symbol", "ts"]).with_columns(
        entry=pl.col("open").shift(-1).over("symbol"),
        exit=pl.col("close").shift(-h).over("symbol"),
        entry_ts=pl.col("ts").shift(-1).over("symbol"),
    ).select("symbol", "ts", "entry", "exit", "entry_ts")
    df = pred.join(px, on=["symbol", "ts"], how="inner").drop_nulls(["entry", "exit"])
    times = df["ts"].unique().sort()
    times = times.gather(np.arange(0, len(times), max(h, 1)))
    df = df.filter(pl.col("ts").is_in(times.implode()))
    df = df.with_columns(rk=pl.col("score").rank("ordinal").over("ts"), n=pl.len().over("ts"))
    df = df.filter(pl.col("n") >= 2 * top_k).with_columns(
        w=pl.when(pl.col("rk") > pl.col("n") - top_k).then(1.0 / top_k)
        .when(pl.col("rk") <= top_k).then(-1.0 / top_k).otherwise(0.0),
        r=(pl.col("exit") / pl.col("entry")).log(),
    )
    assert (df["entry_ts"] > df["ts"]).all(), "fill must be strictly after signal bar"
    per = df.group_by("ts").agg(gross=(pl.col("w") * pl.col("r")).sum(),
                                turnover=pl.col("w").abs().sum()).sort("ts")
    # full in/out each rebalance: entry + exit sides
    per = per.with_columns(net=pl.col("gross") - 2 * pl.col("turnover") * cost_bps / 1e4)
    if per.height < 2:
        return {"n_periods": per.height}
    ann = bars_per_year / max(h, 1)
    out = {"n_periods": per.height}
    for col in ("gross", "net"):
        x = per[col].to_numpy()
        eq = np.cumsum(x)
        out[f"{col}_ann_return"] = float(x.mean() * ann)
        out[f"{col}_ann_vol"] = float(x.std() * math.sqrt(ann))
        out[f"{col}_sharpe"] = float(x.mean() / (x.std() + 1e-12) * math.sqrt(ann))
        out[f"{col}_max_dd"] = float((eq - np.maximum.accumulate(eq)).min())
    out["sharpe_per_period"] = float(per["net"].mean() / (per["net"].std() + 1e-12))
    out["turnover_per_rebalance"] = float(per["turnover"].mean())
    return out
