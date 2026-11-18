"""Exchange calendar + regular-session minute grid with is_filled flag."""
from __future__ import annotations

from datetime import datetime

import exchange_calendars as xcals
import polars as pl


def session_minutes(start: datetime, end: datetime, calendar: str = "XNYS") -> pl.DataFrame:
    """All regular-session minute OPEN times (UTC) with their session date. Half-days included."""
    cal = xcals.get_calendar(calendar, start=start.date().isoformat())
    sessions = cal.sessions_in_range(start.date().isoformat(), end.date().isoformat())
    mins = cal.sessions_minutes(sessions[0], sessions[-1])  # minute labels = bar open times, UTC
    sess = cal.minutes_to_sessions(mins)
    return pl.DataFrame({
        "ts": pl.Series(mins.tz_convert(None).to_numpy()).cast(pl.Datetime("ns"))
        .dt.replace_time_zone("UTC"),
        "session": pl.Series(sess.to_numpy()).cast(pl.Date),
    })


def session_align(bars: pl.DataFrame, timeframe: str = "1Min", calendar: str = "XNYS") -> pl.DataFrame:
    """Restrict to regular session and reindex each symbol to the full minute grid.

    Gaps: close/vwap forward-filled, open=high=low=close, volume=0, is_filled=1. Daily bars only get a
    session column and is_filled=0.
    """
    if timeframe == "1Day":
        return bars.with_columns(session=pl.col("ts").dt.convert_time_zone("America/New_York").dt.date(),
                                 is_filled=pl.lit(0, pl.Int8))
    if timeframe != "1Min":
        raise NotImplementedError("align supports 1Min and 1Day")
    grid = session_minutes(bars["ts"].min(), bars["ts"].max(), calendar)
    out = []
    for sym, g in bars.group_by("symbol", maintain_order=True):
        g = g.drop("symbol")
        first = g["ts"].min()
        sg = grid.filter(pl.col("ts") >= first).join(g, on="ts", how="left")
        filled = sg["close"].is_null()
        sg = sg.with_columns(
            is_filled=filled.cast(pl.Int8),
            close=pl.col("close").forward_fill(),
            vwap=pl.col("vwap").forward_fill(),
        ).with_columns(
            open=pl.col("open").fill_null(pl.col("close")),
            high=pl.col("high").fill_null(pl.col("close")),
            low=pl.col("low").fill_null(pl.col("close")),
            volume=pl.col("volume").fill_null(0.0),
            trade_count=pl.col("trade_count").fill_null(0.0),
            symbol=pl.lit(sym[0]),
        ).drop_nulls("close")
        out.append(sg)
    return pl.concat(out, how="diagonal_relaxed").sort(["symbol", "ts"])
