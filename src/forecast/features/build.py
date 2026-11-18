"""Feature + label construction.

Time convention (asserted by tests/features leakage tests):
  * Row i is the bar whose OPEN time is ts[i]; its close is known at decision time d[i] = ts[i] + bar.
  * Features at row i use bars <= i only ("as of bar close i").
  * Labels use bars i+1 .. i+h: forward log return close[i]->close[i+h] and realized vol of those h
    1-bar returns. label_start = d[i], label_end = d[i+h]. Intraday labels never cross a session close.
Window sizes in configs are minutes and are converted to bars with the timeframe's bar length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

import polars as pl

from forecast.utils.timeutil import TIMEFRAME_MINUTES

C = pl.col


@dataclass
class FeatureSet:
    df: pl.DataFrame               # meta + features + labels
    feature_cols: list[str]
    label_cols: dict[str, str]     # task name -> column, e.g. "ret_15" -> "y_ret_15"
    groups: dict[str, list[str]] = field(default_factory=dict)
    max_lookback_min: int = 0

    META = ("symbol", "ts", "session", "label_start", "label_end")


def _bars(minutes: int, bar_min: int) -> int:
    return max(1, int(round(minutes / bar_min)))


def _roll(expr: pl.Expr, w: int, how: str) -> pl.Expr:
    f = {"sum": expr.rolling_sum, "mean": expr.rolling_mean, "std": expr.rolling_std}[how]
    return f(w, min_samples=max(2, w // 2)).over("symbol")


def build(bars: pl.DataFrame, cfg: dict, timeframe: str = "1Min",
          market_symbols: tuple[str, ...] = ("SPY", "QQQ"),
          enabled_groups: list[str] | None = None) -> FeatureSet:
    bm = TIMEFRAME_MINUTES[timeframe]
    G = cfg["groups"]
    enabled = set(enabled_groups or G.keys())
    df = bars.sort(["symbol", "ts"]).with_columns(
        r1=(C("close") / C("close").shift(1)).log().over("symbol"),
        hl=(C("high") / C("low")).log(),
        co=(C("close") / C("open")).log(),
    )
    groups: dict[str, list[str]] = {}
    exprs: list[pl.Expr] = []

    def add(group: str, name: str, e: pl.Expr) -> None:
        groups.setdefault(group, []).append(name)
        exprs.append(e.cast(pl.Float32).alias(name))

    if "returns" in enabled:
        for L in G["returns"]["lags_min"]:
            b = _bars(L, bm)
            add("returns", f"ret_{L}", (C("close") / C("close").shift(b)).log().over("symbol"))
    if "realized_vol" in enabled:
        for w in G["realized_vol"]["windows_min"]:
            add("realized_vol", f"rv_{w}", _roll(C("r1") ** 2, _bars(w, bm), "sum").sqrt())
    if "range_vol" in enabled:
        for w in G["range_vol"]["windows_min"]:
            b = _bars(w, bm)
            add("range_vol", f"park_{w}", (_roll(C("hl") ** 2, b, "mean") / (4 * math.log(2))).sqrt())
            gk = 0.5 * C("hl") ** 2 - (2 * math.log(2) - 1) * C("co") ** 2
            add("range_vol", f"gk_{w}", _roll(gk, b, "mean").clip(lower_bound=0).sqrt())
    if "volume" in enabled:
        lv = (C("volume") + 1).log()
        for w in G["volume"]["zscore_windows_min"]:
            b = _bars(w, bm)
            add("volume", f"volz_{w}", (lv - _roll(lv, b, "mean")) / (_roll(lv, b, "std") + 1e-6))
        if "is_filled" in df.columns:
            add("volume", "filled_30", _roll(C("is_filled").cast(pl.Float64), _bars(30, bm), "mean"))
    if "vwap_distance" in enabled:
        b = _bars(G["vwap_distance"]["lookback_min"], bm)
        pv = _roll(C("vwap") * C("volume"), b, "sum") / (_roll(C("volume"), b, "sum") + 1e-9)
        add("vwap_distance", "vwap_dist", (C("close") / pv).log())
    if "calendar" in enabled and bm < 390:
        et = C("ts").dt.convert_time_zone("America/New_York")
        mod = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32) - 570 + bm
        add("calendar", "min_of_day", mod)
        add("calendar", "min_to_close", (pl.len().over("symbol", "session") - pl.int_range(pl.len())
                                         .over("symbol", "session")) * bm)
        add("calendar", "dow", et.dt.weekday())
    elif "calendar" in enabled:
        add("calendar", "dow", C("ts").dt.convert_time_zone("America/New_York").dt.weekday())

    df = df.with_columns(exprs)
    feat_cols = [c for g in groups.values() for c in g]

    if "market_context" in enabled:
        for m in market_symbols:
            mk = df.filter(C("symbol") == m)
            if mk.height == 0:
                continue
            cols = [c for c in feat_cols if c.startswith(("ret_", "rv_"))]
            ren = {c: f"{m.lower()}_{c}" for c in cols}
            df = df.join(mk.select(["ts", *cols]).rename(ren), on="ts", how="left")
            groups.setdefault("market_context", []).extend(ren.values())
    if "cross_sectional" in enabled:
        xs = [c for c in ("ret_15", "ret_60", "rv_60", "volz_30") if c in df.columns]
        n = pl.len().over("ts")
        df = df.with_columns([((C(c).rank("average").over("ts") - 1) / (n - 1).clip(lower_bound=1))
                              .cast(pl.Float32).alias(f"xs_{c}") for c in xs])
        groups.setdefault("cross_sectional", []).extend(f"xs_{c}" for c in xs)
    feat_cols = [c for g in groups.values() for c in g]

    # ---- labels (forward), with intervals ----
    bar = timedelta(minutes=bm if bm < 390 else 0)
    df = df.with_columns(decision=C("ts") + bar if bm < 390 else C("ts"))
    label_cols = {}
    lab_exprs = []
    for hmin in cfg["labels"]["horizons_min"]:
        h = _bars(hmin, bm)
        same = (C("session").shift(-h).over("symbol") == C("session")) if bm < 390 else pl.lit(True)
        fwd_ret = (C("close").shift(-h) / C("close")).log().over("symbol")
        fwd_rv = (C("r1") ** 2).rolling_sum(h).shift(-h).over("symbol").sqrt()
        tg = cfg["labels"]["targets"]
        if "log_return" in tg:
            lab_exprs.append(pl.when(same).then(fwd_ret).alias(f"y_ret_{hmin}"))
            label_cols[f"ret_{hmin}"] = f"y_ret_{hmin}"
        if "realized_vol" in tg:
            lab_exprs.append(pl.when(same).then(fwd_rv).alias(f"y_rv_{hmin}"))
            label_cols[f"rv_{hmin}"] = f"y_rv_{hmin}"
    df = df.with_columns(lab_exprs)
    hmax = _bars(max(cfg["labels"]["horizons_min"]), bm)
    df = df.with_columns(label_start=C("decision"),
                         label_end=C("decision").shift(-hmax).over("symbol"))
    df = df.with_columns(label_end=C("label_end").fill_null(C("decision") + timedelta(days=36500)))
    max_lb = max(int(g.get("lookback_min", 0)) for k, g in G.items() if k in enabled)
    keep = ["symbol", "ts", "session", "label_start", "label_end", "close", "open",
            *feat_cols, *label_cols.values()]
    return FeatureSet(df.select(keep), feat_cols, label_cols, groups, max_lb)


def subsample(fs: FeatureSet, every: int) -> FeatureSet:
    """Keep every k-th decision time (reduces label overlap and memory)."""
    if every <= 1:
        return fs
    ts = fs.df.select(pl.col("ts").unique().sort()).with_row_index("k")
    keep = ts.filter(pl.col("k") % every == 0).select("ts")
    return FeatureSet(fs.df.join(keep, on="ts", how="semi"), fs.feature_cols, fs.label_cols,
                      fs.groups, fs.max_lookback_min)
