"""Intraday bars: a bar is known one bar after its start, and nothing crosses the overnight gap."""
import numpy as np
import polars as pl
import pytest
import yaml

from forecast.data.sources import synthetic
from forecast.evaluation.splits import holdout_start, walk_forward
from forecast.features.build import build, grid_runs
from forecast.loop.run_job import bar_seconds, run_job
from forecast.utils.common import REPO, interval_seconds

H = [1, 3, 6]
BAR = 300          # 5-minute bars: 78 per session
L = 16
PRICE = ["open", "high", "low", "close"]


def _bars(n: int = 78 * 12, holes: bool = True) -> pl.DataFrame:
    bars = synthetic(["SPY", "AAA", "BBB"], n, seed=4, bar_seconds=BAR)
    if holes:  # a few missing bars in one symbol, as real minute data has
        t = bars["t"].unique().sort()
        gone = [t[200], t[201], t[455]]
        bars = bars.filter(~((pl.col("symbol") == "AAA") & pl.col("t").is_in(gone)))
    return bars


def test_interval_seconds_and_job_wiring():
    assert [interval_seconds(s) for s in ("1m", "5m", "1h", "1d")] == [60, 300, 3600, 86400]
    with pytest.raises(ValueError):
        interval_seconds("fast")
    assert bar_seconds({"source": "yahoo"}) is None and bar_seconds({"interval": "1d"}) is None
    assert bar_seconds({"source": "alpaca"}) == 60 and bar_seconds({"interval": "1h"}) == 3600
    assert grid_runs(np.array([0, 60, 120, 300, 360, 9000]), 60).tolist() == [1, 2, 3, 1, 2, 1]


def test_synthetic_intraday_has_sessions_and_overnight_jumps():
    bars = synthetic(["SPY"], 78 * 3, seed=1, bar_seconds=BAR)
    assert bars["t"].dtype == pl.Datetime("us", "UTC")
    t = bars["t"].dt.epoch("s").to_numpy()
    assert (np.diff(t) > BAR).sum() == 2 and grid_runs(t, BAR).max() == 78
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    inside = np.diff(t) == BAR
    np.testing.assert_allclose(o[1:][inside], c[:-1][inside])     # continuous inside a session
    assert np.all(np.abs(np.log(o[1:][~inside] / c[:-1][~inside])) > 1e-6)   # jump at each open


@pytest.mark.leakage
def test_intraday_rows_are_known_one_bar_after_their_start():
    bars = _bars()
    ut = bars["t"].unique().sort()
    cut = ut[78 * 7 + 40]                       # "now": the start of a mid-session bar
    cut_s = int(cut.timestamp())
    # Everything not finished at `cut` is unknown then: corrupt every bar starting at or after it.
    bad = bars.with_columns([
        pl.when(pl.col("t") >= cut).then(pl.col(c) * 3.7).otherwise(pl.col(c)).alias(c)
        for c in [*PRICE, "volume"]
    ])
    a, b = build(bars, H, L, bar_seconds=BAR), build(bad, H, L, bar_seconds=BAR)
    np.testing.assert_array_equal(a.known, a.t + BAR)
    assert (a.t >= cut_s).sum() > 100 and not np.any(a.known[a.t >= cut_s] <= cut_s)
    keep, keep_b = a.known <= cut_s, b.known <= cut_s
    assert keep.sum() > 500
    np.testing.assert_array_equal(a.t[keep], b.t[keep_b])
    np.testing.assert_array_equal(a.X[keep], b.X[keep_b])
    np.testing.assert_array_equal(a.tab[keep], b.tab[keep_b])
    np.testing.assert_array_equal(a.scale[keep], b.scale[keep_b])
    # The bar that started one bar before `cut` is the newest usable one, and it is used.
    assert a.t[keep].max() == cut_s - BAR
    # Labels whose last bar has finished by `cut` are untouched too.
    done, done_b = a.t_end + BAR <= cut_s, b.t_end + BAR <= cut_s
    np.testing.assert_array_equal(a.y[done], b.y[done_b])


@pytest.mark.leakage
def test_intraday_windows_and_labels_stay_on_one_unbroken_grid():
    bars = _bars()
    ds = build(bars, H, L, bar_seconds=BAR)
    hmax = max(H)
    assert len(ds.t) > 1000
    np.testing.assert_array_equal(ds.t_end, ds.t + hmax * BAR)      # label bars are the next hmax bars
    n_rows = {}
    for si, s in enumerate(ds.symbols):
        have = set(bars.filter(pl.col("symbol") == s)["t"].dt.epoch("s").to_list())
        t = ds.t[ds.sym == si]
        n_rows[s] = len(t)
        for k in range(-L + 1, hmax + 1):                            # every window and label bar exists
            assert have.issuperset((t + k * BAR).tolist()), (s, k)
    assert n_rows["AAA"] < n_rows["BBB"]                             # rows next to the holes are gone
    # No row in the first L-1 bars of a session or its last hmax bars (sessions start 14:30 UTC).
    bar_of_day = (ds.t - 14 * 3600 - 1800) % 86400 // BAR
    assert bar_of_day.min() == L - 1 and bar_of_day.max() == 78 - 1 - hmax


@pytest.mark.leakage
def test_overnight_moves_enter_nothing():
    bars = _bars()
    # Shift each session's prices by its own factor: only the overnight moves change.
    day = pl.col("t").dt.epoch("s") // 86400
    shifted = bars.with_columns([(pl.col(c) * (1.0 + 0.03 * (day % 7))).alias(c) for c in PRICE])
    a, b = build(bars, H, L, bar_seconds=BAR), build(shifted, H, L, bar_seconds=BAR)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_allclose(np.asarray(a.X), np.asarray(b.X), atol=2e-6)
    for name in ("tab", "y", "scale"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-4, atol=2e-6)
    # ...whereas the daily rule on the same bars does let them in.
    c, d = build(bars, H, L), build(shifted, H, L)
    assert np.abs(np.asarray(c.X) - np.asarray(d.X)).max() > 1e-3


@pytest.mark.leakage
def test_intraday_folds_identical_across_lookbacks_and_holdout_unreachable():
    bars = _bars(78 * 30)
    hs = holdout_start(np.unique(bars["t"].dt.epoch("s").to_numpy()), 0.1)
    a = build(bars, H, 8, max_t_end=hs, first_index=31, bar_seconds=BAR)
    b = build(bars, H, 32, max_t_end=hs, first_index=31, bar_seconds=BAR)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.X, b.X[..., -8:])
    assert a.t_end.max() + BAR <= hs
    folds = walk_forward(a.t, a.t_end, n_folds=3, val_frac=0.08, embargo_bars=6, es_frac=0.1)
    assert len(folds) == 3
    for f in folds:
        assert a.t_end[f.train].max() < a.t[f.es].min()
        assert a.t_end[f.es].max() + BAR <= a.t[f.val].min()


def test_daily_build_ignores_the_grid_rule():
    bars = synthetic(["SPY", "AAA"], 400, seed=3)
    ds = build(bars, H, L)
    np.testing.assert_array_equal(ds.known, ds.t + 86400)
    assert len(ds.t) == 2 * (400 - 60 - max(H))


def test_job_runs_end_to_end_on_intraday_bars(tmp_path):
    cfg = yaml.safe_load((REPO / "configs" / "loop.yaml").read_text(encoding="utf-8"))
    cfg["data"] = {"source": "synthetic", "symbols": ["SPY", "AAA", "BBB"], "n_bars": 78 * 40, "seed": 7,
                   "interval": "5m"}
    cfg["task"].update(horizons=H, max_lookback=32)
    cfg["train"].update(max_epochs=1, batch_size=256)
    cfg["tiers"]["nano"]["model"].update(lookback=16, d_model=8, n_blocks=1, kernel=5, patch=4, stride=2)
    res = run_job({"id": "intra", "tier": "nano", "seed": 1}, cfg, repo=tmp_path, device="cpu")
    assert res["status"] == "ok", res.get("traceback")
    assert res["horizons"] == H and np.isfinite(res["score"])
