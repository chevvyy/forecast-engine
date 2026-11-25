"""Look-ahead guards. These must never be skipped or weakened (CLAUDE.md)."""
import numpy as np
import polars as pl
import pytest

from forecast.data.sources import synthetic
from forecast.evaluation.splits import holdout_start, walk_forward
from forecast.features.build import build

H = [1, 5, 20]


@pytest.mark.leakage
def test_features_do_not_see_the_future():
    bars = synthetic(["SPY", "AAA", "BBB"], 400, seed=1)
    cut = bars["t"].unique().sort()[250]
    # Corrupt every bar strictly after `cut`.
    bad = bars.with_columns([
        pl.when(pl.col("t") > cut).then(pl.col(c) * 3.7).otherwise(pl.col(c)).alias(c)
        for c in ["open", "high", "low", "close", "volume"]
    ])
    a, b = build(bars, H, 32), build(bad, H, 32)
    cut_s = int(cut.timestamp())
    keep = a.t <= cut_s
    assert keep.sum() > 100
    np.testing.assert_array_equal(a.X[keep], b.X[b.t <= cut_s])
    np.testing.assert_array_equal(a.tab[keep], b.tab[b.t <= cut_s])
    np.testing.assert_array_equal(a.scale[keep], b.scale[b.t <= cut_s])
    # ...and labels that end at or before the cut are untouched too.
    done = a.t_end <= cut_s
    np.testing.assert_array_equal(a.y[done], b.y[b.t_end <= cut_s])


@pytest.mark.leakage
def test_folds_are_purged_and_holdout_is_unreachable():
    bars = synthetic(["SPY", "AAA", "BBB", "CCC"], 900, seed=2)
    ut = np.unique(bars["t"].dt.epoch("s").to_numpy())
    hs = holdout_start(ut, 0.1)
    ds = build(bars, H, 32, max_t_end=hs, first_index=63)
    assert ds.t_end.max() < hs
    folds = walk_forward(ds.t, ds.t_end, n_folds=3, val_frac=0.08, embargo_bars=5, es_frac=0.1)
    assert len(folds) == 3
    for f in folds:
        vs = ds.t[f.val].min()
        assert ds.t_end[f.train].max() < ds.t[f.es].min()      # train core purged from early-stop slice
        assert ds.t_end[f.es].max() < vs                        # nothing fitted overlaps validation labels
        assert len(np.intersect1d(f.train, f.val)) == 0 and len(np.intersect1d(f.es, f.val)) == 0
        gap_bars = np.sum((ut > ds.t_end[f.es].max()) & (ut < vs))
        assert gap_bars >= 4                                    # embargo


@pytest.mark.leakage
def test_folds_identical_across_lookbacks():
    bars = synthetic(["SPY", "AAA"], 700, seed=3)
    a = build(bars, H, 32, first_index=127)
    b = build(bars, H, 128, first_index=127)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.X, b.X[..., -32:])
