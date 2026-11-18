import numpy as np
import polars as pl
import pytest

from forecast.features.build import build

pytestmark = pytest.mark.leakage


@pytest.mark.parametrize("frac", [0.3, 0.55, 0.8])
def test_future_perturbation_invariance(aligned, fcfg, fs, frac):
    ts = aligned["ts"].unique().sort()
    t = ts[int(len(ts) * frac)]
    rng = np.random.default_rng(int(frac * 100))
    fut = (aligned["ts"] > t).to_numpy()
    shock = np.where(fut, np.exp(rng.normal(0, 0.05, aligned.height)), 1.0)
    pert = aligned.with_columns([(pl.col(c) * pl.Series(shock)).alias(c)
                                 for c in ("open", "high", "low", "close", "vwap")]
                                + [(pl.col("volume") * pl.Series(shock ** 3)).alias("volume")])
    fs2 = build(pert, fcfg)
    a = fs.df.filter(pl.col("ts") <= t).select(["symbol", "ts", *fs.feature_cols])
    b = fs2.df.filter(pl.col("ts") <= t).select(["symbol", "ts", *fs.feature_cols])
    assert a.height == b.height > 0
    assert a.equals(b), "features at <= t changed when only data after t changed (look-ahead)"
    # sanity: something after t did change
    assert not fs.df.filter(pl.col("ts") > t)["ret_1"].equals(fs2.df.filter(pl.col("ts") > t)["ret_1"])


def test_labels_are_forward_and_in_session(fs):
    d = fs.df.filter(pl.col("symbol") == "AAA").sort("ts")
    c = d["close"].to_numpy()
    sess = d["session"].to_numpy()
    y = d["y_ret_15"].to_numpy()
    for i in range(0, len(d) - 15, 97):
        if sess[i + 15] == sess[i]:
            assert np.isclose(y[i], np.log(c[i + 15] / c[i]), atol=1e-12)
        else:
            assert np.isnan(y[i]) or y[i] is None
    assert (d["label_start"] > d["ts"]).all()        # decision after bar open (bar close)
    assert (d["label_end"] > d["label_start"]).all()


def test_label_never_crosses_session(fs):
    d = fs.df.filter(pl.col("symbol") == "BBB").sort("ts")
    last = d.group_by("session").agg(pl.col("y_ret_5").tail(5).is_null().all())
    assert last["y_ret_5"].all()
