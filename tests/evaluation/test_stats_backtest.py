import numpy as np
import polars as pl
import pytest

from forecast.evaluation.backtest import backtest
from forecast.evaluation.metrics import coverage, hit_rate, pinball, qlike
from forecast.evaluation.stats import deflated_sharpe, diebold_mariano


def test_dm_behaviour():
    rng = np.random.default_rng(0)
    base = rng.gamma(2, 1, 3000)
    same = base + rng.normal(0, 0.01, 3000)
    _, p_same = diebold_mariano(same, base, alternative="two-sided")
    assert p_same > 0.05
    _, p_better = diebold_mariano(base * 0.9, base)
    assert p_better < 1e-6
    _, p_worse = diebold_mariano(base * 1.1, base)
    assert p_worse > 0.99


def test_metrics_basic():
    qs = np.array([0.1, 0.5, 0.9])
    y = np.zeros(4)
    q = np.tile([-1.0, 0.0, 1.0], (4, 1))
    assert pinball(y, q, qs).mean() == pytest.approx(0.2 / 3)
    assert coverage(y, q, qs) == 1.0
    assert qlike(np.ones(3), np.ones(3)).max() == pytest.approx(0)
    hr, p, n = hit_rate(np.array([1.0, -1, 1, 1]), np.array([1.0, -1, -1, 1]))
    assert hr == 0.75 and n == 4


def test_dsr_penalises_trials():
    a = deflated_sharpe(0.05, 2000, 1, 0.001)
    b = deflated_sharpe(0.05, 2000, 1000, 0.001)
    assert a > b


@pytest.mark.leakage
def test_backtest_fills_after_signal(aligned):
    pred = aligned.select("symbol", "ts").with_columns(score=pl.lit(np.random.default_rng(0)
                                                       .normal(size=aligned.height)))
    out = backtest(pred, aligned, h=15, top_k=2)   # asserts entry_ts > ts internally
    assert out["n_periods"] > 10
    assert out["net_sharpe"] < out["gross_sharpe"]
