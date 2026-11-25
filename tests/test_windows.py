"""Lazy windows must be indistinguishable from the eager ``X[n, C, L]`` array they replaced."""
import numpy as np
import polars as pl
import pytest
import torch

from forecast.data.sources import synthetic
from forecast.features.build import WARMUP, Windows, build, symbol_arrays
from forecast.models.patchnet import PatchNet
from forecast.training.trainer import fit, predict

H = [1, 5, 20]


def _eager(bars: pl.DataFrame, lookback: int, first_index: int = 0, benchmark: str = "SPY") -> np.ndarray:
    """The original builder's window code, kept here as the reference implementation."""
    hmax = max(H)
    bench = bars.filter(pl.col("symbol") == benchmark).sort("t")
    bc = bench["close"].to_numpy()
    br = np.zeros(len(bc))
    br[1:] = np.log(bc[1:] / bc[:-1])
    bench = pl.DataFrame({"t": bench["t"], "mkt_ret": br})
    out = []
    for s in sorted(bars["symbol"].unique().to_list()):
        d = bars.filter(pl.col("symbol") == s).sort("t")
        d = d.join(bench, on="t", how="left").with_columns(pl.col("mkt_ret").fill_null(0.0))
        first = max(WARMUP, lookback - 1, first_index)
        if d.height < first + hmax + 2:
            continue
        ch, *_ = symbol_arrays(*(d[c].to_numpy() for c in ["open", "high", "low", "close", "volume", "mkt_ret"]))
        idx = np.arange(first, d.height - hmax)
        win = np.lib.stride_tricks.sliding_window_view(ch, lookback, axis=0)
        out.append(win[idx - lookback + 1].astype(np.float32))
    return np.concatenate(out)


def _bars() -> pl.DataFrame:
    # Symbols of different lengths (one too short to use) so offsets between symbols are exercised.
    bars = synthetic(["SPY", "AAA", "BBB", "CCC", "TINY"], 500, seed=11)
    t = bars["t"].unique().sort()
    return bars.filter(
        ((pl.col("symbol") != "BBB") | (pl.col("t") >= t[137]))
        & ((pl.col("symbol") != "CCC") | (pl.col("t") <= t[401]))
        & ((pl.col("symbol") != "TINY") | (pl.col("t") >= t[470]))
    )


@pytest.mark.leakage
@pytest.mark.parametrize(("lookback", "first_index"), [(32, 0), (64, 127), (128, 127)])
def test_lazy_windows_equal_the_eager_builder(lookback, first_index):
    bars = _bars()
    ds = build(bars, H, lookback, first_index=first_index)
    ref = _eager(bars, lookback, first_index)
    assert isinstance(ds.X, Windows) and "TINY" in ds.symbols and len(np.unique(ds.sym)) == 4
    assert ds.X.shape == ref.shape and len(ds.X) == len(ds.y) == len(ref)
    full = np.asarray(ds.X)
    assert full.dtype == np.float32
    np.testing.assert_array_equal(full, ref)
    # Every way callers index it gives the eager answer.
    rows = np.random.default_rng(0).permutation(len(ref))[:300]
    np.testing.assert_array_equal(ds.X[rows], ref[rows])
    np.testing.assert_array_equal(ds.X[100:357], ref[100:357])
    np.testing.assert_array_equal(ds.X[ds.sym == 2], ref[ds.sym == 2])
    np.testing.assert_array_equal(ds.X[5], ref[5])
    np.testing.assert_array_equal(ds.X[..., -8:], ref[..., -8:])
    np.testing.assert_array_equal(ds.X.subset(rows)[10:20], ref[rows][10:20])
    # The last column of every window is bar i itself: channel 0 there is that bar's return, and
    # nothing in the stored channels lies beyond the newest bar of the newest window.
    assert ds.X.end.max() < len(ds.X.ch)
    np.testing.assert_array_equal(full[:, :, -1], ds.X.ch[ds.X.end])


def test_training_on_lazy_windows_matches_training_on_arrays():
    ds = build(synthetic(["SPY", "AAA", "BBB"], 420, seed=5), H, 32)
    n = len(ds.y)
    tr, es = np.arange(0, int(0.8 * n)), np.arange(int(0.8 * n), n)
    Q = [0.1, 0.5, 0.9]
    cfg = {"batch_size": 128, "max_epochs": 2, "lr": 1e-3}
    preds = []
    for lazy in (True, False):
        def part(idx, lazy=lazy):
            return {"X": ds.X.subset(idx) if lazy else ds.X[idx], "tab": ds.tab[idx], "y": ds.y[idx]}
        torch.manual_seed(0)
        m = PatchNet(6, 2, len(H), len(Q), lookback=32, patch=4, stride=2, d_model=8, n_blocks=1, kernel=5,
                     ffn=1, dropout=0.0)
        st = fit(m, part(tr), part(es), cfg, Q, "cpu")
        assert st["epochs"] == 2
        preds.append(predict(m, part(es)["X"], ds.tab[es], "cpu"))
    np.testing.assert_allclose(preds[0], preds[1], rtol=0, atol=1e-6)
