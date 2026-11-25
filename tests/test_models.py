import numpy as np
import torch

from forecast.data.sources import parse_yahoo_chart
from forecast.evaluation.metrics import diebold_mariano, pinball_rows
from forecast.models.baselines import HarRV, Unconditional
from forecast.models.patchnet import PatchNet
from forecast.serving.export import export_onnx
from forecast.training.compress import magnitude_masks, nonzero_params
from forecast.training.trainer import fit, predict

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def _data(n=600, seed=0):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, 6, 64)).astype(np.float32)
    tab = g.standard_normal((n, 2)).astype(np.float32)
    y = np.zeros((n, 3, 2), np.float32)
    y[:, :, 1] = 0.8 * tab[:, :1] + 0.1 * g.standard_normal((n, 3))   # learnable from tab
    y[:, :, 0] = g.standard_normal((n, 3))
    return {"X": X, "tab": tab, "y": y}


def test_quantiles_are_monotone_and_shapes_match():
    m = PatchNet(6, 2, 3, 5, lookback=32, d_model=16)
    out = m(torch.randn(7, 6, 64), torch.randn(7, 2))
    assert out.shape == (7, 3, 2, 5)
    assert bool((out.diff(dim=-1) >= 0).all())


def test_training_learns_and_beats_unconditional_on_learnable_target():
    tr, va = _data(1200, 1), _data(400, 2)
    m = PatchNet(6, 2, 3, 5, lookback=32, d_model=16, n_blocks=1, dropout=0.0)
    fit(m, tr, va, {"max_epochs": 25, "lr": 3e-3, "batch_size": 128, "patience": 25}, Q, "cpu")
    base = Unconditional(Q)
    base.fit(tr)
    lm = pinball_rows(va["y"], predict(m, va["X"], va["tab"], "cpu"), Q)[..., 1].mean()
    lb = pinball_rows(va["y"], base.predict(va["X"], va["tab"]), Q)[..., 1].mean()
    assert lm < 0.6 * lb


def test_har_beats_unconditional_when_vol_is_linear_in_tab():
    tr, va = _data(1500, 3), _data(500, 4)
    u, h = Unconditional(Q), HarRV(Q)
    u.fit(tr)
    h.fit(tr)
    lu = pinball_rows(va["y"], u.predict(va["X"], va["tab"]), Q)[..., 1]
    lh = pinball_rows(va["y"], h.predict(va["X"], va["tab"]), Q)[..., 1]
    assert lh.mean() < 0.5 * lu.mean()
    stat, p = diebold_mariano(lh.mean(1), lu.mean(1), np.arange(len(lh)), lag=3)
    assert stat < 0 and p < 0.01


def test_pruning_keeps_masked_weights_at_zero_through_finetune():
    tr = _data(300, 5)
    m = PatchNet(6, 2, 3, 5, lookback=32, d_model=16)
    masks = magnitude_masks(m, 0.6)
    fit(m, tr, tr, {"max_epochs": 2, "batch_size": 64}, Q, "cpu", masks=masks)
    total = sum(p.numel() for n, p in m.named_parameters() if n in masks)
    zeros = sum(int((p == 0).sum()) for n, p in m.named_parameters() if n in masks)
    assert zeros / total > 0.58
    assert nonzero_params(m) < sum(p.numel() for p in m.parameters())


def test_onnx_export_matches_torch(tmp_path):
    m = PatchNet(6, 2, 3, 5, lookback=64, d_model=24)
    r = export_onnx(m, tmp_path / "m.onnx", 2)
    assert r["onnx_max_abs_diff"] < 1e-4 and r["onnx_bytes"] < 1_000_000


def test_yahoo_parser_adjusts_and_drops_gaps():
    payload = {"chart": {"result": [{"timestamp": [86400, 172800, 259200], "indicators": {
        "quote": [{"open": [10, None, 12], "high": [11, None, 13], "low": [9, None, 11],
                   "close": [10, None, 12], "volume": [100, None, 300]}],
        "adjclose": [{"adjclose": [5, None, 12]}]}}]}}
    df = parse_yahoo_chart("X", payload)
    assert df.height == 2
    assert df["close"].to_list() == [5.0, 12.0] and df["high"].to_list() == [5.5, 13.0]
    assert str(df["t"].dtype) == "Datetime(time_unit='us', time_zone='UTC')"
