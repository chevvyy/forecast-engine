import numpy as np
import pytest
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


def test_inference_batch_shrinks_for_wide_models_and_forecasts_do_not_change():
    """2026-11-26: predict() used 4096 rows whatever the model, so a wide large-tier model ran out of
    VRAM in inference after training fine. The batch now follows the model's widest activation."""
    from forecast.features.build import CHANNELS, TAB
    from forecast.training.trainer import predict_batch

    def net(**kw):
        return PatchNet(len(CHANNELS), len(TAB), 3, 5, **kw)

    nano = net(lookback=128, patch=2, stride=1, d_model=32, n_blocks=3, kernel=25, ffn=2)
    assert predict_batch(nano) == 4096                      # small models keep the old batch
    c = dict(lookback=256, patch=2, stride=1, n_blocks=1, kernel=5, ffn=4)
    oom, spill = net(d_model=512, **c), net(d_model=384, **c)   # the two runs that went wrong
    for m in (oom, spill):
        b = predict_batch(m)
        assert 64 <= b < 4096
        assert b * 255 * m.cfg["d_model"] * 4 * 4 <= 2**29  # widest activation at most 0.5 GB
    assert predict_batch(oom) < predict_batch(spill)
    torch.manual_seed(0)
    x, tab = np.random.default_rng(0).standard_normal((300, len(CHANNELS), 128)).astype(np.float32), \
        np.zeros((300, len(TAB)), np.float32)
    a, b = predict(nano, x, tab, "cpu"), predict(nano, x, tab, "cpu", batch=64)
    assert a.shape == (300, 3, 2, 5) and np.allclose(a, b, atol=1e-5)


def test_har_with_time_of_day_learns_the_session_shape_and_plain_har_never_sees_the_clock():
    from forecast.models.baselines import HarRV, HarTod

    rng = np.random.default_rng(0)
    q = [0.1, 0.25, 0.5, 0.75, 0.9]

    def make(n, lo=0.0, hi=1.0):
        tod = rng.uniform(lo, hi, n).astype(np.float32)
        tab = np.stack([rng.normal(size=n), rng.normal(size=n), tod, (2 * tod - 1) ** 2], axis=1).astype(np.float32)
        y = np.zeros((n, 2, 2), np.float32)
        y[:, :, 0] = rng.normal(size=(n, 2))
        # log-vol: linear in the HAR columns plus a U over the session, deeper at the longer horizon
        u = (2 * tod - 1) ** 2
        y[:, 0, 1] = 0.5 * tab[:, 0] - 0.2 * tab[:, 1] + 1.0 * u + 0.1 * rng.normal(size=n)
        y[:, 1, 1] = 0.3 * tab[:, 0] + 2.0 * u + 0.1 * rng.normal(size=n)
        return {"tab": tab, "y": y}

    tr, va = make(20000), make(5000)
    har, hod = HarRV(q, 2), HarTod(q, tod_col=2, n_har=2)
    har.fit(tr)
    hod.fit(tr)
    lh = pinball_rows(va["y"], har.predict(None, va["tab"]), q)[..., 1].mean()
    lt = pinball_rows(va["y"], hod.predict(None, va["tab"]), q)[..., 1].mean()
    assert lt < 0.5 * lh
    # Plain HAR reads its own two columns only: the clock columns can hold anything.
    scrambled = va["tab"].copy()
    scrambled[:, 2:] = 99.0
    np.testing.assert_array_equal(har.predict(None, va["tab"]), har.predict(None, scrambled))
    # Returns stay the unconditional quantiles in both.
    np.testing.assert_array_equal(hod.predict(None, va["tab"])[:, :, 0], har.predict(None, va["tab"])[:, :, 0])
    # A time of day never seen in training (rows only from the first half of the session) still gives a
    # finite forecast near the overall level, not zero and not an explosion.
    half = make(5000, 0.0, 0.5)
    hod2 = HarTod(q, tod_col=2, n_har=2)
    hod2.fit(half)
    late = make(200, 0.6, 1.0)
    p = hod2.predict(None, late["tab"])[:, :, 1]
    assert np.isfinite(p).all() and np.abs(p[:, 0, 2].mean() - half["y"][:, 0, 1].mean()) < 0.5


def test_a_parent_from_before_the_clock_inputs_is_refused_with_a_clear_error():
    from forecast.features.build import CHANNELS
    from forecast.loop.run_job import _same_inputs

    m = PatchNet(len(CHANNELS), 2, 3, 5, lookback=16, d_model=8, n_blocks=1, kernel=5)
    assert _same_inputs(m, 2, "old") is m
    with pytest.raises(ValueError, match="old was trained on 2 side inputs, this track has 4"):
        _same_inputs(m, 4, "old")
