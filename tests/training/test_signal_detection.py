"""The pipeline must find a planted signal and must not find one in noise."""
import pytest

from forecast.models.baselines import ZeroReturnVolScaled
from forecast.models.gbdt import LGBMQuantile
from forecast.training import experiment as E


def _cfg(signal: float) -> dict:
    cfg = E.load_run_config("configs/run/synthetic.yaml")
    cfg["data"]["synthetic"] = {"n_symbols": 6, "start": "2024-01-02", "end": "2024-04-15",
                                "signal": signal, "seed": 3}
    cfg["eval"].update(first_val_start="2024-02-20", val_block="3W", step="3W", final_holdout="2W")
    return cfg


def _run(signal):
    cfg = _cfg(signal)
    fs, _ = E.prepare(cfg, use_cache=False)
    task = E.make_tasks(fs, cfg)["ret_15"]
    fl = E.folds(fs, cfg)
    b = E.run_oof(ZeroReturnVolScaled, fs, task, fl, fs.feature_cols)
    m = E.run_oof(lambda: LGBMQuantile(num_boost_round=400, early_stopping=40, n_jobs=2,
                                       params={"min_data_in_leaf": 200}), fs, task, fl, fs.feature_cols)
    return E.compare(m, b), E.score(m)


@pytest.mark.slow
def test_finds_planted_signal():
    dm, s = _run(signal=0.5)
    assert dm["rel"] < 0 and dm["dm_p"] < 0.01, dm
    assert s["hit_rate"] > 0.51 and s["hit_p"] < 0.01


@pytest.mark.slow
def test_no_false_edge_in_noise():
    dm, s = _run(signal=0.0)
    assert dm["dm_p"] > 0.05, f"found an 'edge' in pure noise: {dm}"
