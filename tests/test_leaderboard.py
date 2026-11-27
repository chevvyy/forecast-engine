"""The configuration board: seed means next to the single-run board."""
import math

from forecast.loop.leaderboard import champions, configurations, render


def _run(i, score, seed, tier="nano", ver=2, d_model=24, lr=0.001, **job):
    return {"id": f"r{i}", "status": "ok", "fits_tier": True, "tier": tier, "eval_version": ver,
            "finished": f"2026-11-27T00:{i:02d}:00+00:00", "score": score, "skill_ret": score - 0.01,
            "skill_vol": score + 0.01, "dm": {"all": [0.0, 0.5]}, "params": 1, "onnx_bytes": 1024,
            "cpu_ms_per_forecast": 0.1, "peak_vram_gb": 0.1, "kind": "train",
            "model": {"d_model": d_model, "lookback": 64}, "train": {"lr": lr, "batch_size": 512},
            "job": {"seed": seed, **job}}


def test_configurations_rank_by_seed_mean_not_by_the_luckiest_seed():
    lucky = [_run(1, 0.050, 1, d_model=32), _run(2, 0.010, 2, d_model=32), _run(3, 0.012, 3, d_model=32)]
    steady = [_run(4, 0.030, 1), _run(5, 0.032, 2)]
    res = lucky + steady
    assert champions(res, 2)["nano"]["id"] == "r1"          # the single-run board is unchanged
    confs = configurations(res, 2)
    assert [c["best"]["id"] for c in confs] == ["r5", "r1"]
    assert confs[0]["n"] == 2 and math.isclose(confs[0]["mean"], 0.031)
    assert math.isclose(confs[0]["sd"], 0.002 / math.sqrt(2))   # sample sd of two seeds
    assert confs[1]["n"] == 3 and (confs[1]["min"], confs[1]["max"]) == (0.010, 0.050)
    assert [r["id"] for r in confs[1]["runs"]] == ["r1", "r2", "r3"]
    text = render(res, 2)
    assert "## Configurations by seed mean" in text
    assert text.index("| +0.0310 |") < text.index("| +0.0240 |")


def test_a_configuration_is_shape_settings_teacher_and_parent_and_needs_different_seeds():
    res = [_run(1, 0.02, 1), _run(2, 0.03, 1),                # the same seed twice: one draw, first kept
           _run(3, 0.04, 2, lr=0.0003),                       # other training settings
           _run(4, 0.05, 2, teacher="t"),                     # other teacher
           _run(5, 0.06, 2, init_from="p", sparsity=0.5),     # a prune of something
           _run(6, 0.07, 2, tier="mid"),                      # other tier
           _run(7, 0.08, 2, ver=1),                           # other evaluation version
           _run(8, math.nan, 2),                              # not a score
           {**_run(9, 0.09, 2), "fits_tier": False},
           {**_run(10, 0.09, 2), "status": "error"},
           {"id": "junk", "status": "ok", "fits_tier": True, "score": 0.5, "eval_version": 2,
            "tier": "nano", "job": "not a dict", "model": 3}]
    assert configurations(res, 2) == []                       # nothing ran with two different seeds
    assert "Configurations by seed mean" not in render(res[:-1], 2)   # (the run table itself needs complete rows)
    singles = configurations(res, 2, min_seeds=1)
    assert sorted(c["best"]["id"] for c in singles) == ["junk", "r1", "r3", "r4", "r5", "r6"]
    assert all(c["n"] == 1 and math.isnan(c["sd"]) for c in singles)
    # a second seed joins its own configuration only; `kind` is a label and does not split one
    more = res + [{**_run(11, 0.04, 3), "kind": "distill"}]
    (c,) = configurations(more, 2)
    assert [r["id"] for r in c["runs"]] == ["r1", "r11"] and math.isclose(c["mean"], 0.03)
