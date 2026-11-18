"""Score the latest bar per symbol with every champion -> serving/latest.json (read by api.py)."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from forecast.models.base import PickleMixin
from forecast.training import experiment as E
from forecast.training.autotune import champions_dir
from forecast.utils import env


def score_latest(cfg: dict) -> dict:
    cfg = {**cfg, "data": {**cfg["data"], "subsample_every": 1}}
    fs, data_hash = E.prepare(cfg, use_cache=False)
    last = fs.df.group_by("symbol").agg(pl.all().sort_by("ts").last())
    out = {"generated_at": datetime.now(UTC).isoformat(), "data_hash": data_hash, "symbols": {}}
    for cj in sorted(champions_dir().glob("*.json")):
        champ = json.loads(cj.read_text())
        m = PickleMixin.load(cj.with_suffix(".pkl"))
        q = m.predict(last.select(m.features))
        for i, sym in enumerate(last["symbol"].to_list()):
            s = out["symbols"].setdefault(sym, {"as_of": str(last["ts"][i]),
                                                "close": float(last["close"][i]), "forecasts": {}})
            entry = {"kind": champ["kind"], "horizon_min": champ["horizon_min"],
                     "quantiles": dict(zip(map(str, champ["quantiles"]), map(float, q[i]), strict=True)),
                     "model": champ["model"],
                     "validation": {"vs_baseline": champ["vs_best_baseline"]["baseline"],
                                    "rel_loss_change": champ["vs_best_baseline"]["rel"],
                                    "dm_p": champ["vs_best_baseline"]["dm_p"],
                                    "significant": champ["vs_best_baseline"]["significant"],
                                    "trials_tried": champ["n_trials_total"]}}
            if champ["kind"] == "ret":   # price band derived at display time (ADR-0001)
                entry["price_band"] = {k: float(last["close"][i] * np.exp(v))
                                       for k, v in entry["quantiles"].items()}
            s["forecasts"][champ["task"]] = entry
    p = env.data_root() / "serving" / "latest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return out
