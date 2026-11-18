"""One-shot final holdout evaluation of the current champion. Writes a lock so it can't be repeated
silently: re-running requires --force and a reason, which is recorded (and counts against DSR)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np

from forecast.evaluation.report import write_card
from forecast.evaluation.splits import holdout_cutoff, research_mask
from forecast.models.base import PickleMixin
from forecast.models.baselines import baselines_for
from forecast.training import experiment as E
from forecast.training.autotune import champions_dir
from forecast.utils import env
from forecast.utils.timeutil import parse_duration


def run(cfg: dict, task_name: str, force_reason: str | None = None) -> dict:
    lock = env.experiments_dir() / "holdout_lock.json"
    used = json.loads(lock.read_text()) if lock.exists() else []
    prior = [u for u in used if u["task"] == task_name]
    if prior and not force_reason:
        raise SystemExit(f"Holdout for {task_name} already used on {prior[-1]['at']}. "
                         "Use --force '<reason>' (recorded) only if you accept the bias.")
    champ = json.loads((champions_dir() / f"{task_name}.json").read_text())
    fs, data_hash = E.prepare(cfg)
    task = E.make_tasks(fs, cfg)[task_name]
    e = cfg["eval"]
    hold = parse_duration(e["final_holdout"])
    df = fs.df
    y = df[task.label_col].to_numpy()
    cut = holdout_cutoff(df, hold)
    research = np.flatnonzero(research_mask(df, hold, timedelta(minutes=e["embargo_min"])) & np.isfinite(y))
    test = np.flatnonzero((df["label_start"] >= cut).to_numpy() & np.isfinite(y))
    model = PickleMixin.load(champions_dir() / f"{task_name}.pkl")
    from forecast.evaluation.splits import inner_split

    def oof_of(m, fit_needed: bool):
        if fit_needed:
            fit, es = inner_split(df, research)
            X = df.select(fs.feature_cols)
            m.fit(X[fit], y[fit], X[es], y[es], task)
            q = m.predict(X[test])
        else:
            q = m.predict(df.select(m.features)[test])
        return E.OOF(getattr(m, "name", "m"), task, df["ts"].to_numpy()[test],
                     df["symbol"].to_numpy()[test], y[test], q, np.zeros(len(test), int))

    mo = oof_of(model, False)
    res = {"task": task_name, "champion": champ["model"], "n": int(len(test)), "baselines": {}}
    best_name, best = None, None
    for b in baselines_for(task):
        bo = oof_of(type(b)(), True)
        res["baselines"][bo.model] = E.compare(mo, bo)
        if best is None or np.mean(bo.loss()) < np.mean(best.loss()):
            best_name, best = bo.model, bo
    dm = E.compare(mo, best)
    concl = ("Holdout confirms the edge." if dm["dm_p"] < 0.05 and dm["rel"] < 0 else
             "Holdout does NOT confirm an edge over the best baseline.")
    card = write_card(task, champ["model"], E.score(mo), best_name, E.score(best), dm,
                      {"data_hash": data_hash, "n_folds": "holdout", "embargo_min": e["embargo_min"],
                       "n_trials_total": champ["n_trials_total"], "params": champ["params"],
                       "groups": champ["groups"]}, concl, tag="-HOLDOUT")
    used.append({"task": task_name, "at": datetime.now(UTC).isoformat(), "champion": champ["model"],
                 "reason": force_reason or "first use", "result": dm, "card": card.name})
    lock.write_text(json.dumps(used, indent=2, default=float))
    res["vs_best"] = {"baseline": best_name, **dm}
    return res
