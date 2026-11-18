"""Self-improving tuning loop (champion / challenger).

Each round:
  1. Optuna (TPE + median pruner) proposes LightGBM hyperparameters AND which feature groups to use.
     Trials are scored on walk-forward validation folds only: relative loss vs the best baseline.
  2. The round's best trial is a challenger. It replaces the champion only if its per-timestamp loss
     is lower by >= min_rel_improvement AND a Diebold-Mariano test gives p < promote_alpha.
  3. A promoted champion is refit on all research data, saved, and gets a result card.
The final holdout is never read here (see holdout.py). Every trial counts toward the Deflated Sharpe
Ratio bookkeeping, so failures are logged too. Several machines can share one study via OPTUNA_STORAGE
(`fe-worker`), while one process runs `fe-autotune` to handle promotions.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna

from forecast.evaluation.report import log_line, write_card
from forecast.models.baselines import baselines_for
from forecast.models.gbdt import LGBMQuantile
from forecast.training import experiment as E
from forecast.utils import env
from forecast.utils.config import config_hash, load_yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)


def champions_dir() -> Path:
    d = env.experiments_dir() / "champions"
    d.mkdir(exist_ok=True)
    return d


def storage_url() -> str:
    return env.get("OPTUNA_STORAGE") or f"sqlite:///{env.experiments_dir() / 'optuna.db'}"


class TaskTuner:
    def __init__(self, cfg: dict, fs, data_hash: str, task_name: str):
        self.cfg, self.fs, self.data_hash = cfg, fs, data_hash
        self.tc = cfg["tune"]
        self.task = E.make_tasks(fs, cfg)[task_name]
        self.folds = E.folds(fs, cfg, self.tc.get("max_folds"))
        self.space = load_yaml("configs/model/lightgbm.yaml")["search_space"]
        self.groups = list(fs.groups)
        self.study_name = f"lgbm-{task_name}-{data_hash}-{config_hash(cfg['features_cfg'])}"
        self.dir = env.experiments_dir() / "autotune" / self.study_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._baselines()

    # -------------------------------------------------------------- baselines
    def _baselines(self) -> None:
        self.base = {}
        for b in baselines_for(self.task):
            o = E.run_oof(lambda b=b: type(b)(), self.fs, self.task, self.folds, self.fs.feature_cols)
            self.base[o.model] = (o, E.score(o))
        self.best_base = min(self.base, key=lambda k: np.mean(self.base[k][0].loss()))
        bo, bs = self.base[self.best_base]
        self.base_fold = bs["per_fold"]
        self.base_ts, self.base_loss = E.ts_loss(bo)
        print(f"[{self.task.name}] best baseline: {self.best_base} "
              f"loss={np.mean(bo.loss()):.6g} over {len(self.folds)} folds", flush=True)

    # -------------------------------------------------------------- search
    def suggest(self, trial: optuna.Trial) -> tuple[dict, list[str]]:
        params = {}
        for k, s in self.space.items():
            if s["type"] == "int":
                params[k] = trial.suggest_int(k, s["low"], s["high"], log=s.get("log", False))
            else:
                params[k] = trial.suggest_float(k, s["low"], s["high"], log=s.get("log", False))
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-3, 100, log=True)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        groups = [g for g in self.groups if trial.suggest_categorical(f"use_{g}", [True, False])]
        if len(groups) < 2:
            raise optuna.TrialPruned("need >= 2 feature groups")
        return params, groups

    def features_for(self, groups: list[str]) -> list[str]:
        return [c for g in groups for c in self.fs.groups[g]]

    def factory(self, params: dict):
        tc = self.tc
        return lambda: LGBMQuantile(params, device=tc.get("device", "cpu"), n_jobs=tc.get("n_jobs", -1),
                                    num_boost_round=tc.get("num_boost_round", 2000),
                                    early_stopping=tc.get("early_stopping", 100))

    def objective(self, trial: optuna.Trial) -> float:
        params, groups = self.suggest(trial)
        rels: list[float] = []

        def on_fold(k, loss, _m):
            rels.append(loss / self.base_fold[k])
            trial.report(float(np.mean(rels)), len(rels) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        o = E.run_oof(self.factory(params), self.fs, self.task, self.folds,
                      self.features_for(groups), on_fold)
        ts, loss = E.ts_loss(o)
        np.savez_compressed(self.dir / f"trial_{trial.number}.npz", ts=ts.astype("int64"), loss=loss)
        trial.set_user_attr("groups", groups)
        trial.set_user_attr("params", params)
        trial.set_user_attr("score", {k: v for k, v in E.score(o).items() if k != "per_fold"})
        trial.set_user_attr("seconds", o.seconds)
        return float(np.mean(rels))

    def study(self) -> optuna.Study:
        return optuna.create_study(
            study_name=self.study_name, storage=storage_url(), direction="minimize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=None, multivariate=True, n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))

    # -------------------------------------------------------------- champion / challenger
    def champion_path(self) -> Path:
        return champions_dir() / f"{self.task.name}.json"

    def load_champion(self) -> dict | None:
        p = self.champion_path()
        if not p.exists():
            return None
        c = json.loads(p.read_text())
        return c if c.get("data_hash") == self.data_hash else None   # new data -> re-earn the title

    def maybe_promote(self, st: optuna.Study) -> bool:
        done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            return False
        best = min(done, key=lambda t: t.value)
        f = self.dir / f"trial_{best.number}.npz"
        if not f.exists():          # completed on another worker's disk
            return False
        z = np.load(f)
        ts = z["ts"].astype("datetime64[ns]")
        champ = self.load_champion()
        if champ and champ.get("trial") == best.number and champ.get("study") == self.study_name:
            return False
        if champ:
            cz = np.load(champions_dir() / f"{self.task.name}.npz")
            c_ts, c_loss, c_name = cz["ts"].astype("datetime64[ns]"), cz["loss"], champ["model"]
        else:
            c_ts, c_loss, c_name = self.base_ts.astype("datetime64[ns]"), self.base_loss, self.best_base
        dm = E.compare_series(ts, z["loss"], c_ts, c_loss, self.task.horizon_bars)
        vs_base = E.compare_series(ts, z["loss"], self.base_ts.astype("datetime64[ns]"),
                                   self.base_loss, self.task.horizon_bars)
        ok = dm["rel"] <= -self.tc["min_rel_improvement"] and dm["dm_p"] < self.tc["promote_alpha"]
        n_total = len(st.trials)
        print(f"[{self.task.name}] challenger trial {best.number}: {dm['rel']:+.3%} vs {c_name} "
              f"(DM p={dm['dm_p']:.3g}) -> {'PROMOTE' if ok else 'keep champion'}", flush=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if not ok:
            log_line(today, "lightgbm", self.task.name, n_total, f"{best.value:.4f} rel",
                     f"{dm['rel']:+.2%} vs {c_name}, p={dm['dm_p']:.2g} (not promoted)", "—")
            return False
        params, groups = best.user_attrs["params"], best.user_attrs["groups"]
        model = self.refit(params, groups)
        model.save(champions_dir() / f"{self.task.name}.pkl")
        np.savez_compressed(champions_dir() / f"{self.task.name}.npz", ts=z["ts"], loss=z["loss"])
        score = best.user_attrs["score"]
        beats = vs_base["dm_p"] < self.tc["report_alpha"] and vs_base["rel"] < 0
        champ_new = {
            "task": self.task.name, "model": f"lightgbm#{best.number}", "study": self.study_name,
            "trial": best.number, "params": params, "groups": groups, "data_hash": self.data_hash,
            "features_hash": config_hash(self.cfg["features_cfg"]), "score": score,
            "vs_best_baseline": {"baseline": self.best_base, **vs_base, "significant": beats},
            "vs_previous": {"previous": c_name, **dm}, "n_trials_total": n_total,
            "promoted_at": datetime.now(UTC).isoformat(), "quantiles": list(self.task.quantiles),
            "horizon_min": self.task.horizon_min, "kind": self.task.kind,
        }
        self.champion_path().write_text(json.dumps(champ_new, indent=2, default=float))
        concl = (f"Beats {self.best_base} (p={vs_base['dm_p']:.2g})." if beats else
                 f"Does NOT significantly beat {self.best_base} (p={vs_base['dm_p']:.2g}); "
                 "treat as no edge.")
        bs = self.base[self.best_base][1]
        card = write_card(self.task, f"lightgbm#{best.number}", {**score, "per_fold": {}},
                          self.best_base, bs, vs_base,
                          {"data_hash": self.data_hash, "config_hash": config_hash(params),
                           "device": self.tc.get("device"), "seconds": best.user_attrs["seconds"],
                           "n_folds": len(self.folds), "embargo_min": self.cfg["eval"]["embargo_min"],
                           "n_trials_total": n_total, "params": params, "groups": groups}, concl)
        log_line(today, "lightgbm", self.task.name, n_total, f"{best.value:.4f} rel",
                 f"{vs_base['rel']:+.2%} vs {self.best_base}, p={vs_base['dm_p']:.2g}", card.name)
        # exploit: queue small perturbations of the new champion for the next round
        for scale in (0.8, 1.25):
            st.enqueue_trial({**{k: (type(v)(v * scale) if k in ("num_leaves", "min_data_in_leaf",
                              "learning_rate") else v) for k, v in params.items()},
                              **{f"use_{g}": g in groups for g in self.groups}}, skip_if_exists=True)
        return True

    def refit(self, params: dict, groups: list[str]) -> LGBMQuantile:
        from datetime import timedelta

        from forecast.evaluation.splits import inner_split, research_mask
        from forecast.utils.timeutil import parse_duration
        e = self.cfg["eval"]
        df = self.fs.df
        y = df[self.task.label_col].to_numpy()
        idx = np.flatnonzero(research_mask(df, parse_duration(e["final_holdout"]),
                                           timedelta(minutes=e["embargo_min"])) & np.isfinite(y))
        fit, es = inner_split(df, idx)
        feats = self.features_for(groups)
        m = self.factory(params)()
        X = df.select(feats)
        m.fit(X[fit], y[fit], X[es], y[es], self.task)
        return m


def run(cfg: dict, promote: bool = True, rounds: int | None = None) -> None:
    fs, data_hash = E.prepare(cfg)
    tuners = [TaskTuner(cfg, fs, data_hash, t) for t in cfg["tune"]["tasks"]]
    rounds = cfg["tune"]["rounds"] if rounds is None else rounds
    r = 0
    while rounds == 0 or r < rounds:
        for tt in tuners:
            st = tt.study()
            t0 = time.time()
            st.optimize(tt.objective, n_trials=cfg["tune"]["trials_per_round"], catch=(Exception,),
                        gc_after_trial=True)
            print(f"[{tt.task.name}] round {r}: {len(st.trials)} trials total, best rel "
                  f"{st.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if promote:
                tt.maybe_promote(st)
        r += 1
