"""Run one job end to end and return its result record.

A job is a small dict (see jobs/README.md). Every job is evaluated on the same purged walk-forward
folds, against baselines fitted on the same folds, and never sees the final holdout.

  python -m forecast.loop.run_job jobs/queued/<id>.yaml
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from forecast.data.sources import load_bars, resolve_symbols
from forecast.evaluation import metrics as M
from forecast.evaluation.splits import holdout_start, walk_forward
from forecast.features.build import CHANNELS, TAB, Dataset, build
from forecast.models.baselines import HarRV, Unconditional
from forecast.models.patchnet import PatchNet, n_params
from forecast.serving.export import export_onnx
from forecast.training.compress import magnitude_masks, nonzero_params
from forecast.training.trainer import fit, predict
from forecast.utils.common import DAY, REPO, device_info, git_sha, interval_seconds, seed_everything


def load_cfg(repo: Path = REPO, name: str = "loop") -> dict:
    """``configs/<name>.yaml``. ``loop`` is the daily track every worker runs by default; a job picks
    another track with ``config: <name>`` (``loop_minute``). The name comes from a job file, so it may
    only be a plain file stem: nothing outside ``configs/`` can be named."""
    if not re.fullmatch(r"[a-z0-9_]{1,40}", str(name)):
        raise ValueError(f"not a config name: {name!r}")
    return yaml.safe_load((repo / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def job_cfg(job: dict, repo: Path = REPO) -> dict:
    """The config a job runs under. Tracks never compete: each config has its own ``eval.version``
    (a test keeps them distinct), and champions, proposals and leaderboards are per version."""
    return load_cfg(repo, job.get("config") or "loop")


def artifact_dir(run_id: str, repo: Path = REPO) -> Path | None:
    """Where a finished run's checkpoints live: committed champions first, then this machine's cache."""
    for base in (repo / "models", repo / "artifacts"):
        if (base / run_id / "fold0.pt").exists():
            return base / run_id
    return None


def load_model(path: Path) -> PatchNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = PatchNet(**ck["cfg"])
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return m


def save_model(model: PatchNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"cfg": model.cfg, "state": state}, path)


def _slice(ds: Dataset, idx: np.ndarray) -> dict:
    return {"X": ds.X.subset(idx), "tab": ds.tab[idx], "y": ds.y[idx]}


def bar_seconds(data_cfg: dict) -> int | None:
    """Bar length for intraday data, None for daily bars (the Alpaca source is minute bars)."""
    default = "1m" if data_cfg.get("source") == "alpaca" else "1d"
    sec = interval_seconds(str(data_cfg.get("interval", default)))
    return sec if sec < DAY else None


def prepare(cfg: dict, lookback: int) -> tuple[Dataset, list, str]:
    bars = load_bars(cfg["data"])
    closes = bars["close"].to_numpy()
    dhash = hashlib.sha256(f"{bars.height}|{bars['t'].max()}|{closes.sum():.6f}".encode()).hexdigest()[:12]
    ut = np.unique(bars["t"].dt.epoch("s").to_numpy())
    hs = holdout_start(ut, cfg["eval"]["holdout_frac"])
    max_lb = int(cfg["task"]["max_lookback"])
    if lookback > max_lb:
        raise ValueError(f"lookback {lookback} exceeds task.max_lookback {max_lb}")
    ds = build(bars, cfg["task"]["horizons"], lookback, cfg["data"].get("benchmark", "SPY"),
               max_t_end=hs, first_index=max_lb - 1, bar_seconds=bar_seconds(cfg["data"]))
    ev = cfg["eval"]
    folds = walk_forward(ds.t, ds.t_end, ev["n_folds"], ev["val_frac"], ev["embargo_bars"], ev["es_frac"])
    if not folds:
        raise ValueError("no folds: not enough history")
    return ds, folds, dhash


def run_job(job: dict, cfg: dict | None = None, repo: Path = REPO, device: str | None = None) -> dict:
    t0 = time.time()
    cfg = cfg or job_cfg(job, repo)
    info = device_info()
    device = device or info["device"]
    res: dict = {"id": job["id"], "kind": job.get("kind", "train"), "tier": job["tier"], "job": job,
                 "config": job.get("config") or "loop", "git": git_sha(repo), "gpu": info["gpu"],
                 "eval_version": cfg["eval"].get("version", 1), "device": device, "status": "error"}
    try:
        seed_everything(int(job.get("seed", 1337)))
        tier = cfg["tiers"][job["tier"]]
        mcfg = {**tier["model"], **(job.get("model") or {})}
        tcfg = {**cfg["train"], **(job.get("train") or {})}
        Q = list(cfg["task"]["quantiles"])
        teacher_dir = init_dir = None
        lookback = int(mcfg["lookback"])
        if job.get("teacher"):
            teacher_dir = artifact_dir(job["teacher"], repo)
            if teacher_dir is None:
                raise FileNotFoundError(f"teacher artifacts for {job['teacher']} not on this machine")
            lookback = max(lookback, load_model(teacher_dir / "fold0.pt").lookback)
        if job.get("init_from"):
            init_dir = artifact_dir(job["init_from"], repo)
            if init_dir is None:
                raise FileNotFoundError(f"parent artifacts for {job['init_from']} not on this machine")
            # The model that is pruned / finetuned is the parent, so its shape (not the tier preset) is
            # what this job trains and reports, and its lookback is what the windows must be built at.
            # Until 2026-11-26 the windows used the preset's lookback: a 256-bar nano champion was
            # finetuned and scored on 64-bar windows, and every prune result recorded the preset shape.
            parent = load_model(init_dir / "fold0.pt").cfg
            mcfg = {k: parent.get(k, v) for k, v in mcfg.items()}
            lookback = max(lookback, int(mcfg["lookback"])) if teacher_dir else int(mcfg["lookback"])
        # How many symbols were asked for (``symbols`` further down is how many had usable history) and
        # the config's caveat about how the universe was chosen, so every result carries it.
        res["universe"] = {"requested": len(resolve_symbols(cfg["data"])),
                           "note": cfg["data"].get("universe_note")}
        ds, folds, dhash = prepare(cfg, lookback)
        out_dir = repo / "artifacts" / job["id"]
        if out_dir.exists():
            shutil.rmtree(out_dir)
        budget = float(job.get("max_minutes", tier["max_minutes"])) * 60.0
        per_fold = budget / len(folds)
        L_model, L_unc, L_har, T_all, cov, ics, stats = [], [], [], [], [], [], []
        model = None
        for k, f in enumerate(folds):
            tr, es, va = _slice(ds, f.train), _slice(ds, f.es), _slice(ds, f.val)
            fit_all = {key: np.concatenate([tr[key], es[key]]) for key in ("tab", "y")}  # baselines ignore X
            unc, har = Unconditional(Q), HarRV(Q)
            unc.fit(fit_all)
            har.fit(fit_all)
            if init_dir is not None:
                model = load_model(init_dir / f"fold{k}.pt")
            else:
                model = PatchNet(len(CHANNELS), len(TAB), len(ds.horizons), len(Q), **mcfg)
            teacher_pred = None
            if teacher_dir is not None:
                teacher = load_model(teacher_dir / f"fold{k}.pt").to(device)
                teacher_pred = predict(teacher, tr["X"], tr["tab"], device)
                del teacher
            masks = None
            if job.get("sparsity"):
                masks = magnitude_masks(model.to(device), float(job["sparsity"]))
            st = fit(model, tr, es, tcfg, Q, device, teacher_pred=teacher_pred, masks=masks,
                     deadline=time.time() + per_fold)
            pv = predict(model, va["X"], va["tab"], device)
            L_model.append(M.pinball_rows(va["y"], pv, Q))
            L_unc.append(M.pinball_rows(va["y"], unc.predict(va["X"], va["tab"]), Q))
            L_har.append(M.pinball_rows(va["y"], har.predict(va["X"], va["tab"]), Q))
            T_all.append(ds.t[f.val])
            cov.append(M.coverage(va["y"], pv))
            ics.append(M.rank_ic(va["y"][:, 0, 0], pv[:, 0, 0, len(Q) // 2], ds.t[f.val]))
            stats.append({**st, "n_train": int(len(f.train)), "n_val": int(len(f.val))})
            save_model(model, out_dir / f"fold{k}.pt")
        lm, lu, lh, tt = (np.concatenate(a) for a in (L_model, L_unc, L_har, T_all))
        ret_m, ret_b = lm[..., 0].mean(), lu[..., 0].mean()
        vol_m, vol_b = lm[..., 1].mean(), min(lu[..., 1].mean(), lh[..., 1].mean())
        if not np.isfinite([ret_m, ret_b, vol_m, vol_b]).all():
            # A diverged model is a failed run, not a score (a NaN score could never be beaten or ranked).
            raise FloatingPointError(f"non-finite loss (ret {ret_m}, vol {vol_m}): the model diverged")
        best_base = np.stack([lu[..., 0], np.where(lh[..., 1].mean() <= lu[..., 1].mean(), lh[..., 1], lu[..., 1])],
                             axis=-1)
        # Newey-West lag in forecast timestamps: labels overlap for max(horizons) - 1 bars. A config may
        # ask for more (minute bars: a whole session, since losses within one day move together).
        hmax = int(cfg["eval"].get("dm_lag", max(ds.horizons)))
        dm_all = M.diebold_mariano(lm.mean((1, 2)), best_base.mean((1, 2)), tt, lag=hmax)
        dm_ret = M.diebold_mariano(lm[..., 0].mean(1), best_base[..., 0].mean(1), tt, lag=hmax)
        dm_vol = M.diebold_mariano(lm[..., 1].mean(1), best_base[..., 1].mean(1), tt, lag=hmax)
        onnx = export_onnx(model, out_dir / "model.onnx", len(TAB))
        model.to("cpu")
        peak = max(s["peak_vram_gb"] for s in stats)
        res.update({
            "status": "ok", "model": mcfg, "train": tcfg, "data_hash": dhash,
            "symbols": len(ds.symbols), "horizons": ds.horizons, "folds": stats,
            "pinball": {"ret_model": float(ret_m), "ret_baseline": float(ret_b),
                        "vol_model": float(vol_m), "vol_baseline": float(vol_b),
                        "vol_har": float(lh[..., 1].mean()), "vol_unconditional": float(lu[..., 1].mean())},
            "skill_ret": float(1 - ret_m / ret_b), "skill_vol": float(1 - vol_m / vol_b),
            "score": float(0.5 * (1 - ret_m / ret_b) + 0.5 * (1 - vol_m / vol_b)),
            "dm": {"all": dm_all, "ret": dm_ret, "vol": dm_vol},
            "coverage_80": np.mean(cov, axis=0).round(4).tolist(),
            "rank_ic_h1": float(np.mean(ics)),
            "params": n_params(model), "nonzero_params": nonzero_params(model),
            "peak_vram_gb": peak, **onnx,
        })
        res["fits_tier"] = fits_tier(res, tier)
    except Exception as e:  # a failed job is a result too; the worker must keep going
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-2000:]
    res["minutes"] = round((time.time() - t0) / 60, 2)
    return res


def fits_tier(res: dict, tier: dict) -> bool:
    return (res["params"] <= tier.get("max_params", 10**12)
            and res["onnx_bytes"] <= tier.get("max_onnx_bytes", 10**12)
            and res["cpu_ms_per_forecast"] <= tier.get("max_cpu_ms", 10**9)
            and res["peak_vram_gb"] <= tier.get("max_train_vram_gb", 10**9))


if __name__ == "__main__":
    job = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = run_job(job)
    print(yaml.safe_dump({k: v for k, v in out.items() if k not in ("job", "traceback")}, sort_keys=False))
    sys.exit(0 if out["status"] == "ok" else 1)
