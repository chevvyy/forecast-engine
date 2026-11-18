"""Result cards (markdown) + append-only experiment log."""
from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from forecast.utils.config import REPO_ROOT
from forecast.utils.env import experiments_dir


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _f(x, nd=6):
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def write_card(task, model_name: str, model_score: dict, base_name: str, base_score: dict,
               dm: dict, meta: dict, conclusion: str, tag: str = "") -> Path:
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    path = experiments_dir() / "cards" / f"{day}-{model_name}-{task.name}{tag}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ("qlike", "crps", "mse_log_rv", "coverage_80", "rank_ic", "hit_rate", "hit_p")
            if k in model_score]
    rows = "\n".join(f"| {k} | {_f(model_score[k])} | {_f(base_score.get(k, float('nan')))} |"
                     for k in keys)
    folds = " · ".join(f"{k}: {_f(v, 4)}" for k, v in model_score.get("per_fold", {}).items())
    bfolds = " · ".join(f"{k}: {_f(v, 4)}" for k, v in base_score.get("per_fold", {}).items())
    path.write_text(f"""# {model_name} — {task.name} ({task.kind}, {task.horizon_min} min)
- Git: {git_sha()}    Data hash: {meta.get('data_hash')}    Config hash: {meta.get('config_hash')}
- Device: {meta.get('device')}    Wall time: {meta.get('seconds', 0):.0f}s    Folds: {meta.get('n_folds')}
  walk-forward, purge=label interval, embargo={meta.get('embargo_min')} min
- Configurations tried so far for this task (feeds DSR): {meta.get('n_trials_total')}
- Params: `{meta.get('params')}`
- Feature groups: {meta.get('groups')}

| Metric | {model_name} | {base_name} |
|---|---|---|
{rows}

Primary loss ({'QLIKE' if task.kind == 'rv' else 'CRPS'}) vs **{base_name}**: Δ = {dm['rel']:+.2%},
Diebold–Mariano (HAC, per-timestamp) stat = {dm['dm_stat']:.2f}, one-sided p = {dm['dm_p']:.3g},
n timestamps = {dm['n_ts']}.

Per-fold loss — model: {folds}
Per-fold loss — baseline: {bfolds}

Conclusion: {conclusion}
""")
    return path


def log_line(date: str, model: str, target: str, n_configs: int, best: str, vs: str, card: str) -> None:
    p = experiments_dir() / "LOG.md"
    with p.open("a") as f:
        f.write(f"| {date} | {model} | {target} | {n_configs} | {best} | {vs} | {card} |\n")
