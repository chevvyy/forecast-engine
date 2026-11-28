"""Derive champions and the leaderboard from results/*.json. Nothing here is stored state: any machine
can recompute it, so two workers never fight over a shared file.

  python -m forecast.loop.leaderboard          # rewrites experiments/LEADERBOARD.md
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from forecast.utils.common import REPO, read_json

TIERS = ["nano", "mid", "large"]


def load_results(repo: Path = REPO) -> list[dict]:
    out = []
    for p in sorted((repo / "results").glob("*.json")):
        try:
            out.append(read_json(p))
        except Exception:
            continue
    return out


def champions(results: list[dict], eval_version: int | None = None) -> dict[str, dict]:
    """Best run per tier by score among runs that finished and fit the tier budget.
    ``score`` is the mean skill against the best baseline on returns and on volatility."""
    best: dict[str, dict] = {}
    for r in results:
        if r.get("status") != "ok" or not r.get("fits_tier"):
            continue
        if not math.isfinite(r.get("score", math.nan)):
            continue   # nothing compares greater than NaN: such a champion would never be replaced
        if eval_version is not None and r.get("eval_version", 1) != eval_version:
            continue   # scores from a different evaluation protocol are not comparable
        t = r["tier"]
        if t not in best or r["score"] > best[t]["score"]:
            best[t] = r
    return best


def config_key(r: dict) -> str:
    """What a run trained, without its seed: tier, model shape, training settings as the run recorded
    them (presets and defaults already filled in), teacher, parent and sparsity. ``kind`` is left out
    because it is only a label (jobs/README.md). Results are worker-written data: anything that is not
    the expected type is ignored, which at worst puts a malformed run in a group of its own."""
    job = r.get("job") if isinstance(r.get("job"), dict) else {}
    part = {k: r.get(k) if isinstance(r.get(k), dict) else job.get(k) for k in ("model", "train")}
    return json.dumps({"tier": r.get("tier"), **part,
                       **{k: job.get(k) for k in ("teacher", "init_from", "sparsity")}},
                      sort_keys=True, default=str)


def configurations(results: list[dict], eval_version: int | str | None = None, min_seeds: int = 2) -> list[dict]:
    """Runs grouped into configurations (``config_key``) and scored by the MEAN over their seeds.

    The single-run board ranks the luckiest seed of everything tried, so its top score overstates the
    configuration behind it (by about two seed-sd on 2026-11-25). A configuration counts here only with
    ``min_seeds`` different seeds; two runs with one seed are one draw (before 2026-11-26 distil jobs had
    no seed and repeated score for score), and the first to finish is kept. Same filter as ``champions``:
    finished, inside the tier budget, finite score, this ``eval_version``. ``sd`` is the sample standard
    deviation. Best mean first; ``runs`` are in order of finishing, so ``runs[0]`` is the first run."""
    groups: dict[str, dict[int, dict]] = {}
    for r in sorted(results, key=lambda r: str(r.get("finished", ""))):
        if r.get("status") != "ok" or not r.get("fits_tier"):
            continue
        if not math.isfinite(r.get("score", math.nan)):
            continue
        if eval_version is not None and r.get("eval_version", 1) != eval_version:
            continue
        job = r.get("job") if isinstance(r.get("job"), dict) else {}
        try:
            seed = int(job.get("seed", 1337))
        except (TypeError, ValueError):
            continue
        groups.setdefault(config_key(r), {}).setdefault(seed, r)
    out = []
    for by_seed in groups.values():
        runs = list(by_seed.values())
        if len(runs) < max(min_seeds, 1):
            continue
        scores = [r["score"] for r in runs]
        def mean(k: str, runs: list[dict] = runs) -> float:
            vals = [r[k] for r in runs if isinstance(r.get(k), int | float) and math.isfinite(r[k])]
            return statistics.fmean(vals) if vals else math.nan
        out.append({"tier": runs[0]["tier"], "n": len(runs), "mean": statistics.fmean(scores),
                    "sd": statistics.stdev(scores) if len(runs) > 1 else math.nan,
                    "min": min(scores), "max": max(scores),
                    "skill_ret": mean("skill_ret"), "skill_vol": mean("skill_vol"),
                    "best": max(runs, key=lambda r: r["score"]), "runs": runs})
    return sorted(out, key=lambda c: -c["mean"])


# A configuration takes a tier over only if its seed mean beats the holder's by more than this many
# standard errors of the difference of two seed means (``search.promote_z`` overrides).
PROMOTE_Z = 2.0


def promoted(results: list[dict], eval_version: int | str | None = None, min_seeds: int = 3,
             z: float = PROMOTE_Z) -> dict[str, dict]:
    """Per tier, the configuration that holds it once seed noise is taken into account.

    ``champions`` gives a tier to the best single run, which is the luckiest seed of everything tried,
    and to a configuration whose mean is 0.001 above another's although one seed moves a score by
    0.002 to 0.004. Here a configuration counts only once it has ``min_seeds`` different seeds, and it
    takes the tier from the holder only if its seed mean is higher by more than
    ``z * noise * sqrt(1/n_new + 1/n_holder)``, the standard error of the difference of the two means.
    ``noise`` is the MEDIAN seed sd of the tier's confirmed configurations: one sd from 3 seeds is too
    rough to use alone (0.0002 to 0.010 on 2026-11-27), and the median keeps one unstable configuration
    from setting the bar for all the others.

    Nothing is stored: configurations are taken in the order in which their ``min_seeds``-th seed
    finished, the first one holds the tier, and each later one is compared with the holder of that
    moment using the means over all seeds on record now. So the answer can depend on the order (a
    configuration just inside the margin never takes over, and a later one must beat the holder, not
    it), which is what a threshold means. The first run of a configuration that topped the single-run
    board is part of its mean, so a mean still flatters a configuration found by search; ``z`` = 2
    leaves room for that. A tier without a confirmed configuration is absent from the result.
    Each value is the ``configurations`` entry plus ``noise``, ``margin`` (what the next challenger of
    the same seed count must exceed) and ``challengers`` (confirmed, higher mean, inside the margin)."""
    if min_seeds < 2 or not math.isfinite(z) or z < 0:
        return {}
    confs = configurations(results, eval_version, min_seeds=min_seeds)
    out: dict[str, dict] = {}
    for t in TIERS:
        cs = sorted((c for c in confs if c["tier"] == t),
                    key=lambda c: (str(c["runs"][min_seeds - 1].get("finished", "")), str(c["runs"][0].get("id"))))
        if not cs:
            continue
        noise = statistics.median(c["sd"] for c in cs)
        def margin(a: dict, b: dict, noise: float = noise) -> float:
            return z * noise * math.sqrt(1 / a["n"] + 1 / b["n"])
        holder = cs[0]
        for c in cs[1:]:
            if c["mean"] > holder["mean"] + margin(c, holder):
                holder = c
        out[t] = {**holder, "noise": noise, "margin": margin(holder, holder),
                  "challengers": [c for c in cs if c is not holder and c["mean"] > holder["mean"]]}
    return out


def promotion_settings(cfg: dict) -> tuple[int, float]:
    """``(min_seeds, z)`` for ``promoted`` from a track's config: ``search.confirm_seeds`` (the seeds a
    champion is re-run to anyway) and ``search.promote_z``. Anything unreadable gives the defaults."""
    s = cfg.get("search") if isinstance(cfg.get("search"), dict) else {}
    try:
        seeds = int(s.get("confirm_seeds", 3))
    except (TypeError, ValueError):
        seeds = 3
    try:
        z = float(s.get("promote_z", PROMOTE_Z))
    except (TypeError, ValueError):
        z = PROMOTE_Z
    return seeds, z


def _made_by(r: dict) -> str:
    job = r.get("job") if isinstance(r.get("job"), dict) else {}
    if job.get("init_from"):
        return f"prune {job.get('sparsity')} of {job['init_from']}"
    return f"distilled from {job['teacher']}" if job.get("teacher") else "trained directly"


def render(results: list[dict], eval_version: int | None = None, min_seeds: int = 3,
           z: float = PROMOTE_Z) -> str:
    ok = [r for r in results if r.get("status") == "ok"
          and (eval_version is None or r.get("eval_version", 1) == eval_version)]
    ch = champions(results, eval_version)
    failed = sum(1 for r in results if r.get("status") != "ok")
    other = len(results) - len(ok) - failed   # finished under another eval.version: not comparable
    head = f"{len(results)} runs recorded, {len(ok)} finished, {failed} failed."
    if eval_version is not None:
        head = (f"Evaluation version {eval_version}: {len(ok)} finished runs. All versions: {len(results)} runs "
                f"recorded, {failed} failed, {other} finished under other versions (not comparable, not shown).")
    lines = ["# Leaderboard", "", head,
             "Skill = 1 - model pinball / best-baseline pinball on purged walk-forward validation folds.",
             "Positive means better than the baseline. The final holdout is not used here.",
             "Every run in this table was selected on the same validation folds, so the best score is",
             "biased upward by the number of runs tried; only the final holdout settles it.", "",
             "## Champions", "",
             "| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    def row(r: dict) -> str:
        return (f"| {r['tier']} | {r['id']} | {r['score']:+.4f} | {r['skill_ret']:+.4f} | {r['skill_vol']:+.4f} "
                f"| {r['dm']['all'][1]:.3f} | {r['params']:,} | {r['onnx_bytes'] / 1024:.0f} "
                f"| {r['cpu_ms_per_forecast']} | {r['peak_vram_gb']} |")
    lines += [row(ch[t]) for t in TIERS if t in ch] or ["| (none yet) | | | | | | | | | |"]
    for t in TIERS:
        top = sorted((r for r in ok if r["tier"] == t), key=lambda r: -r["score"])[:10]
        if top:
            lines += ["", f"## Top {t} runs", "",
                      "| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |",
                      "|---|---|---|---|---|---|---|---|---|---|"] + [row(r) for r in top]
    confs = configurations(results, eval_version)
    if confs:
        lines += ["", "## Configurations by seed mean", "",
                  "One row per configuration (same shape, training settings, teacher / parent) that ran with at",
                  "least 2 different seeds, ranked by the mean over its seeds: the tables above rank single runs,",
                  "and the best single run is the luckiest seed. A score difference smaller than the sd column is",
                  "noise. Up to 10 per tier. Still selected on the validation folds, like everything here."]
    for t in TIERS:
        top = [c for c in confs if c["tier"] == t][:10]
        if top:
            lines += ["", f"### {t}", "",
                      "| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |",
                      "|---|---|---|---|---|---|---|---|---|"]
            lines += [f"| {c['mean']:+.4f} | {c['sd']:.4f} | {c['n']} | {c['min']:+.4f} | {c['max']:+.4f} "
                      f"| {c['skill_ret']:+.4f} | {c['skill_vol']:+.4f} | {c['best']['id']} | {_made_by(c['best'])} |"
                      for c in top]
    held = promoted(results, eval_version, min_seeds, z)
    if held:
        lines += ["", "## Tier holders after seed noise", "",
                  f"The configuration that holds each tier when a newcomer needs {min_seeds} seeds and a seed mean",
                  f"above the holder's by more than {z:g} standard errors of the difference (noise = median seed",
                  "sd of the tier's confirmed configurations). Workers mutate, distil and prune from these, not",
                  "from the luckiest single run. 'Inside the margin' lists confirmed configurations with a higher",
                  "mean that did not clear it: not shown to be better, not shown to be worse.", "",
                  "| Tier | Mean score | sd | Seeds | Best run | How it was made | Noise | Margin | Inside the margin |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for t in TIERS:
            if t in held:
                h = held[t]
                inside = ", ".join(f"{c['best']['id']} ({c['mean']:+.4f} over {c['n']})" for c in h["challengers"])
                lines.append(f"| {t} | {h['mean']:+.4f} | {h['sd']:.4f} | {h['n']} | {h['best']['id']} "
                             f"| {_made_by(h['best'])} | {h['noise']:.4f} | {h['margin']:.4f} | {inside or '-'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    from forecast.loop.run_job import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="loop", help="track: loop (daily, default) or e.g. loop_minute")
    name = ap.parse_args().config
    cfg = load_cfg(REPO, name)
    ver = cfg["eval"].get("version", 1)
    text = render(load_results(), ver, *promotion_settings(cfg))
    # One file per track, so minute and daily scores are never read side by side as if comparable.
    out = "LEADERBOARD.md" if name == "loop" else f"LEADERBOARD_{name.removeprefix('loop_')}.md"
    (REPO / "experiments" / out).write_text(text, encoding="utf-8")
    print(text)
