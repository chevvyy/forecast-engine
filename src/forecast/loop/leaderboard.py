"""Derive champions and the leaderboard from results/*.json. Nothing here is stored state: any machine
can recompute it, so two workers never fight over a shared file.

  python -m forecast.loop.leaderboard          # rewrites experiments/LEADERBOARD.md
"""
from __future__ import annotations

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
        if eval_version is not None and r.get("eval_version", 1) != eval_version:
            continue   # scores from a different evaluation protocol are not comparable
        t = r["tier"]
        if t not in best or r["score"] > best[t]["score"]:
            best[t] = r
    return best


def render(results: list[dict], eval_version: int | None = None) -> str:
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
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import yaml

    ver = yaml.safe_load((REPO / "configs" / "loop.yaml").read_text(encoding="utf-8"))["eval"].get("version", 1)
    text = render(load_results(), ver)
    (REPO / "experiments" / "LEADERBOARD.md").write_text(text, encoding="utf-8")
    print(text)
