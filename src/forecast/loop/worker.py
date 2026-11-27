"""GPU worker daemon. Run one per machine; the git branch is the only thing workers share.

  python -m forecast.loop.worker --name pc-5070ti --tiers large,mid,nano

Loop: pull -> claim a queued job that fits this machine (or invent one when the queue is empty; a
share of invented jobs goes to other tracks, see ``pick_track``) -> run it -> commit the result -> push. Files are one-per-job and one-per-worker, so pushes from two
machines never conflict; a lost claim race is detected by the rejected push and simply retried.
Exit code 3 means "code changed upstream, restart me" (the run_worker scripts do that).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from forecast.data import alpaca
from forecast.loop.leaderboard import champions, load_results
from forecast.loop.run_job import artifact_dir, job_cfg, load_cfg, run_job
from forecast.models.patchnet import train_vram_gb
from forecast.utils.common import REPO, device_info, release_vram, write_json

RESTART = 3


class Git:
    def __init__(self, repo: Path, branch: str, enabled: bool = True):
        self.repo, self.branch, self.enabled = repo, branch, enabled

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, check=check)

    def sync(self) -> None:
        """Make the working tree equal to the remote branch. Local artifacts/ and data/ are ignored
        by git and survive; anything else local is discarded on purpose."""
        if not self.enabled:
            return
        self.run("fetch", "origin", self.branch)
        self.run("reset", "--hard", f"origin/{self.branch}")

    def src_rev(self) -> str:
        if not self.enabled:
            return "local"
        return self.run("rev-parse", "HEAD:src", "HEAD:pyproject.toml").stdout

    def publish(self, message: str, paths: list[str]) -> bool:
        """Commit ``paths`` and push. Returns False if the remote moved in a way we cannot rebase onto
        (for a claim that means another worker took the job)."""
        if not self.enabled:
            return True
        self.run("add", "-A", "--", *paths)
        if self.run("diff", "--cached", "--quiet", check=False).returncode == 0:
            return True
        self.run("commit", "-q", "-m", message)
        for attempt in range(6):
            if self.run("push", "origin", f"HEAD:{self.branch}", check=False).returncode == 0:
                return True
            time.sleep(2 + 3 * attempt)
            self.run("fetch", "origin", self.branch, check=False)
            if self.run("rebase", f"origin/{self.branch}", check=False).returncode != 0:
                self.run("rebase", "--abort", check=False)
                self.run("reset", "--hard", f"origin/{self.branch}")
                return False
        self.run("reset", "--hard", f"origin/{self.branch}")
        return False


def error_pause(fails: int) -> int:
    """Seconds to wait after ``fails`` failed jobs in a row. One failure costs nothing (a bad job must not
    slow the queue); a run of them is a broken machine or broken code, and retrying every few seconds
    only floods the branch with identical error results. Capped so a fix is picked up within 15 minutes."""
    return 0 if fails < 2 else min(60 * 2 ** (fails - 2), 900)


def new_id(worker: str, rng: random.Random) -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{worker}-{rng.randrange(16**4):04x}"


# A distil or prune configuration is run with at most this many seeds per eval.version. Before
# 2026-11-26 these jobs had no seed at all: 142 distil runs were 11 distinct jobs, score for score.
MAX_REPEATS = 3
# A self-invented job in a tier whose budget is at most this is not announced with a claim commit:
# nobody else can claim an id this worker just made up, and at one claim plus one result per job the
# short jobs wrote about 7 commits a minute (3000+ in four hours on 2026-11-25).
QUIET_CLAIM_MINUTES = 20


def signature(job: dict, cfg: dict, seed: bool = True) -> str:
    """What a job trains, with tier presets and train defaults filled in, so two jobs written
    differently but training the same thing compare equal. ``seed=False`` names the configuration."""
    tier = cfg["tiers"][job["tier"]]
    sig = {"tier": job["tier"], "kind": job.get("kind", "train"),
           "model": {**tier["model"], **(job.get("model") or {})},
           "train": {**cfg["train"], **(job.get("train") or {})},
           "teacher": job.get("teacher"), "init_from": job.get("init_from"), "sparsity": job.get("sparsity")}
    if seed:
        sig["seed"] = int(job.get("seed", 1337))
    return json.dumps(sig, sort_keys=True, default=str)


def fits_vram(job: dict, cfg: dict) -> bool:
    """False if training this shape would need more GPU memory than its tier allows (estimate)."""
    tier = cfg["tiers"][job["tier"]]
    limit = tier.get("max_train_vram_gb")
    if limit is None or job.get("init_from"):   # a pruned model has its parent's shape, which already ran
        return True
    batch = {**cfg["train"], **(job.get("train") or {})}.get("batch_size", 512)
    return train_vram_gb({**tier["model"], **(job.get("model") or {})}, batch) <= float(limit)


# The training settings a mutation or seed re-run takes over from the champion (the ones in the grid).
RECIPE_KEYS = ("lr", "weight_decay", "batch_size")


def recipe(champ: dict, results: list[dict], cfg: dict, repo: Path = REPO) -> dict:
    """How a champion was made, as job fields: its non-default ``train`` settings and its ``teacher``.

    Until 2026-11-27 a mutation kept the champion's model shape only, so every mutation went back to the
    default lr and batch size and a mutation of a distilled champion lost its teacher: the search could
    not climb from a champion whose advantage was in how it was trained. A pruned champion's own
    settings are finetuning settings, so its recipe is that of the run it descends from (``init_from``,
    followed through the results of this ``eval.version``); if that run is not on record the recipe is
    empty. Results are worker-written data: only values that are in the search grid are taken over, and a
    teacher only if it is a finished run of this version whose checkpoints are on this machine."""
    ver = cfg["eval"].get("version", 1)
    by_id = {r["id"]: r for r in results if isinstance(r, dict) and isinstance(r.get("id"), str)
             and r.get("status") == "ok" and r.get("eval_version", 1) == ver}
    run, seen = champ, set()
    while True:
        job = run.get("job") if isinstance(run.get("job"), dict) else {}
        parent = job.get("init_from")
        if not parent:
            break
        if not isinstance(parent, str) or parent in seen or parent not in by_id:
            return {"train": {}}
        seen.add(parent)
        run = by_id[parent]
    out: dict = {"train": {}}
    used = run.get("train") if isinstance(run.get("train"), dict) else {}
    for key in RECIPE_KEYS:
        v = used.get(key)
        grid = (cfg.get("search") or {}).get(key)
        if (isinstance(v, int | float) and not isinstance(v, bool) and isinstance(grid, list) and v in grid
                and v != cfg["train"].get(key)):
            out["train"][key] = grid[grid.index(v)]
    teacher = job.get("teacher")
    if isinstance(teacher, str) and teacher in by_id and artifact_dir(teacher, repo):
        out["teacher"] = teacher
    return out


# A tier's champion is the best single run, so it is also the luckiest seed of everything tried. Before
# the worker mutates it, its configuration is run until it has this many seeds (``search.confirm_seeds``
# overrides; 1 switches the re-runs off). Same number as ``MAX_REPEATS`` on purpose.
CONFIRM_SEEDS = 3


def _number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def rerun(champ: dict, results: list[dict], cfg: dict, repo: Path = REPO) -> dict | None:
    """The champion's own job again, without a seed: same tier, shape, training settings, teacher, parent
    and sparsity, so that its ``signature`` (and the leaderboard's configuration) is the champion's.

    Results are worker-written data, so the job is rebuilt field by field: ``model`` and ``train`` may
    only hold numbers under keys the tier preset / the train defaults already have, a teacher or parent
    must be a finished run of this ``eval.version`` whose checkpoints are on this machine, and a sparsity
    must lie inside (0, 1). Anything else returns None: the champion is then not re-run here (another
    machine, or the planner, may still do it) and the worker goes on to its usual proposals."""
    job = champ.get("job")
    tier = champ.get("tier")
    if not isinstance(job, dict) or tier not in cfg["tiers"]:
        return None
    ver = cfg["eval"].get("version", 1)
    ok = {r["id"] for r in results if isinstance(r, dict) and isinstance(r.get("id"), str)
          and r.get("status") == "ok" and r.get("eval_version", 1) == ver}
    kind = job.get("kind", "train")
    out: dict = {"tier": tier, "kind": kind if kind in ("train", "distill", "prune") else "train"}
    for key, known in (("model", cfg["tiers"][tier]["model"]), ("train", cfg["train"])):
        part = job.get(key)
        if part is None:
            continue
        if not isinstance(part, dict) or any(k not in known or not _number(v) for k, v in part.items()):
            return None
        out[key] = dict(part)
    for key in ("teacher", "init_from"):
        ref = job.get(key)
        if ref is None:
            continue
        if not isinstance(ref, str) or ref not in ok or not artifact_dir(ref, repo):
            return None
        out[key] = ref
    if job.get("sparsity") is not None:
        if not _number(job["sparsity"]) or not 0.0 < job["sparsity"] < 1.0:
            return None
        out["sparsity"] = float(job["sparsity"])
    return out


def _with_recipe(job: dict, rec: dict) -> dict:
    """A job trained the way the champion was: with its teacher, if it had one."""
    if rec.get("teacher"):
        job["kind"], job["teacher"] = "distill", rec["teacher"]
    return job


def _draw(cfg: dict, ch: dict, tier: str, done: int, rng: random.Random, repo: Path, rec: dict) -> dict:
    s, order = cfg["search"], ["nano", "mid", "large"]
    base = dict(ch[tier]["model"])
    roll = rng.random()
    bigger = [t for t in order[order.index(tier) + 1:] if t in ch and artifact_dir(ch[t]["id"], repo)]
    if roll < s["p_distill"] and bigger:
        return {"tier": tier, "kind": "distill", "teacher": ch[rng.choice(bigger)]["id"], "model": base,
                "note": f"distil into {tier} champion shape"}
    if roll < s["p_distill"] + s["p_prune"] and artifact_dir(ch[tier]["id"], repo):
        return {"tier": tier, "kind": "prune", "init_from": ch[tier]["id"],
                "sparsity": rng.choice(s["prune_sparsity"]), "train": {"lr": 0.0003, "max_epochs": 15},
                "note": "prune + finetune champion"}
    model, train = base, dict(rec["train"])   # the champion's recipe; a mutated key overrides it below
    explore = 3 if done < 12 else 2
    for key in rng.sample(["d_model", "n_blocks", "kernel", "lookback", "patch", "ffn", "dropout",
                           "lr", "weight_decay", "batch_size"], k=explore):
        opts = s[key][tier] if isinstance(s[key], dict) else s[key]
        v = rng.choice(opts)
        if key == "patch":
            model["patch"], model["stride"] = v
        elif key in ("lr", "weight_decay", "batch_size"):
            train[key] = v
        else:
            model[key] = v
    if model["lookback"] < 2 * model["patch"]:
        model["patch"], model["stride"] = 4, 2
    return _with_recipe({"tier": tier, "kind": "train", "model": model, "train": train,
                         "note": f"mutation of {ch[tier]['id']}"}, rec)


def propose(cfg: dict, results: list[dict], tiers: list[str], rng: random.Random, repo: Path = REPO) -> dict:
    """Invent the next experiment. First of all, a champion whose configuration has fewer than
    ``CONFIRM_SEEDS`` seeds is run again with a new seed (``rerun``): a champion is the best of many
    single runs, and until 2026-11-27 the worker mutated it for hours before the planner's seed re-runs
    said whether it was a step or a lucky draw. Failed runs count as seeds tried, so a configuration that
    cannot be re-run is given up after the same number of attempts. Otherwise:
    mutate the tier champion (keeping how it was trained, see
    ``recipe``), or distil a bigger champion into a smaller tier, or prune-and-finetune the champion.
    Every proposal carries a seed. A proposal is
    dropped and redrawn if the same job already has a result, if it is a distil / prune configuration
    that already ran ``MAX_REPEATS`` times, or if its training would not fit the tier's VRAM."""
    ver = cfg["eval"].get("version", 1)
    ch = champions(results, ver)
    done = sum(1 for r in results if r.get("status") == "ok")
    # Make sure every tier this worker serves has at least one run before exploring.
    missing = [t for t in tiers if t not in ch]
    if missing:
        return {"tier": missing[0], "kind": "train", "note": "tier preset (first run)"}
    exact: set[str] = set()
    repeats: dict[str, int] = {}
    seeds: dict[str, set[int]] = {}
    for r in results:
        if r.get("eval_version", 1) != ver:
            continue
        try:
            exact.add(signature(r["job"], cfg))
            key = signature(r["job"], cfg, seed=False)
        except (KeyError, TypeError, ValueError):
            continue   # results are worker-written data: skip anything malformed
        repeats[key] = repeats.get(key, 0) + 1
        seeds.setdefault(key, set()).add(int(r["job"].get("seed", 1337)))
    try:
        need = int((cfg.get("search") or {}).get("confirm_seeds", CONFIRM_SEEDS))
    except (TypeError, ValueError):
        need = CONFIRM_SEEDS
    pending = []
    for t in tiers:
        again = rerun(ch[t], results, cfg, repo)
        if again is not None and fits_vram(again, cfg):
            tried = seeds.get(signature(again, cfg, seed=False), set())
            if len(tried) < need:
                pending.append((again, tried, ch[t]["id"]))
    if pending:
        job, tried, champ_id = rng.choice(pending)
        job["seed"] = rng.choice([s for s in rng.sample(range(10**6), k=len(tried) + 1) if s not in tried])
        job["note"] = f"confirm champion {champ_id}: seed {len(tried) + 1} of {need}"
        return job
    tier = rng.choice(tiers)
    rec = recipe(ch[tier], results, cfg, repo)
    for _ in range(20):
        job = _draw(cfg, ch, tier, done, rng, repo, rec)
        job["seed"] = rng.randrange(10**6)
        if job["kind"] != "train" and repeats.get(signature(job, cfg, seed=False), 0) >= MAX_REPEATS:
            continue
        if signature(job, cfg) in exact or not fits_vram(job, cfg):
            continue
        return job
    # Nothing new that fits came up: another seed of the champion's configuration is still information.
    return _with_recipe({"tier": tier, "kind": "train", "model": dict(ch[tier]["model"]),
                         "train": dict(rec["train"]), "seed": rng.randrange(10**6),
                         "note": f"seed re-run of {ch[tier]['id']}"}, rec)


def track_ready(cfg: dict) -> bool:
    """Whether this machine can load a track's bars: the Alpaca source needs keys in ``.env``."""
    return cfg["data"].get("source") != "alpaca" or alpaca.available()


def pick_track(cfg: dict, rng: random.Random, repo: Path = REPO) -> tuple[str | None, dict]:
    """Which track the next self-invented job belongs to. ``search.tracks`` in the daily config gives
    other tracks a share of those jobs (``{loop_minute: 0.3}``); the rest stay daily. A track is only
    picked if its config loads, has its own ``search`` grid and ``eval.version``, and this machine can
    load its bars; otherwise the job is a daily one, so a worker without keys loses nothing.
    Returns ``(None, cfg)`` for the daily track, else the track's config name and config."""
    tracks = (cfg.get("search") or {}).get("tracks") or {}
    if not isinstance(tracks, dict) or not tracks:
        return None, cfg
    roll, acc = rng.random(), 0.0
    for name, share in tracks.items():
        try:
            acc += float(share)
            if roll >= acc:
                continue
            other = load_cfg(repo, name)
            if (other["eval"]["version"] != cfg["eval"].get("version", 1) and other["search"]
                    and set(other["tiers"]) >= set(cfg["tiers"]) and track_ready(other)):
                return str(name), other
        except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
            pass   # no such config on this branch, or an incomplete one: stay on the daily track
        break
    return None, cfg


def runnable(job: dict, cfg: dict, tiers: list[str], vram_free: float, repo: Path) -> bool:
    if job.get("tier") not in tiers:
        return False
    if float(job.get("min_vram_gb", cfg["tiers"][job["tier"]]["min_vram_gb"])) > vram_free + 1e-9:
        return False
    return all(artifact_dir(job[k], repo) for k in ("teacher", "init_from") if job.get(k))


def step(name: str, tiers: list[str], git: Git, rng: random.Random, repo: Path = REPO,
         device: str | None = None, allow_cpu: bool = False) -> str:
    """One iteration. Returns a short status string (useful in logs and tests)."""
    git.sync()
    cfg = load_cfg(repo)
    info = device_info()
    if info["device"] == "cpu" and not allow_cpu:
        tiers = [t for t in tiers if t == "nano"]   # a CPU-only machine may still serve the phone tier
    q, running, done_dir = repo / "jobs" / "queued", repo / "jobs" / "running", repo / "jobs" / "done"
    job, src = None, None
    for p in sorted(q.glob("*.yaml")):
        cand = yaml.safe_load(p.read_text(encoding="utf-8"))
        try:
            cand_cfg = job_cfg(cand, repo)   # a queued job may name another track (config: loop_minute)
        except (OSError, ValueError, AttributeError, yaml.YAMLError):
            continue                         # no such config on this branch: leave the job for the planner
        if runnable(cand, cand_cfg, tiers, info["vram_free_gb"] if info["device"] == "cuda" else 0.0, repo):
            job, src, cfg = cand, p, cand_cfg
            break
    track = None
    if job is None:
        track, cfg = pick_track(cfg, rng, repo)
        usable = [t for t in tiers
                  if info["device"] != "cuda" or cfg["tiers"][t]["min_vram_gb"] <= info["vram_free_gb"]]
        if not usable:
            return "idle: GPU busy (not enough free VRAM for any tier)"
        job = propose(cfg, load_results(repo), usable, rng, repo)
        job["id"] = new_id(name, rng)
        job["origin"] = "worker"
        if track:
            job["config"] = track   # run_job, champions and the leaderboard go by the track's eval.version
    job["worker"] = name
    job["attempts"] = int(job.get("attempts", 0)) + 1
    dst = running / f"{job['id']}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    if src is not None:
        src.unlink()
    quiet = src is None and float(cfg["tiers"][job["tier"]]["max_minutes"]) <= QUIET_CLAIM_MINUTES
    if not quiet and not git.publish(f"loop: {name} claims {job['id']}", ["jobs"]):
        return f"lost claim race for {job['id']}"
    res = run_job(job, cfg, repo, device)
    # Without this the allocator keeps the job's high-water mark and the next VRAM check sees a busy GPU
    # (2026-11-26: after one 6 GB job pc-5070ti reported 1.7 GB free and ran only nano jobs for hours).
    release_vram()
    res["worker"] = name
    res["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(repo / "results" / f"{job['id']}.json", res)
    done_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dst), str(done_dir / dst.name))
    paths = ["jobs", "results", "loop/workers"]
    # Champion weights are small enough to commit; everything else stays in this machine's artifacts/.
    ch = champions(load_results(repo), cfg["eval"].get("version", 1))
    if res["status"] == "ok" and ch.get(job["tier"], {}).get("id") == job["id"]:
        size = sum(f.stat().st_size for f in (repo / "artifacts" / job["id"]).glob("*"))
        if size <= 80 * 2**20:
            shutil.copytree(repo / "artifacts" / job["id"], repo / "models" / job["id"], dirs_exist_ok=True)
            paths.append("models")
    heartbeat(name, info, repo, last=f"{job['id']} {res['status']}")
    msg = f"loop: {name} {job['id']} {res['status']}"
    if res["status"] == "ok":
        msg += f" tier={job['tier']} score={res['score']:+.4f}"
    if not git.publish(msg, paths):
        return f"could not push result for {job['id']} (kept locally in artifacts/)"
    return msg


def heartbeat(name: str, info: dict, repo: Path, last: str = "") -> None:
    # alpaca_keys is a yes/no only: it tells the planner minute bars can be switched on for this machine.
    write_json(repo / "loop" / "workers" / f"{name}.json",
               {"worker": name, "seen": datetime.now(UTC).isoformat(timespec="seconds"), "last": last,
                "alpaca_keys": alpaca.available(), **info})


def recover(name: str, git: Git, repo: Path) -> None:
    """After a crash, hand this worker's half-run jobs back to the queue (twice at most)."""
    git.sync()
    moved = False
    for p in (repo / "jobs" / "running").glob("*.yaml"):
        job = yaml.safe_load(p.read_text(encoding="utf-8"))
        if job.get("worker") != name:
            continue
        target = "queued" if int(job.get("attempts", 1)) < 2 and job.get("origin") != "worker" else "done"
        shutil.move(str(p), str(repo / "jobs" / target / p.name))
        moved = True
    if moved:
        git.publish(f"loop: {name} recovers interrupted jobs", ["jobs"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tiers", default="nano,mid,large")
    ap.add_argument("--no-git", action="store_true", help="local dry run: no fetch, commit or push")
    ap.add_argument("--max-jobs", type=int, default=0, help="stop after N iterations (0 = forever)")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    git = Git(REPO, cfg["branch"], enabled=not a.no_git)
    rng = random.Random()
    recover(a.name, git, REPO)
    rev, n, fails = git.src_rev(), 0, 0
    while True:
        try:
            status = step(a.name, a.tiers.split(","), git, rng, allow_cpu=a.allow_cpu)
        except Exception as e:  # network blips, git hiccups: wait and carry on
            status = f"iteration failed: {type(e).__name__}: {e}"
            time.sleep(60)
        print(f"[{datetime.now(UTC):%H:%M:%S}] {status}", flush=True)
        n += 1
        if a.max_jobs and n >= a.max_jobs:
            return
        if status.startswith("idle"):
            time.sleep(300)
        fails = fails + 1 if status.endswith(" error") else 0
        git.sync()
        if git.src_rev() != rev:
            print("code changed upstream; restarting", flush=True)
            raise SystemExit(RESTART)
        if error_pause(fails):
            print(f"{fails} failed jobs in a row; waiting {error_pause(fails)} s", flush=True)
            time.sleep(error_pause(fails))


if __name__ == "__main__":
    main()
