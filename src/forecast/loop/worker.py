"""GPU worker daemon. Run one per machine; the git branch is the only thing workers share.

  python -m forecast.loop.worker --name pc-5070ti --tiers large,mid,nano

Loop: pull -> claim a queued job that fits this machine (or invent one when the queue is empty) ->
run it -> commit the result -> push. Files are one-per-job and one-per-worker, so pushes from two
machines never conflict; a lost claim race is detected by the rejected push and simply retried.
Exit code 3 means "code changed upstream, restart me" (the run_worker scripts do that).
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from forecast.loop.leaderboard import champions, load_results
from forecast.loop.run_job import artifact_dir, load_cfg, run_job
from forecast.utils.common import REPO, device_info, write_json

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


def new_id(worker: str, rng: random.Random) -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{worker}-{rng.randrange(16**4):04x}"


def propose(cfg: dict, results: list[dict], tiers: list[str], rng: random.Random, repo: Path = REPO) -> dict:
    """Invent the next experiment: mutate the tier champion, or distil a bigger champion into a
    smaller tier, or prune-and-finetune the champion."""
    s, ch = cfg["search"], champions(results, cfg["eval"].get("version", 1))
    done = sum(1 for r in results if r.get("status") == "ok")
    order = ["nano", "mid", "large"]
    # Make sure every tier this worker serves has at least one run before exploring.
    missing = [t for t in tiers if t not in ch]
    tier = missing[0] if missing else rng.choice(tiers)
    if missing:
        return {"tier": tier, "kind": "train", "note": "tier preset (first run)"}
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
    model, train = base, {}
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
    return {"tier": tier, "kind": "train", "model": model, "train": train, "seed": rng.randrange(10**6),
            "note": f"mutation of {ch[tier]['id']}"}


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
        if runnable(cand, cfg, tiers, info["vram_free_gb"] if info["device"] == "cuda" else 0.0, repo):
            job, src = cand, p
            break
    if job is None:
        usable = [t for t in tiers
                  if info["device"] != "cuda" or cfg["tiers"][t]["min_vram_gb"] <= info["vram_free_gb"]]
        if not usable:
            return "idle: GPU busy (not enough free VRAM for any tier)"
        job = propose(cfg, load_results(repo), usable, rng, repo)
        job["id"] = new_id(name, rng)
        job["origin"] = "worker"
    job["worker"] = name
    job["attempts"] = int(job.get("attempts", 0)) + 1
    dst = running / f"{job['id']}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    if src is not None:
        src.unlink()
    if not git.publish(f"loop: {name} claims {job['id']}", ["jobs"]):
        return f"lost claim race for {job['id']}"
    res = run_job(job, cfg, repo, device)
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
    write_json(repo / "loop" / "workers" / f"{name}.json",
               {"worker": name, "seen": datetime.now(UTC).isoformat(timespec="seconds"), "last": last, **info})


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
    rev, n = git.src_rev(), 0
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
        git.sync()
        if git.src_rev() != rev:
            print("code changed upstream; restarting", flush=True)
            raise SystemExit(RESTART)


if __name__ == "__main__":
    main()
