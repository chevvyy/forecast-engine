"""End-to-end loop on synthetic data: two workers sharing a real git remote."""
import random
import shutil
import subprocess
from pathlib import Path

import yaml

from forecast.loop.leaderboard import champions, load_results, render
from forecast.loop.worker import Git, recover, step

ROOT = Path(__file__).resolve().parents[1]


def _git(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def _clone(remote: Path, dst: Path) -> Path:
    _git(dst.parent, "clone", "-q", "-b", "loop", str(remote), dst.name)
    _git(dst, "config", "user.email", "t@example.com")
    _git(dst, "config", "user.name", "t")
    return dst


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    seed = tmp_path / "seed"
    for d in ["configs", "jobs", "results", "models", "loop", "experiments"]:
        (seed / d).mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    cfg["data"] = {"source": "synthetic", "symbols": ["SPY", "AAA", "BBB", "CCC"], "n_bars": 700, "seed": 7}
    cfg["task"]["max_lookback"] = 64
    cfg["train"].update(max_epochs=1, batch_size=256)
    for t in cfg["tiers"].values():
        t["min_vram_gb"] = 0
        t["model"].update(lookback=32, d_model=8, n_blocks=1, kernel=5, patch=4, stride=2)
    cfg["search"]["lookback"] = {k: [32, 64] for k in ("nano", "mid", "large")}
    cfg["search"]["d_model"] = {k: [8, 12] for k in ("nano", "mid", "large")}
    cfg["search"]["tracks"] = {}   # self-invented jobs stay daily unless a test hands out a share
    (seed / "configs" / "loop.yaml").write_text(yaml.safe_dump(cfg))
    for d in ["queued", "running", "done"]:
        (seed / "jobs" / d).mkdir()
        (seed / "jobs" / d / ".gitkeep").touch()
    (seed / "results" / ".gitkeep").touch()
    (seed / ".gitignore").write_text("artifacts/\ndata/\n")
    (seed / "jobs" / "queued" / "q1.yaml").write_text(yaml.safe_dump({"id": "q1", "tier": "nano", "kind": "train"}))
    _git(seed, "init", "-q", "-b", "loop")
    _git(seed, "config", "user.email", "t@example.com")
    _git(seed, "config", "user.name", "t")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "seed")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), remote.name)
    return _clone(remote, tmp_path / "a"), _clone(remote, tmp_path / "b")


def test_two_workers_claim_run_and_publish(tmp_path):
    a, b = _setup(tmp_path)
    ga, gb = Git(a, "loop"), Git(b, "loop")
    rng = random.Random(0)
    s1 = step("wa", ["nano", "mid"], ga, rng, repo=a, device="cpu", allow_cpu=True)
    assert "q1 ok" in s1, s1
    # Worker b must see q1 as done and invent its own job rather than rerun it.
    s2 = step("wb", ["nano", "mid"], gb, rng, repo=b, device="cpu", allow_cpu=True)
    assert " ok" in s2 and "q1" not in s2, s2
    for _ in range(4):
        assert " ok" in step("wa", ["nano", "mid"], ga, rng, repo=a, device="cpu", allow_cpu=True)
    gb.sync()
    res = load_results(b)
    assert len(res) == 6 and all(r["status"] == "ok" for r in res)
    assert not list((b / "jobs" / "running").glob("*.yaml"))
    ch = champions(res)
    assert set(ch) == {"nano", "mid"}
    # Champion weights travel through git, so the other machine can distil from / prune them.
    for r in ch.values():
        assert (b / "models" / r["id"] / "fold0.pt").exists() and (b / "models" / r["id"] / "model.onnx").exists()
    assert {r["kind"] for r in res} & {"distill", "prune", "train"}
    assert "Champions" in render(res)
    # After an eval.version bump the old runs are neither champions nor counted as failures.
    nxt = render(res, 99)
    assert "Evaluation version 99: 0 finished runs" in nxt and " 0 failed" in nxt and "(none yet)" in nxt
    assert f"{len(res)} finished under other versions" in nxt and not champions(res, 99)
    assert (b / "loop" / "workers" / "wa.json").exists()


def test_distill_and_prune_jobs_run_from_committed_champions(tmp_path):
    a, b = _setup(tmp_path)
    ga, gb = Git(a, "loop"), Git(b, "loop")
    rng = random.Random(1)
    step("wa", ["mid"], ga, rng, repo=a, device="cpu", allow_cpu=True)          # q1 is nano -> not runnable
    gb.sync()
    mid = champions(load_results(b))["mid"]["id"]
    q = b / "jobs" / "queued"
    (q / "d1.yaml").write_text(yaml.safe_dump({"id": "d1", "tier": "nano", "kind": "distill", "teacher": mid}))
    (q / "p1.yaml").write_text(yaml.safe_dump({"id": "p1", "tier": "mid", "kind": "prune", "init_from": mid,
                                               "sparsity": 0.5}))
    assert gb.publish("queue", ["jobs"])
    for _ in range(3):
        step("wb", ["nano", "mid"], gb, rng, repo=b, device="cpu", allow_cpu=True)
    res = {r["id"]: r for r in load_results(b)}
    assert res["d1"]["status"] == "ok" and res["p1"]["status"] == "ok", (res["d1"].get("error"), res["p1"].get("error"))
    assert res["p1"]["nonzero_params"] < 0.75 * res["p1"]["params"]


def test_failed_job_is_recorded_and_crashed_claims_are_recovered(tmp_path):
    a, _ = _setup(tmp_path)
    ga = Git(a, "loop")
    (a / "jobs" / "queued" / "q1.yaml").write_text(
        yaml.safe_dump({"id": "q1", "tier": "nano", "model": {"lookback": 9999}}))
    assert ga.publish("bad job", ["jobs"])
    s = step("wa", ["nano"], ga, random.Random(2), repo=a, device="cpu", allow_cpu=True)
    assert "q1 error" in s
    assert "max_lookback" in load_results(a)[0]["error"]
    shutil.copy(a / "jobs" / "done" / "q1.yaml", a / "jobs" / "running" / "q9.yaml")
    assert ga.publish("fake crash", ["jobs"])
    recover("wa", ga, a)
    assert not list((a / "jobs" / "running").glob("*.yaml"))


def test_time_zone_database_is_a_declared_dependency():
    # Windows has no system zone database. Without the tzdata package, polars cannot hand back a UTC
    # datetime and every job on pc-5070ti failed in run_job.prepare (981 runs, 2026-11-25/26).
    import tomllib

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(d.split(">")[0].split("=")[0].strip() == "tzdata" for d in deps)


def test_utc_datetimes_work_without_a_system_zone_database():
    # Reproduces the Windows failure on any OS: hide the system zone files, then do what prepare() does.
    import sys

    import pytest

    pytest.importorskip("tzdata")
    code = ("import zoneinfo, polars as pl; from datetime import UTC, datetime; zoneinfo.reset_tzpath(to=[]); "
            "s = pl.Series([datetime(2024, 1, 2, tzinfo=UTC)]).cast(pl.Datetime('us', 'UTC')); "
            "assert s.max() == datetime(2024, 1, 2, tzinfo=UTC); "
            "s.dt.convert_time_zone('America/New_York').max()")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


def test_error_pause_backs_off_only_on_repeated_failures():
    from forecast.loop.worker import error_pause

    assert error_pause(0) == 0 and error_pause(1) == 0
    assert [error_pause(k) for k in (2, 3, 4, 5)] == [60, 120, 240, 480]
    assert error_pause(50) == 900


def test_own_cached_vram_is_not_counted_as_busy(monkeypatch):
    """pc-5070ti, 2026-11-26: after its own 6 GB job the driver reported 1.66 GB free because PyTorch
    kept the memory cached, so the worker skipped every mid and large job from then on."""
    import torch

    from forecast.utils import common

    gb = 2**30
    assert common.usable_vram_gb(int(1.66 * gb), 13 * gb, 0) > 14
    assert common.usable_vram_gb(4 * gb, 6 * gb, 6 * gb) == 4       # memory in live tensors stays used
    assert common.usable_vram_gb(4 * gb, 1 * gb, 2 * gb) == 4       # never subtracts
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (int(1.66 * gb), int(15.92 * gb)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 13 * gb)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: gb // 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "fake")
    info = common.device_info()
    assert info["device"] == "cuda" and 14 < info["vram_free_gb"] <= info["vram_total_gb"]
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 40 * gb)   # nonsense input: capped at total
    assert common.device_info()["vram_free_gb"] == info["vram_total_gb"]


def test_worker_releases_vram_after_every_job(tmp_path, monkeypatch):
    from forecast.loop import worker

    a, _ = _setup(tmp_path)
    calls = []
    monkeypatch.setattr(worker, "release_vram", lambda: calls.append(1))
    rng = random.Random(0)
    assert "q1 ok" in step("wa", ["nano"], Git(a, "loop"), rng, repo=a, device="cpu", allow_cpu=True)
    (a / "jobs" / "queued" / "bad.yaml").write_text(
        yaml.safe_dump({"id": "bad", "tier": "nano", "model": {"lookback": 9999}}))
    _git(a, "add", "-A")
    _git(a, "commit", "-q", "-m", "bad job")
    _git(a, "push", "-q", "origin", "HEAD:loop")
    assert "bad error" in step("wa", ["nano"], Git(a, "loop"), rng, repo=a, device="cpu", allow_cpu=True)
    assert len(calls) == 2   # after a finished job and after a failed one
    from forecast.utils.common import release_vram
    release_vram()           # harmless without a GPU


def _fake_result(rid: str, tier: str, cfg: dict, score: float = 0.01, **job) -> dict:
    job = {"kind": "train", **job, "id": rid, "tier": tier}
    return {"id": rid, "status": "ok", "fits_tier": True, "tier": tier, "kind": job["kind"], "score": score,
            "eval_version": cfg["eval"]["version"], "job": job,
            "model": {**cfg["tiers"][tier]["model"], **(job.get("model") or {})}}


def test_proposals_carry_a_seed_and_are_not_repeated(tmp_path):
    """2026-11-25: worker-invented distil and prune jobs had no seed and were never compared with
    earlier runs, so 142 distil runs were 11 distinct jobs with identical scores."""
    from forecast.loop.worker import MAX_REPEATS, propose, signature

    cfg = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    res = [_fake_result("n0", "nano", cfg, seed=1), _fake_result("m0", "mid", cfg, seed=1)]
    (tmp_path / "models" / "m0").mkdir(parents=True)
    (tmp_path / "models" / "m0" / "fold0.pt").touch()
    cfg["search"].update(p_distill=1.0, p_prune=0.0, confirm_seeds=1)   # champion re-runs have their own test
    seen = []
    for k in range(MAX_REPEATS):
        job = propose(cfg, res, ["nano"], random.Random(k), tmp_path)
        assert job["kind"] == "distill" and job["teacher"] == "m0" and isinstance(job["seed"], int)
        seen.append(job["seed"])
        res.append(_fake_result(f"d{k}", "nano", cfg, 0.0, **{k_: v for k_, v in job.items() if k_ != "tier"}))
    assert len(set(seen)) == MAX_REPEATS          # same configuration, different seeds
    # The configuration has had its seeds: the worker must now do something else.
    for k in range(20):
        job = propose(cfg, res, ["nano"], random.Random(100 + k), tmp_path)
        assert job["kind"] == "train" and "seed" in job, job
    # Prune proposals are seeded too.
    (tmp_path / "models" / "n0").mkdir()
    (tmp_path / "models" / "n0" / "fold0.pt").touch()
    cfg["search"].update(p_distill=0.0, p_prune=1.0)
    job = propose(cfg, res, ["nano"], random.Random(5), tmp_path)
    assert job["kind"] == "prune" and isinstance(job["seed"], int)
    # An exact repeat (same job, same seed) is redrawn; an error result counts as "already tried" too.
    cfg["search"].update(p_distill=0.0, p_prune=0.0)
    first = propose(cfg, res, ["nano"], random.Random(7), tmp_path)
    tried = {**_fake_result("x", "nano", cfg, **{k: v for k, v in first.items() if k != "tier"}), "status": "error"}
    again = propose(cfg, res + [tried], ["nano"], random.Random(7), tmp_path)
    assert signature(again, cfg) != signature(first, cfg)
    # Spelling out a preset value does not make a job new; runs of another eval.version do not count.
    explicit = {**first, "model": {**cfg["tiers"]["nano"]["model"], **first["model"]}}
    assert signature(explicit, cfg) == signature(first, cfg)
    old = {**tried, "eval_version": cfg["eval"]["version"] - 1}
    assert signature(propose(cfg, res + [old], ["nano"], random.Random(7), tmp_path), cfg) == signature(first, cfg)
    # Results are worker-written data: a malformed one must not stop the worker proposing.
    assert propose(cfg, res + [{"status": "error", "eval_version": cfg["eval"]["version"], "job": "junk"},
                               {"status": "error", "eval_version": cfg["eval"]["version"]}],
                   ["nano"], random.Random(7), tmp_path)["tier"] == "nano"


def test_mutations_and_seed_reruns_keep_how_the_champion_was_trained(tmp_path):
    """2026-11-27: a mutation started from the champion's shape but from the default training settings, so
    with champions made by batch 2048 / lr 0.0003 (mid, large) or by a teacher (nano) nearly every
    self-invented run was a known-worse recipe: the search could not climb."""
    from forecast.loop.worker import propose, recipe

    cfg = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    cfg["search"].update(p_distill=0.0, p_prune=0.0, tracks={}, confirm_seeds=1)
    ver = cfg["eval"]["version"]

    def result(rid, tier, score, train=None, **job):
        r = _fake_result(rid, tier, cfg, score, **job)
        return {**r, "train": {**cfg["train"], **(job.get("train") or {}), **(train or {})}}

    mid = result("m0", "mid", 0.04, train={"batch_size": 2048, "lr": 0.0003})
    nano = result("n0", "nano", 0.05, kind="distill", teacher="m0")
    res = [mid, nano, result("n1", "nano", 0.03)]
    (tmp_path / "models" / "m0").mkdir(parents=True)
    (tmp_path / "models" / "m0" / "fold0.pt").touch()
    assert recipe(mid, res, cfg, tmp_path) == {"train": {"batch_size": 2048, "lr": 0.0003}}
    assert recipe(nano, res, cfg, tmp_path) == {"train": {}, "teacher": "m0"}
    for k in range(60):
        job = propose(cfg, res, ["mid"], random.Random(k), tmp_path)
        assert job["kind"] == "train" and "teacher" not in job
        # Unless the mutation drew that very key, the champion's value is kept; a drawn key is from the grid.
        assert job["train"]["batch_size"] in cfg["search"]["batch_size"] and job["train"]["lr"] in cfg["search"]["lr"]
        job = propose(cfg, res, ["nano"], random.Random(k), tmp_path)
        assert job["kind"] == "distill" and job["teacher"] == "m0" and isinstance(job["seed"], int)
    kept = [propose(cfg, res, ["mid"], random.Random(k), tmp_path)["train"] for k in range(200)]
    assert sum(t["batch_size"] == 2048 for t in kept) > 150 and sum(t["lr"] == 0.0003 for t in kept) > 150
    # The teacher's checkpoints are not on this machine: train without it instead of failing every job.
    assert "teacher" not in propose(cfg, res, ["nano"], random.Random(1), tmp_path / "elsewhere")
    # A pruned champion was finetuned (lr 0.0003, 15 epochs); its recipe is that of the run it came from.
    pruned = result("m1", "mid", 0.05, kind="prune", init_from="m0", sparsity=0.5, train={"lr": 0.001, "max_epochs": 15})
    assert recipe(pruned, res + [pruned], cfg, tmp_path) == {"train": {"batch_size": 2048, "lr": 0.0003}}
    # Early stopping inside the epoch is part of how a champion was trained, although nothing mutates it.
    fine = result("m3", "mid", 0.09, train={"lr": 0.0003, "evals_per_epoch": 4})
    assert recipe(fine, res + [fine], cfg, tmp_path) == {"train": {"evals_per_epoch": 4, "lr": 0.0003}}
    assert propose(cfg, res + [fine], ["mid"], random.Random(3), tmp_path)["train"]["evals_per_epoch"] == 4
    for junk in (1, 400, True, "4", 4.5):
        odd = result("m4", "mid", 0.09, train={"evals_per_epoch": junk})
        assert recipe(odd, res + [odd], cfg, tmp_path) == {"train": {}}, junk
    orphan = result("m2", "mid", 0.06, kind="prune", init_from="gone", sparsity=0.5, train={"batch_size": 2048})
    assert recipe(orphan, res + [orphan], cfg, tmp_path) == {"train": {}}
    loop = [result("a", "mid", 0.07, init_from="b"), result("b", "mid", 0.0, init_from="a")]
    assert recipe(loop[0], loop, cfg, tmp_path) == {"train": {}}
    # Results are worker-written data: values outside the grid, other types, a teacher that is not a
    # finished run of this version, or a junk job are not taken over.
    for bad in ({"train": {"lr": 5.0, "batch_size": True, "weight_decay": "0.01"}},
                {"train": "junk"}, {"job": "junk"}, {"job": {"teacher": "../../etc"}},
                {"job": {"teacher": ["m0"]}}, {"job": {"teacher": "old"}}):
        old = {**mid, "id": "old", "eval_version": ver - 1}
        assert recipe({**result("x", "mid", 0.9), **bad}, res + [old], cfg, tmp_path) == {"train": {}}
    # When no draw is usable (here: none fits the VRAM limit), the fallback is a real seed re-run: the
    # champion's shape AND its recipe. It used to be the shape at the default settings.
    cfg["tiers"]["mid"]["max_train_vram_gb"] = 0.01
    job = propose(cfg, res, ["mid"], random.Random(0), tmp_path)
    assert job["note"] == "seed re-run of m0" and job["model"] == mid["model"]
    assert job["train"] == {"batch_size": 2048, "lr": 0.0003} and isinstance(job["seed"], int)
    cfg["tiers"]["nano"]["max_train_vram_gb"] = 0.01
    job = propose(cfg, res, ["nano"], random.Random(0), tmp_path)
    assert job["note"] == "seed re-run of n0" and job["kind"] == "distill" and job["teacher"] == "m0"


def test_a_champion_is_rerun_with_new_seeds_before_it_is_mutated(tmp_path):
    """2026-11-27: every tier's champion was one seed for hours (mid c1f5, nano 5167, large e6) while the
    worker mutated it; whether it was a step or a lucky draw waited for the planner's next cycle."""
    from forecast.loop.worker import CONFIRM_SEEDS, propose, rerun, signature

    cfg = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    cfg["search"].update(p_distill=0.0, p_prune=0.0, tracks={})
    assert cfg["search"]["confirm_seeds"] == CONFIRM_SEEDS == 3
    for name in ("m0", "n0"):
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "fold0.pt").touch()
    mid = _fake_result("m0", "mid", cfg, 0.04, seed=11, train={"batch_size": 2048, "lr": 0.0003},
                       model={"kernel": 9})
    nano = _fake_result("n0", "nano", cfg, 0.05, kind="distill", teacher="m0", seed=11, max_minutes=5)
    res = [mid, nano, _fake_result("n1", "nano", cfg, 0.03, seed=11)]
    # The re-run is the champion's own job: shape, settings and teacher, nothing else from the result.
    assert rerun(mid, res, cfg, tmp_path) == {"tier": "mid", "kind": "train", "model": {"kernel": 9},
                                               "train": {"batch_size": 2048, "lr": 0.0003}}
    assert rerun(nano, res, cfg, tmp_path) == {"tier": "nano", "kind": "distill", "teacher": "m0"}
    seen = {11}
    for k in range(CONFIRM_SEEDS - 1):
        job = propose(cfg, res, ["nano"], random.Random(k), tmp_path)
        assert job["note"] == f"confirm champion n0: seed {k + 2} of 3" and job["teacher"] == "m0"
        assert signature(job, cfg, seed=False) == signature(nano["job"], cfg, seed=False)
        assert job["seed"] not in seen
        seen.add(job["seed"])
        # A lower score: n0 stays the champion. The second one fails: a failed run is a seed tried.
        r = _fake_result(f"r{k}", "nano", cfg, 0.045, **{k_: v for k_, v in job.items() if k_ != "tier"})
        res.append(r if k == 0 else {**r, "status": "error"})
    # Three seeds tried: the worker goes back to mutating the champion (with its recipe).
    for k in range(20):
        job = propose(cfg, res, ["nano"], random.Random(k), tmp_path)
        assert job["note"] == "mutation of n0" and job["teacher"] == "m0", job
    # A worker serving several tiers confirms whichever champion still needs it (mid here) first.
    for k in range(20):
        job = propose(cfg, res, ["nano", "mid"], random.Random(k), tmp_path)
        assert job["tier"] == "mid" and job["note"].startswith("confirm champion m0: seed 2")
        assert job["train"] == {"batch_size": 2048, "lr": 0.0003} and job["model"] == {"kernel": 9}
    # A re-run that scores higher becomes the champion; the configuration, and so the count, is the same.
    better = _fake_result("m1", "mid", cfg, 0.06, **{k_: v for k_, v in job.items() if k_ != "tier"})
    job = propose(cfg, res + [better], ["mid"], random.Random(0), tmp_path)
    assert job["note"] == "confirm champion m1: seed 3 of 3" and job["seed"] not in (11, better["job"]["seed"])
    # A pruned champion is re-run as the same prune of the same parent.
    pruned = _fake_result("p0", "mid", cfg, 0.07, kind="prune", init_from="m0", sparsity=0.5, seed=5,
                          train={"lr": 0.0003, "max_epochs": 15})
    job = propose(cfg, res + [pruned], ["mid"], random.Random(0), tmp_path)
    assert (job["kind"], job["init_from"], job["sparsity"]) == ("prune", "m0", 0.5) and job["seed"] != 5
    assert job["train"] == {"lr": 0.0003, "max_epochs": 15}
    # A champion whose early stopping looked inside the epoch (``train.evals_per_epoch``, a key the
    # config has no default for) is re-run the same way, and its configuration is not the plain one.
    fine = _fake_result("e0", "mid", cfg, 0.08, seed=5, train={"lr": 0.0003, "evals_per_epoch": 4})
    job = propose(cfg, res + [fine], ["mid"], random.Random(0), tmp_path)
    assert job["note"].startswith("confirm champion e0: seed 2") and job["train"] == fine["job"]["train"]
    plain = {"tier": "mid", "train": {"lr": 0.0003}}
    assert signature(job, cfg, seed=False) != signature(plain, cfg, seed=False)
    # Not re-run here: the teacher's checkpoints are on another machine, the shape no longer fits the
    # tier's VRAM, or the switch is off. The worker then proposes as before.
    assert propose(cfg, res, ["nano"], random.Random(0), tmp_path / "elsewhere")["note"] == "mutation of n0"
    tight = yaml.safe_load(yaml.safe_dump(cfg))
    tight["tiers"]["mid"]["max_train_vram_gb"] = 0.01
    assert propose(tight, res, ["mid"], random.Random(0), tmp_path)["note"] == "seed re-run of m0"
    off = yaml.safe_load(yaml.safe_dump(cfg))
    off["search"]["confirm_seeds"] = 1
    assert propose(off, res, ["mid"], random.Random(0), tmp_path)["note"] == "mutation of m0"
    # Results are worker-written data: a job with fields of the wrong type, keys the presets do not
    # have, a path for a teacher, a parent from another eval.version or an impossible sparsity is not
    # copied into a new job, and proposing carries on.
    old = {**mid, "id": "old", "eval_version": cfg["eval"]["version"] - 1}
    for bad in ("junk", None, {"model": "junk"}, {"model": {"kernel": "9; rm"}}, {"model": {"device": 1}},
                {"train": {"lr": float("nan")}}, {"train": {"batch_size": True}}, {"train": {"shell": 1}},
                {"train": {"evals_per_epoch": "4"}},
                {"teacher": "../../etc"}, {"teacher": ["m0"]}, {"init_from": "old", "sparsity": 0.5},
                {"sparsity": 1.5}, {"sparsity": "0.5"}):
        champ = {**_fake_result("x", "mid", cfg, 0.9), "job": bad}
        assert rerun(champ, res + [old, champ], cfg, tmp_path) is None, bad
        assert propose(cfg, res + [old, champ], ["mid"], random.Random(0), tmp_path)["tier"] == "mid"
    assert rerun({**mid, "tier": "huge"}, res, cfg, tmp_path) is None


def test_mutations_start_from_the_tier_holder_not_from_a_champion_inside_the_noise(tmp_path):
    """Seed-noise backlog line, part 5 (2026-11-28). A configuration that tops the single-run board is
    re-run to three seeds; it becomes what the worker mutates only if its mean then beats the holder's
    by more than the noise (``leaderboard.promoted``)."""
    from forecast.loop.worker import propose

    cfg = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    cfg["search"].update(p_distill=0.0, p_prune=0.0, tracks={})

    def seeds(name, scores, hour, **job):
        return [{**_fake_result(f"{name}{k}", "nano", cfg, s, seed=k + 1, **job),
                 "finished": f"2026-11-28T{hour:02d}:0{k}:00+00:00"} for k, s in enumerate(scores)]

    old = seeds("a", [0.030, 0.032, 0.034], 1)
    near = seeds("b", [0.036, 0.033, 0.0335], 2, model={"kernel": 15})     # champion run b0; mean +0.002
    res = old + near
    assert champions(res, cfg["eval"]["version"])["nano"]["id"] == "b0"
    for k in range(20):
        job = propose(cfg, res, ["nano"], random.Random(k), tmp_path)
        assert job["note"] == "mutation of a2", job                         # the best run of a
    far = seeds("c", [0.045, 0.046, 0.047], 3, model={"kernel": 25})
    assert propose(cfg, res + far, ["nano"], random.Random(0), tmp_path)["note"] == "mutation of c2"
    # A new champion with one seed is confirmed first, as before; and with the re-runs switched off
    # there is no seed noise to gate on, so the single-run champion is mutated as before 2026-11-27.
    top = _fake_result("d0", "nano", cfg, 0.09, seed=1, model={"kernel": 5})
    assert propose(cfg, res + [top], ["nano"], random.Random(0), tmp_path)["note"].startswith("confirm champion d0")
    off = yaml.safe_load(yaml.safe_dump(cfg))
    off["search"]["confirm_seeds"] = 1
    assert propose(off, res, ["nano"], random.Random(0), tmp_path)["note"] == "mutation of b0"
    # Worker-written data: a confirmed "configuration" whose result has no model is not built upon.
    # (j is confirmed first and nothing later clears its margin, so it holds the tier.)
    junk = [{**r, "model": 3, "job": {**r["job"], "model": {"kernel": 51}}}
            for r in seeds("j", [0.033, 0.033, 0.0335], 0)]
    assert propose(cfg, res + junk, ["nano"], random.Random(0), tmp_path)["note"] == "mutation of b0"


def test_proposed_shapes_fit_the_tier_vram():
    """A 255-token, d_model 512, 8-block mutation cannot train in 16 GB at batch 512; proposing it
    costs a failed or spilled run every time (2026-11-26)."""
    from forecast.loop.worker import fits_vram, propose
    from forecast.models.patchnet import train_vram_gb

    # Measured training peaks on pc-5070ti (batch, lookback, patch, stride, d_model, n_blocks, ffn) -> GB.
    measured = [((512, 256, 2, 1, 384, 8, 4), 16.6), ((512, 256, 2, 1, 256, 8, 4), 11.1),
                ((512, 256, 8, 4, 768, 8, 4), 8.7), ((512, 256, 8, 4, 512, 8, 4), 6.0),
                ((512, 256, 8, 4, 384, 12, 4), 5.5), ((512, 256, 8, 4, 384, 8, 2), 2.7),
                ((512, 256, 8, 4, 384, 8, 1), 1.9)]
    for (bs, lb, pt, st, d, nb, ffn), gb in measured:
        est = train_vram_gb({"lookback": lb, "patch": pt, "stride": st, "d_model": d, "n_blocks": nb, "ffn": ffn}, bs)
        assert abs(est - gb) < 1.25, (est, gb)
    cfg = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    big = {"tier": "large", "model": {"lookback": 256, "patch": 2, "stride": 1, "d_model": 512, "n_blocks": 8}}
    assert not fits_vram(big, cfg)
    assert fits_vram({"tier": "large"}, cfg) and fits_vram({"tier": "mid"}, cfg) and fits_vram({"tier": "nano"}, cfg)
    assert not fits_vram({"tier": "large", "train": {"batch_size": 2048}}, cfg)   # 15.9 GB and spilled, measured
    # Mutations of a champion that sits next to the limit never leave it.
    res = [_fake_result("l0", "large", cfg, model=big["model"] | {"d_model": 256}),
           _fake_result("m0", "mid", cfg, model={"lookback": 256, "patch": 2, "stride": 1, "d_model": 192})]
    rng = random.Random(3)
    jobs = [propose(cfg, res, ["mid", "large"], rng, ROOT / "no-such-dir") for _ in range(400)]
    assert all(fits_vram(j, cfg) for j in jobs)
    assert len({j["tier"] for j in jobs}) == 2 and all("seed" in j for j in jobs)


def test_short_self_invented_jobs_make_one_commit_not_two(tmp_path):
    """Nano jobs took 15 s and each wrote a claim commit and a result commit: 3000+ commits in four
    hours (2026-11-25). A claim only matters for a queued job, which another worker could take."""
    a, _ = _setup(tmp_path)
    ga = Git(a, "loop")
    rng = random.Random(4)

    def subjects() -> list[str]:
        return subprocess.run(["git", "log", "--format=%s", "origin/loop"], cwd=a, capture_output=True,
                              text=True, check=True).stdout.splitlines()

    assert "q1 ok" in step("wa", ["nano"], ga, rng, repo=a, device="cpu", allow_cpu=True)
    assert sum("claims q1" in s for s in subjects()) == 1            # queued job: still claimed first
    n = len(subjects())
    s = step("wa", ["nano"], ga, rng, repo=a, device="cpu", allow_cpu=True)   # invented, budget 15 min
    assert " ok" in s and len(subjects()) == n + 1 and "claims" not in subjects()[0]
    assert not list((a / "jobs" / "running").glob("*.yaml"))
    assert len(list((a / "jobs" / "done").glob("*.yaml"))) == 2
    n = len(subjects())
    s = step("wa", ["mid"], ga, rng, repo=a, device="cpu", allow_cpu=True)    # invented, budget 30 min
    assert " ok" in s and len(subjects()) == n + 2                   # long jobs stay visible while running
    assert not subprocess.run(["git", "status", "--porcelain"], cwd=a, capture_output=True, text=True).stdout


def test_non_finite_score_is_a_failed_run_and_never_a_champion(tmp_path, monkeypatch):
    """Nothing compares greater than NaN, so a diverged first run would become a tier's champion
    and never be replaced."""
    import numpy as np

    from forecast.loop import run_job as rj

    a, _ = _setup(tmp_path)
    cfg = rj.load_cfg(a)
    real = rj.predict
    monkeypatch.setattr(rj, "predict", lambda *x, **k: np.full_like(real(*x, **k), np.nan))
    res = rj.run_job({"id": "nan1", "tier": "nano"}, cfg, a, "cpu")
    assert res["status"] == "error" and "non-finite" in res["error"] and "score" not in res
    good = _fake_result("g", "nano", cfg, score=-0.5)
    bad = _fake_result("b", "nano", cfg, score=float("nan"))
    assert champions([bad, good])["nano"]["id"] == "g" and champions([good, bad])["nano"]["id"] == "g"
    assert not champions([bad])


def test_prune_job_uses_its_parents_shape_and_lookback(tmp_path, monkeypatch):
    """2026-11-26: prune jobs built their windows at the tier preset's lookback (64 for nano) although the
    parent they finetune had lookback 256, so the pruned model never saw the bars it was trained on, and
    the result recorded the preset shape instead of the parent's."""
    from forecast.loop import run_job as rj

    a, _ = _setup(tmp_path)
    cfg = rj.load_cfg(a)
    parent = rj.run_job({"id": "t1", "tier": "nano", "model": {"lookback": 64, "d_model": 12}}, cfg, a, "cpu")
    assert parent["status"] == "ok", parent.get("error")
    seen = []
    real = rj.build
    monkeypatch.setattr(rj, "build", lambda bars, horizons, lookback, *a_, **k: (
        seen.append(lookback), real(bars, horizons, lookback, *a_, **k))[1])
    res = rj.run_job({"id": "p1", "tier": "nano", "kind": "prune", "init_from": "t1", "sparsity": 0.5},
                     cfg, a, "cpu")
    assert res["status"] == "ok", res.get("error")
    assert seen == [64]                                   # the preset says 32
    assert res["model"]["lookback"] == 64 and res["model"]["d_model"] == 12
    assert res["params"] == parent["params"]


def _minute_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "loop_minute.yaml").read_text())
    cfg["data"] = {"source": "synthetic", "interval": "1m", "symbols": ["SPY", "AAA", "BBB"],
                   "n_bars": 390 * 14, "seed": 3}
    cfg["train"].update(max_epochs=1, batch_size=256)
    for t in cfg["tiers"].values():
        t["min_vram_gb"] = 0
        t["model"].update(d_model=8, n_blocks=1, kernel=5)
    return cfg


def test_minute_track_runs_from_its_own_config_and_never_competes_with_daily(tmp_path):
    from forecast.loop.run_job import load_cfg

    a, _ = _setup(tmp_path)
    ga = Git(a, "loop")
    (a / "configs" / "loop_minute.yaml").write_text(yaml.safe_dump(_minute_cfg()))
    q = a / "jobs" / "queued"
    (q / "min1.yaml").write_text(yaml.safe_dump({"id": "min1", "tier": "nano", "config": "loop_minute"}))
    (q / "bad1.yaml").write_text(yaml.safe_dump({"id": "bad1", "tier": "nano", "config": "../loop"}))
    (q / "bad2.yaml").write_text(yaml.safe_dump({"id": "bad2", "tier": "nano", "config": "no_such_track"}))
    assert ga.publish("minute track", ["configs", "jobs"])
    rng = random.Random(2)
    for _ in range(3):   # min1 (minute), q1 (daily), then a self-invented job: daily, no share is handed out
        step("wa", ["nano"], ga, rng, repo=a, device="cpu", allow_cpu=True)
    res = load_results(a)
    by = {r["id"]: r for r in res}
    assert by["min1"]["status"] == "ok", by["min1"].get("error")
    mver = _minute_cfg()["eval"]["version"]
    assert by["min1"]["eval_version"] == mver and by["min1"]["config"] == "loop_minute"
    # The minute track has a clock: time-of-day inputs, and one more volatility baseline to beat.
    pb = by["min1"]["pinball"]
    assert by["min1"]["tab"][-2:] == ["tod", "tod_u"] and by["q1"]["tab"] == ["log_rv5_rv20", "log_rv60_rv20"]
    assert pb["vol_baseline"] == min(pb["vol_unconditional"], pb["vol_har"], pb["vol_har_tod"])
    assert by["min1"]["vol_baseline"] in ("unconditional", "har", "har_tod")
    assert "vol_har_tod" not in by["q1"]["pinball"] and by["q1"]["vol_baseline"] in ("unconditional", "har")
    assert by["min1"]["horizons"] == [5, 15, 30, 60]
    daily_ver = load_cfg(a)["eval"]["version"]
    assert by["q1"]["eval_version"] == daily_ver and by["q1"]["config"] == "loop"
    assert all(r["eval_version"] == daily_ver for r in res if r["id"] != "min1")
    assert champions(res, mver)["nano"]["id"] == "min1"
    assert champions(res, daily_ver)["nano"]["id"] != "min1"
    assert "min1" not in render(res, daily_ver) and "min1" in render(res, mver)
    # Jobs naming a config that is not a plain file in configs/ are left alone, and the worker carries on.
    assert sorted(p.stem for p in q.glob("*.yaml")) == ["bad1", "bad2"]


def test_worker_gives_a_share_of_its_own_jobs_to_the_minute_track_only_if_it_can_load_it(tmp_path, monkeypatch):
    """Minute bars part (c): a worker with Alpaca keys invents minute jobs too, from the minute track's
    own champions and grid; a worker without keys, or a branch without the track, stays daily."""
    from forecast.data import alpaca
    from forecast.loop.run_job import load_cfg
    from forecast.loop.worker import pick_track

    a, _ = _setup(tmp_path)
    ga = Git(a, "loop")
    daily = load_cfg(a)
    daily["search"]["tracks"] = {"loop_minute": 1.0}
    rng = random.Random(1)
    assert pick_track(daily, rng, a) == (None, daily)                      # no such config on the branch
    assert pick_track({**daily, "search": {**daily["search"], "tracks": {"../loop": 1.0}}}, rng, a)[0] is None
    minute = _minute_cfg()
    minute["search"]["d_model"] = {k: [8, 12] for k in ("nano", "mid", "large")}
    (a / "configs" / "loop_minute.yaml").write_text(yaml.safe_dump({**minute, "data": {"source": "alpaca"}}))
    monkeypatch.setattr(alpaca, "available", lambda: False)
    assert pick_track(daily, rng, a)[0] is None                            # the track needs keys: none here
    monkeypatch.setattr(alpaca, "available", lambda: True)
    assert pick_track(daily, rng, a)[0] == "loop_minute"
    assert pick_track({**daily, "search": {**daily["search"], "tracks": {"loop_minute": 0.0}}}, rng, a)[0] is None
    same = {**minute, "eval": {**minute["eval"], "version": daily["eval"]["version"]}, "data": {"source": "alpaca"}}
    (a / "configs" / "loop_minute.yaml").write_text(yaml.safe_dump(same))
    assert pick_track(daily, rng, a)[0] is None                            # a shared version would compete
    (a / "configs" / "loop_minute.yaml").write_text(yaml.safe_dump({**minute, "search": None}))
    assert pick_track(daily, rng, a)[0] is None                            # no grid to draw from
    # End to end on synthetic sessions: every self-invented job goes to the minute track.
    (a / "configs" / "loop_minute.yaml").write_text(yaml.safe_dump(minute))
    (a / "configs" / "loop.yaml").write_text(yaml.safe_dump(daily))
    assert ga.publish("hand the minute track a share", ["configs"])
    assert "q1 ok" in step("wa", ["nano"], ga, rng, repo=a, device="cpu", allow_cpu=True)   # queue first
    for _ in range(4):
        assert " ok" in step("wa", ["nano", "mid"], ga, rng, repo=a, device="cpu", allow_cpu=True)
    res = {r["id"]: r for r in load_results(a)}
    own = [r for r in res.values() if r["id"] != "q1"]
    mver, dver = minute["eval"]["version"], daily["eval"]["version"]
    assert res["q1"]["eval_version"] == dver and res["q1"]["config"] == "loop"
    assert len(own) == 4 and all(r["status"] == "ok" for r in own), [r.get("error") for r in own]
    assert all(r["eval_version"] == mver and r["config"] == "loop_minute" and r["job"]["config"] == "loop_minute"
               and r["job"]["origin"] == "worker" and r["tab"][-2:] == ["tod", "tod_u"] for r in own)
    assert {r["tier"] for r in own} == {"nano", "mid"}                     # each tier's preset first
    assert all(r["model"]["lookback"] <= minute["task"]["max_lookback"] for r in own)
    # The tracks still never compete: the daily champion is the daily run, whatever the minute scores are.
    assert champions(list(res.values()), dver)["nano"]["id"] == "q1"
    assert champions(list(res.values()), mver)["nano"]["id"] != "q1"
    assert not list((a / "jobs" / "running").glob("*.yaml"))


def test_minute_config_is_a_separate_track_that_fits_inside_a_session():
    daily = yaml.safe_load((ROOT / "configs" / "loop.yaml").read_text())
    minute = yaml.safe_load((ROOT / "configs" / "loop_minute.yaml").read_text())
    versions = [yaml.safe_load(p.read_text())["eval"]["version"] for p in (ROOT / "configs").glob("loop*.yaml")]
    assert len(set(map(str, versions))) == len(versions)       # two tracks sharing a version would compete
    assert isinstance(daily["eval"]["version"], int) and isinstance(minute["eval"]["version"], str)
    task, ev = minute["task"], minute["eval"]
    per_session = 390 - task["max_lookback"] - max(task["horizons"]) + 1
    assert per_session >= 180                                   # forecasts for at least three hours a day
    assert ev["embargo_bars"] >= per_session and ev["dm_lag"] >= max(task["horizons"])
    assert ev["holdout_frac"] >= daily["eval"]["holdout_frac"]
    assert all(t["model"]["lookback"] <= task["max_lookback"] for t in minute["tiers"].values())
    assert set(minute["tiers"]) == set(daily["tiers"])
    # Self-invented minute jobs: the grid has every key the worker draws and no lookback past a session's
    # window; the daily config hands out the share, and less than half of the worker's own jobs.
    assert set(daily["search"]) - {"tracks"} <= set(minute["search"]) and "tracks" not in minute["search"]
    assert all(lb <= task["max_lookback"] for opts in minute["search"]["lookback"].values() for lb in opts)
    assert set(minute["search"]["lookback"]) == set(minute["tiers"])
    assert 0 < sum(daily["search"]["tracks"].values()) < 0.5
    assert all((ROOT / "configs" / f"{name}.yaml").exists() for name in daily["search"]["tracks"])
