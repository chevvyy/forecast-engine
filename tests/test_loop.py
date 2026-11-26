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
