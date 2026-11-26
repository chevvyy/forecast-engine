"""Small shared helpers: seeding, device/VRAM discovery, JSON IO, git info."""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def usable_vram_gb(free_bytes: int, reserved_bytes: int = 0, allocated_bytes: int = 0) -> float:
    """VRAM this process could still use, in GB: what the driver reports free plus the blocks PyTorch's
    caching allocator holds for this process but has no tensor in. The driver counts that cache as
    used, so without adding it back a worker sees its own earlier job as "someone else is using the
    GPU" and stops taking jobs that need more than what is left."""
    return (free_bytes + max(reserved_bytes - allocated_bytes, 0)) / 2**30


def release_vram() -> None:
    """Hand this process's cached GPU memory back to the driver (call between jobs)."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_info() -> dict[str, Any]:
    """Describe the compute this process can use. VRAM numbers are in GB; ``vram_free_gb`` is what
    this process could use (see ``usable_vram_gb``), not just what the driver reports free."""
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "gpu": None, "vram_total_gb": 0.0, "vram_free_gb": 0.0}
    free, total = torch.cuda.mem_get_info(0)
    usable = usable_vram_gb(free, torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))
    return {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": round(total / 2**30, 2),
        "vram_free_gb": round(min(usable, total / 2**30), 2),
    }


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def git_sha(repo: Path = REPO) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT") or REPO / "data")


DAY = 86400
_UNIT = {"m": 60, "h": 3600, "d": DAY}


def interval_seconds(interval: str) -> int:
    """``"1m"`` -> 60, ``"5m"`` -> 300, ``"1h"`` -> 3600, ``"1d"`` -> 86400."""
    n, unit = interval[:-1], interval[-1:].lower()
    if unit not in _UNIT or not n.isdigit() or int(n) < 1:
        raise ValueError(f"unknown bar interval {interval!r} (expected like 1m, 5m, 1h, 1d)")
    return int(n) * _UNIT[unit]
