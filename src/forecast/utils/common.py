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


def device_info() -> dict[str, Any]:
    """Describe the compute this process can use. VRAM numbers are in GB."""
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "gpu": None, "vram_total_gb": 0.0, "vram_free_gb": 0.0}
    free, total = torch.cuda.mem_get_info(0)
    return {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": round(total / 2**30, 2),
        "vram_free_gb": round(free / 2**30, 2),
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
