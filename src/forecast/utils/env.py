"""Environment / .env access. Fail fast on missing required keys."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from forecast.utils.config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


def get(key: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var {key}; set it in .env (see .env.example)")
    return val


def data_root() -> Path:
    root = Path(get("DATA_ROOT", str(REPO_ROOT / "data")) or "")
    root.mkdir(parents=True, exist_ok=True)
    return root


def experiments_dir() -> Path:
    d = REPO_ROOT / "experiments"
    d.mkdir(exist_ok=True)
    return d
