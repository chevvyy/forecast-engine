#!/usr/bin/env bash
# Update code, sync deps, run the worker. systemd restarts this whenever the worker exits.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
git fetch origin loop && git reset --hard origin/loop
uv sync --quiet
exec uv run python -m forecast.loop.worker --name "${1:-server-3060}" --tiers "${2:-mid,nano}" >> logs/worker.log 2>&1
