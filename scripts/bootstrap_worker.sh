#!/usr/bin/env bash
# One-time setup of a forecast-engine GPU worker on Linux (e.g. the RTX 3060 guest). Safe to re-run.
#   FORECAST_WORKER_NAME=server-3060 FORECAST_WORKER_TIERS=mid,nano bash scripts/bootstrap_worker.sh
# Needs: git, curl, an NVIDIA driver that `nvidia-smi` can see, and push access to the repo. For push
# access on a headless box use a fine-grained GitHub token limited to this one repository
# (Contents: read and write), stored with `git config credential.helper store`. Never commit it.
set -euo pipefail
NAME="${FORECAST_WORKER_NAME:-server-3060}"
TIERS="${FORECAST_WORKER_TIERS:-mid,nano}"
ROOT="${FORECAST_WORKER_ROOT:-$HOME/forecast-worker}"
REPO="$ROOT/forecast-engine"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
mkdir -p "$ROOT"
[ -d "$REPO/.git" ] || git clone -b loop https://github.com/chevvyy/forecast-engine "$REPO"
cd "$REPO"
git fetch origin loop && git reset --hard origin/loop
git config user.name "forecast-worker-$NAME"
git config user.email "forecast-worker@users.noreply.github.com"
mkdir -p logs
uv sync
uv run python -m forecast.loop.doctor
UNIT="$HOME/.config/systemd/user/forecast-worker.service"
mkdir -p "$(dirname "$UNIT")"
cat > "$UNIT" <<UNITEOF
[Unit]
Description=forecast-engine GPU worker ($NAME)
After=network-online.target

[Service]
WorkingDirectory=$REPO
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/env bash $REPO/scripts/run_worker.sh $NAME $TIERS
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
UNITEOF
systemctl --user daemon-reload
systemctl --user enable --now forecast-worker.service
loginctl enable-linger "$USER" 2>/dev/null || true
echo "Worker '$NAME' running. Logs: $REPO/logs/worker.log   Stop: systemctl --user disable --now forecast-worker"
