# Full pipeline on the PC: ingest (resumable) -> DQ -> baselines -> self-improving tuning loop.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run.ps1 [-Config configs\run\pc.yaml] [-SkipIngest]
param([string]$Config = "configs/run/pc.yaml", [switch]$SkipIngest, [int]$Rounds = 0)
$ErrorActionPreference = "Stop"
$log = "experiments\run-$(Get-Date -Format yyyyMMdd-HHmm).log"
if (-not $SkipIngest) { uv run fe-ingest --config $Config 2>&1 | Tee-Object -Append $log }
uv run fe-dq --config $Config 2>&1 | Tee-Object -Append $log
uv run fe-baselines --config $Config 2>&1 | Tee-Object -Append $log
# Runs until stopped (Ctrl+C) when Rounds = 0. Safe to kill and restart: the Optuna study resumes.
uv run fe-autotune --config $Config --rounds $Rounds 2>&1 | Tee-Object -Append $log
