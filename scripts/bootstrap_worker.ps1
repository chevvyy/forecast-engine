# One-time setup of a forecast-engine GPU worker on Windows. Safe to re-run.
#   irm https://raw.githubusercontent.com/chevvyy/forecast-engine/loop/scripts/bootstrap_worker.ps1 | iex
# What it does: clones the `loop` branch to %USERPROFILE%\forecast-worker, installs Python deps with uv
# (CUDA 12.8 PyTorch), runs a pre-flight check, stops the PC sleeping on AC power, and starts the worker
# now and at every logon (a .cmd in your Startup folder; delete it to uninstall).
$ErrorActionPreference = 'Stop'
$Name  = if ($env:FORECAST_WORKER_NAME)  { $env:FORECAST_WORKER_NAME }  else { 'pc-5070ti' }
$Tiers = if ($env:FORECAST_WORKER_TIERS) { $env:FORECAST_WORKER_TIERS } else { 'large,mid,nano' }
$Root  = Join-Path $env:USERPROFILE 'forecast-worker'
$Repo  = Join-Path $Root 'forecast-engine'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is not installed. Install Git for Windows, then re-run.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'Installing uv (Python package manager)...'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
New-Item -ItemType Directory -Force -Path $Root | Out-Null
if (-not (Test-Path (Join-Path $Repo '.git'))) {
    git clone -b loop https://github.com/chevvyy/forecast-engine $Repo
}
Set-Location $Repo
git fetch origin loop; git reset --hard origin/loop
git config user.name  "forecast-worker-$Name"
git config user.email "forecast-worker@users.noreply.github.com"
New-Item -ItemType Directory -Force -Path (Join-Path $Repo 'logs') | Out-Null

Write-Host 'Installing dependencies (first run downloads ~3 GB of PyTorch)...'
uv sync
Write-Host 'Pre-flight check (a GitHub sign-in window may open for push access)...'
uv run python -m forecast.loop.doctor
if ($LASTEXITCODE -ne 0) { throw 'Pre-flight failed. Fix the [FAIL] line above and re-run this script.' }

powercfg /change standby-timeout-ac 0 | Out-Null     # do not sleep while plugged in
powercfg /change hibernate-timeout-ac 0 | Out-Null

$Startup = [Environment]::GetFolderPath('Startup')
$Cmd = Join-Path $Startup 'forecast-worker.cmd'
"@echo off`r`nstart `"forecast-worker`" /min powershell -NoProfile -ExecutionPolicy Bypass -File `"$Repo\scripts\run_worker.ps1`" -Name $Name -Tiers $Tiers`r`n" |
    Set-Content -Encoding ASCII $Cmd
Start-Process powershell -WindowStyle Minimized -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$Repo\scripts\run_worker.ps1",'-Name',$Name,'-Tiers',$Tiers)
Write-Host ''
Write-Host "Worker '$Name' is running (minimized window). Log: $Repo\logs\worker.log"
Write-Host 'It restarts at every logon. To stop for good: close the window and delete' $Cmd
