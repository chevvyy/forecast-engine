# Keeps the worker alive: update code, sync deps, run, repeat. Started by bootstrap_worker.ps1.
param([string]$Name = 'pc-5070ti', [string]$Tiers = 'large,mid,nano')
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Force -Path logs | Out-Null
while ($true) {
    git fetch origin loop 2>&1 | Out-Null
    git reset --hard origin/loop 2>&1 | Out-Null
    uv sync --quiet 2>&1 | Out-File -Append -Encoding utf8 logs\worker.log
    cmd /c "uv run python -m forecast.loop.worker --name $Name --tiers $Tiers >> logs\worker.log 2>&1"
    Start-Sleep -Seconds 30
}
