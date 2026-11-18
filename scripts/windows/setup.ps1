# One-time setup on the Windows GPU box. Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv python install 3.12
uv sync --extra dev
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    $k = Read-Host "Alpaca API key (paper)"
    $s = Read-Host "Alpaca secret key (paper)"
    (Get-Content .env) -replace '^ALPACA_API_KEY=.*', "ALPACA_API_KEY=$k" `
                       -replace '^ALPACA_SECRET_KEY=.*', "ALPACA_SECRET_KEY=$s" `
                       -replace '^DATA_ROOT=.*', "DATA_ROOT=$PWD\data" `
                       -replace '^OPTUNA_STORAGE=.*', "OPTUNA_STORAGE=" `
                       -replace '^MLFLOW_TRACKING_URI=.*', "MLFLOW_TRACKING_URI=" | Set-Content .env
}
uv run python -c "import torch, lightgbm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('lightgbm', lightgbm.__version__)"
uv run pytest -q -m "not slow and not network"
Write-Host "Setup done. Next: scripts\windows\run.ps1"
