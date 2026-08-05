$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repo
$python = if (Test-Path ".venv-gpu\Scripts\python.exe") { ".venv-gpu\Scripts\python.exe" } else { "python" }
Write-Host "Running public-data preflight..."
& $python run.py change-preflight --config configs/change_public_windows.yaml
