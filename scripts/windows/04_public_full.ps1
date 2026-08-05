$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repo
$python = if (Test-Path ".venv-gpu\Scripts\python.exe") { ".venv-gpu\Scripts\python.exe" } else { "python" }
Write-Host "Starting explicit public full training..."
& $python -u run.py change-all --config configs/change_public_windows.yaml --device cuda:0
