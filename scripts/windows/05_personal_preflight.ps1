param(
  [Parameter(Mandatory=$true)][string]$AoiId,
  [string]$DataRoot = "D:/Data Set/Personal"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repo
$python = if (Test-Path ".venv-gpu\Scripts\python.exe") { ".venv-gpu\Scripts\python.exe" } else { "python" }
Write-Host "Checking personal AOI $AoiId..."
& $python run.py personal-preflight --aoi $AoiId --data-root $DataRoot
