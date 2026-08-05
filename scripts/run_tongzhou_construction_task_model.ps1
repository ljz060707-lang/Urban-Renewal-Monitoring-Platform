[CmdletBinding()]
param(
    [ValidateSet('cuda','cuda:0')][string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv-gpu\Scripts\python.exe'
& $Python (Join-Path $ProjectRoot 'tools\urrs.py') construction tongzhou run --device $Device
if ($LASTEXITCODE -ne 0) { throw "Tongzhou construction candidate inference failed with exit code $LASTEXITCODE" }
