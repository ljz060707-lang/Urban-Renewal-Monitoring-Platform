[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PackageDir,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$SampleId,
    [ValidateSet('cuda','cuda:0','cpu')][string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ScriptRoot '.venv-gpu\Scripts\python.exe'
$Python = if ($env:URRS_PYTHON) { $env:URRS_PYTHON } elseif (Test-Path -LiteralPath $ProjectPython) { $ProjectPython } else { 'python' }
$Urrs = Join-Path $ScriptRoot 'tools\urrs.py'

$arguments = @($Urrs, 'construction', 'task-model', 'infer', '--package-dir', $PackageDir, '--manifest', $Manifest, '--output-dir', $OutputDir, '--device', $Device)
if ($SampleId) { $arguments += @('--sample-id', $SampleId) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "construction task-model inference failed with exit code $LASTEXITCODE" }
