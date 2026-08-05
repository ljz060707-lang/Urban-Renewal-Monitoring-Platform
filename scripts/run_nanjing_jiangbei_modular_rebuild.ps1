param(
    [ValidateSet('Audit','Baseline','Construction','CrossReference','BuildReport','VerifyInvariance','All','Status')]
    [string]$Mode = 'All',
    [string]$SourceRoot = 'D:\Data Set\Satellite\nanjing_jiangbei',
    [string]$OutputRoot = '',
    [string]$Device = 'cuda:0',
    [string]$ConstructionDevice = 'cuda',
    [string]$RsBuildingPython = 'C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project runtime Python is missing: $Python"
}

$Arguments = @(
    (Join-Path $ProjectRoot 'tools\nanjing_jiangbei_modular_rebuild.py'),
    '--mode', $Mode,
    '--source-root', $SourceRoot,
    '--device', $Device,
    '--construction-device', $ConstructionDevice,
    '--rsbuilding-python', $RsBuildingPython,
    '--python', $Python
)
if ($OutputRoot) {
    $Arguments += @('--output-root', $OutputRoot)
}

& $Python @Arguments
exit $LASTEXITCODE
