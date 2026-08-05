param(
    [ValidateSet('Audit','Building','EvaluateBuilding','Construction','EvaluateConstruction','Evaluate','Cross','Figures','Report','All','Status')]
    [string]$Mode = 'All',
    [string]$RunRoot = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv-gpu\Scripts\python.exe'
$Tool = Join-Path $ProjectRoot 'tools\run_nanjing_building_construction_integrated.py'
if (-not $RunRoot) {
    $RunRoot = Join-Path $ProjectRoot ('outputs\inference\nanjing_building_construction_integrated_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$ModeMap = @{
    Audit = 'audit'; Building = 'building'; EvaluateBuilding = 'evaluate-building'; Construction = 'construction';
    EvaluateConstruction = 'evaluate-construction'; Evaluate = 'evaluate'; Cross = 'cross'; Figures = 'figures';
    Report = 'report'; All = 'all'; Status = 'status'
}
& $Python $Tool --run-root $RunRoot --mode $ModeMap[$Mode]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
