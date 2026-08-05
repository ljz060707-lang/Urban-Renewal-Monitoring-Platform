[CmdletBinding(DefaultParameterSetName = 'All')]
param(
    [Parameter(ParameterSetName = 'Audit')][switch]$Audit,
    [Parameter(ParameterSetName = 'ValidateRecipe')][switch]$ValidateRecipe,
    [Parameter(ParameterSetName = 'Infer')][switch]$Infer,
    [Parameter(ParameterSetName = 'Objectize')][switch]$Objectize,
    [Parameter(ParameterSetName = 'Temporal')][switch]$Temporal,
    [Parameter(ParameterSetName = 'Figures')][switch]$Figures,
    [Parameter(ParameterSetName = 'All')][switch]$All,
    [Parameter(ParameterSetName = 'Status')][switch]$Status,
    [string]$RunRoot,
    [ValidateSet('cuda','cuda:0','cpu')][string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv-gpu\Scripts\python.exe'
$Tool = Join-Path $ProjectRoot 'tools\run_tongzhou_construction_images_best.py'
$Action = if ($Audit) { '--audit' } elseif ($ValidateRecipe) { '--validate-recipe' } elseif ($Infer) { '--infer' } elseif ($Objectize) { '--objectize' } elseif ($Temporal) { '--temporal' } elseif ($Figures) { '--figures' } elseif ($Status) { '--status' } else { '--all' }
$Arguments = @($Tool, '--project-root', $ProjectRoot, '--device', $Device, $Action)
if ($RunRoot) { $Arguments += @('--run-root', $RunRoot) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Tongzhou dedicated Construction pipeline failed with exit code $LASTEXITCODE" }
