[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [switch]$Audit,
    [switch]$Benchmark,
    [switch]$Building,
    [switch]$Tracking,
    [switch]$TrainingDecision,
    [switch]$TrainBuilding,
    [switch]$Construction,
    [switch]$Figures,
    [switch]$Accept,
    [switch]$All,
    [switch]$Resume,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$tool = Join-Path $projectRoot "tools\run_nanjing_stable_building_construction.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}

$selected = @(
    @{ Flag = $Audit; Mode = "audit" },
    @{ Flag = $Benchmark; Mode = "benchmark" },
    @{ Flag = $Building; Mode = "building" },
    @{ Flag = $Tracking; Mode = "tracking" },
    @{ Flag = $TrainingDecision; Mode = "training-decision" },
    @{ Flag = $TrainBuilding; Mode = "train-building" },
    @{ Flag = $Construction; Mode = "construction" },
    @{ Flag = $Figures; Mode = "figures" },
    @{ Flag = $Accept; Mode = "accept" },
    @{ Flag = $All; Mode = "all" },
    @{ Flag = $Resume; Mode = "resume" },
    @{ Flag = $Status; Mode = "status" }
) | Where-Object { $_.Flag }

if ($selected.Count -gt 1) {
    throw "Select only one mode switch."
}
$mode = if ($selected.Count -eq 1) { $selected[0].Mode } else { "status" }

& $python $tool --run-root $RunRoot --mode $mode
exit $LASTEXITCODE
