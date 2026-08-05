[CmdletBinding()]
param(
    [string]$RunRoot,
    [switch]$Audit,
    [switch]$Full20Attribution,
    [switch]$LeakageScan,
    [switch]$PrepareLabels,
    [switch]$IngestLabels,
    [switch]$Train,
    [switch]$SelectModel,
    [switch]$InferBuilding,
    [switch]$TrackBuilding,
    [switch]$RunConstruction,
    [switch]$RenderFigures,
    [switch]$Accept,
    [switch]$All,
    [switch]$Resume,
    [switch]$Status,
    [string]$LabelRoot
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$tool = Join-Path $projectRoot "tools\run_nanjing_domain_adaptation_stable_tracking.py"
if (-not (Test-Path -LiteralPath $python)) { throw "Project GPU Python not found: $python" }
if (-not (Test-Path -LiteralPath $tool)) { throw "Controller not found: $tool" }

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $projectRoot ("outputs\inference\nanjing_domain_adaptation_stable_tracking_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}

$selected = @()
if ($Audit) { $selected += "audit" }
if ($Full20Attribution) { $selected += "full20" }
if ($LeakageScan) { $selected += "leakage" }
if ($PrepareLabels) { $selected += "prepare" }
if ($IngestLabels) { $selected += "ingest" }
if ($Train) { $selected += "train" }
if ($SelectModel) { $selected += "select" }
if ($InferBuilding) { $selected += "infer" }
if ($TrackBuilding) { $selected += "track" }
if ($RunConstruction) { $selected += "construction" }
if ($RenderFigures) { $selected += "figures" }
if ($Accept) { $selected += "accept" }
if ($All) { $selected += "all" }
if ($Resume) { $selected += "resume" }
if ($Status) { $selected += "status" }
if ($selected.Count -gt 1) { throw "Select one mode switch only: $($selected -join ', ')" }
if ($selected.Count -eq 0) { $selected = @("all") }

$arguments = @($tool, "--run-root", $RunRoot, "--mode", $selected[0])
if (-not [string]::IsNullOrWhiteSpace($LabelRoot)) { $arguments += @("--label-root", $LabelRoot) }
& $python @arguments
exit $LASTEXITCODE
