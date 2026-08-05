param(
    [ValidateSet("Audit", "Infer", "BuildReport", "All", "Status")]
    [string]$Mode = "Status",
    [string]$SourceRoot = "D:\Data Set\Satellite\nanjing_jiangbei",
    [string]$OutputRoot = "",
    [string]$Device = "cuda:0",
    [string]$ConstructionDevice = "cuda",
    [string]$RsbuildingPython = "C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv-gpu\Scripts\python.exe"
$Tool = Join-Path $Project "tools\nanjing_jiangbei_acceptance.py"
if (-not (Test-Path -LiteralPath $Python)) { throw "Project inference Python missing: $Python" }
if (-not (Test-Path -LiteralPath $Tool)) { throw "Acceptance tool missing: $Tool" }

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $Project ("outputs\inference\nanjing_jiangbei_external_acceptance_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Log = Join-Path $OutputRoot ("nanjing_acceptance_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$Arguments = @(
    $Tool,
    "--mode", $Mode,
    "--source-root", $SourceRoot,
    "--output-root", $OutputRoot,
    "--device", $Device,
    "--construction-device", $ConstructionDevice,
    "--rsbuilding-python", $RsbuildingPython
)
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "Nanjing Jiangbei acceptance failed with exit code $LASTEXITCODE. See $Log" }
Write-Host "Nanjing Jiangbei acceptance output: $OutputRoot"
