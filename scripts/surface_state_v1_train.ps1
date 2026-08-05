param(
    [string]$FusuRoot = "D:\Data Set\FUSU",
    [string]$CwldRoot = "D:\Data Set\CWLD",
    [ValidateSet("audit","cwld-train","fusu-train","joint-train","temporal-train","mti","register","overlay","all")]
    [string]$Stage = "audit"
)
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
$Proj = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Proj ".venv-gpu\Scripts\python.exe"
$Cli = Join-Path $Proj "tools\surface_state_v1_train_cli.py"
if (-not (Test-Path $Python)) { throw "Missing project Python: $Python" }
if (-not (Test-Path $FusuRoot)) { throw "Missing FUSU root: $FusuRoot" }
if (-not (Test-Path $CwldRoot)) { throw "Missing CWLD root: $CwldRoot" }
& $Python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Stage -in @("audit","all")) {
    & $Python $Cli fusu-split
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Stage -in @("cwld-train","all")) {
    & $Python $Cli cwld-train --config "extensions/surface_state_v1/configs/cwld_train.yaml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Stage -in @("fusu-train","all")) {
    & $Python $Cli fusu-train --config "extensions/surface_state_v1/configs/fusu_sample_train.yaml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Stage -in @("temporal-train","all")) {
    & $Python $Cli temporal-train --config "extensions/surface_state_v1/configs/temporal_tfr_sample_train.yaml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Stage -in @("joint-train","all")) {
    & $Python $Cli joint-train --config "extensions/surface_state_v1/configs/surface_joint_sample_train.yaml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Full20 expected SHA256: 70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
Write-Host "RSBuilding expected SHA256: f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
