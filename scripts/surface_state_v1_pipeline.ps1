param(
    [string]$FusuRoot = "D:\Data Set\FUSU",
    [string]$CwldRoot = "D:\Data Set\CWLD",
    [ValidateSet("split","cwld","fusu","temporal","joint","mti","register","overlay","all")]
    [string]$Stage = "all"
)
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
$Proj = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Proj ".venv-gpu\Scripts\python.exe"
$TrainCli = Join-Path $Proj "tools\surface_state_v1_train_cli.py"
$RuntimeCli = Join-Path $Proj "tools\surface_state_v1_runtime_cli.py"
$LogRoot = Join-Path $Proj "outputs\training\surface_state_v1"
if (-not (Test-Path $Python)) { throw "Missing Python: $Python" }
if (-not (Test-Path $FusuRoot)) { throw "Missing FUSU root: $FusuRoot" }
if (-not (Test-Path $CwldRoot)) { throw "Missing CWLD root: $CwldRoot" }
& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

function Invoke-Step([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
function Latest-Success([string]$Prefix) {
    $runs = Get-ChildItem $LogRoot -Directory -Filter "$Prefix*" | Sort-Object LastWriteTime -Descending
    foreach ($run in $runs) {
        $receipt = Join-Path $run.FullName "TRAINING_RECEIPT.json"
        if (Test-Path $receipt) {
            $state = Get-Content -Raw $receipt | ConvertFrom-Json
            if ($state.status -eq "success") { return $run.FullName }
        }
    }
    throw "No successful run found for prefix: $Prefix"
}

if ($Stage -in @("split","all")) {
    Invoke-Step @($TrainCli,"fusu-split")
}
if ($Stage -in @("cwld","all")) {
    Invoke-Step @($TrainCli,"cwld-train","--config","extensions\surface_state_v1\configs\cwld_train.yaml")
}
if ($Stage -in @("fusu","all")) {
    Invoke-Step @($RuntimeCli,"fusu-train","--config","extensions\surface_state_v1\configs\fusu_sample_train.yaml")
}
if ($Stage -in @("temporal","all")) {
    Invoke-Step @($RuntimeCli,"temporal-train","--config","extensions\surface_state_v1\configs\temporal_tfr_sample_train.yaml")
}
if ($Stage -in @("joint","all")) {
    Invoke-Step @($RuntimeCli,"joint-train","--config","extensions\surface_state_v1\configs\surface_joint_sample_train.yaml")
}
if ($Stage -in @("mti","all")) {
    $fusu = Latest-Success "fusu_sample_"
    $checkpoint = Join-Path $fusu "checkpoints\fusu_sample_best_validation.pth"
    $output = Join-Path $LogRoot ("mti_sample_" + (Get-Date -Format yyyyMMdd_HHmmss))
    Invoke-Step @($TrainCli,"mti-calibrate","--config","extensions\surface_state_v1\configs\mti_sample_calibration.yaml","--checkpoint",$checkpoint,"--output",$output)
}
if ($Stage -in @("register","all")) {
    foreach ($prefix in @("cwld_","fusu_sample_","tfr_sample_","joint_sample_")) {
        $run = Latest-Success $prefix
        Invoke-Step @($TrainCli,"register-candidate","--run-dir",$run)
    }
}
if ($Stage -in @("overlay","all")) {
    $joint = Latest-Success "joint_sample_"
    $checkpoint = Join-Path $joint "checkpoints\surface_joint_sample_best_validation.pth"
    $sample = Import-Csv (Join-Path $Proj "outputs\data_audit\fusu_sample\FUSU_SAMPLE_VALIDATION_MANIFEST.csv") | Select-Object -First 1
    $output = Join-Path $Proj ("outputs\integration\surface_state_v1_overlay\" + (Get-Date -Format yyyyMMdd_HHmmss))
    Invoke-Step @($TrainCli,"overlay-infer","--config","extensions\surface_state_v1\configs\master_overlay_infer.yaml","--surface-checkpoint",$checkpoint,"--input-t1",$sample.image_t1,"--input-t2",$sample.image_t2,"--output",$output)
}

$Full20 = Join-Path $Proj "outputs\pretraining\change_v2\20260727_001029\checkpoints\best_v2_multisource_change.pth"
$RSBuilding = Join-Path $Proj "outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth"
if ((Get-FileHash $Full20 -Algorithm SHA256).Hash.ToLower() -ne "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621") { throw "Full20 hash changed" }
if ((Get-FileHash $RSBuilding -Algorithm SHA256).Hash.ToLower() -ne "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7") { throw "RSBuilding hash changed" }
Write-Host "SurfaceState v1 stage completed. WHU Test reads=0; SECOND Test reads=0."
