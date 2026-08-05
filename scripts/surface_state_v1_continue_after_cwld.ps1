param(
    [string]$CwldRunDir = "outputs\training\surface_state_v1\cwld_20260729_165008",
    [ValidateSet(
        "status",
        "finalize-cwld",
        "fusu",
        "temporal",
        "joint",
        "mti",
        "register",
        "overlay",
        "tongzhou",
        "all"
    )]
    [string]$Stage = "status"
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project ".venv-gpu\Scripts\python.exe"
$ContinueCli = Join-Path $Project "tools\surface_state_v1_continue_cli.py"
$TrainCli = Join-Path $Project "tools\surface_state_v1_train_cli.py"
$RuntimeCli = Join-Path $Project "tools\surface_state_v1_runtime_cli.py"
$PipelineScript = Join-Path $Project "scripts\surface_state_v1_pipeline.ps1"
$Run = if ([IO.Path]::IsPathRooted($CwldRunDir)) { $CwldRunDir } else { Join-Path $Project $CwldRunDir }
$PipelineRoot = Join-Path $Project ("outputs\pipeline\surface_state_v1_continue\" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $PipelineRoot | Out-Null
Start-Transcript -Path (Join-Path $PipelineRoot "pipeline.log") | Out-Null

function Stop-WithError([string]$Message) {
    Write-Error $Message
    Stop-Transcript | Out-Null
    exit 1
}

function Invoke-Python([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "Stage command failed with exit code $LASTEXITCODE"
    }
}

function Get-CwldProcess {
    $leaf = Split-Path -Leaf $Run
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "^python" -and
            ($_.CommandLine -like "*$leaf*" -or $_.CommandLine -like "*cwld_train.yaml*")
        })
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Show-CwldStatus {
    $receipt = Read-Json (Join-Path $Run "TRAINING_RECEIPT.json")
    $latest = $null
    $log = Join-Path $Run "logs\epochs.jsonl"
    if (Test-Path -LiteralPath $log) {
        $line = Get-Content -LiteralPath $log -Tail 1
        if ($line) { $latest = $line | ConvertFrom-Json }
    }
    $selection = Read-Json (Join-Path $Run "CHECKPOINT_SELECTION.json")
    [ordered]@{
        status = if ((Get-CwldProcess).Count -gt 0) { "CWLD_TRAINING_ACTIVE" } elseif ($receipt) { $receipt.status } else { "missing_receipt" }
        run_dir = $Run
        latest_epoch = if ($latest) { $latest.epoch } else { $null }
        latest_validation_macro_mIoU = if ($latest) { $latest.metrics.cwld.macro_mIoU } else { $null }
        best = if ($selection) { $selection.best } else { $null }
        test_reads = if ($receipt) { $receipt.test_read_count } else { $null }
    } | ConvertTo-Json -Depth 8
}

if (-not (Test-Path -LiteralPath $Python)) { Stop-WithError "Missing project Python: $Python" }
if (-not (Test-Path -LiteralPath $Run)) { Stop-WithError "Missing CWLD run: $Run" }

$active = (Get-CwldProcess).Count -gt 0
if ($active) {
    Show-CwldStatus
    Write-Host "CWLD_TRAINING_ACTIVE"
    if ($Stage -ne "status") {
        Stop-WithError "CWLD training is active; no dependent GPU stage was started."
    }
    Stop-Transcript | Out-Null
    exit 0
}

if ($Stage -eq "status") {
    Show-CwldStatus
    Stop-Transcript | Out-Null
    exit 0
}

$receiptPath = Join-Path $Run "TRAINING_RECEIPT.json"
$receipt = Read-Json $receiptPath
if (-not $receipt -or $receipt.status -ne "success") {
    if ($Stage -notin @("finalize-cwld", "all")) {
        Stop-WithError "CWLD run is incomplete. Run -Stage finalize-cwld to resume safely first."
    }
    Write-Host "CWLD run incomplete; resuming from last.pth with historical-best guard."
    Invoke-Python @(
        $ContinueCli,
        "resume-cwld",
        "--config", "extensions\surface_state_v1\configs\cwld_train.yaml",
        "--run-dir", $Run
    )
}

if ($Stage -in @("finalize-cwld", "all")) {
    Invoke-Python @($ContinueCli, "cwld-finalize", "--run-dir", $Run)
}

$final = Read-Json (Join-Path $Run "CWLD_FINALIZATION_RECEIPT.json")
if ($Stage -notin @("finalize-cwld", "status") -and
    (-not $final -or $final.candidate_status -ne "cwld_validation_candidate")) {
    Stop-WithError "CWLD final candidate gate did not pass; dependent training is blocked."
}

if ($Stage -in @("fusu", "all")) {
    & $PipelineScript -Stage fusu
    if ($LASTEXITCODE -ne 0) { Stop-WithError "FUSU baseline training failed." }
}
if ($Stage -in @("temporal", "all")) {
    & $PipelineScript -Stage temporal
    if ($LASTEXITCODE -ne 0) { Stop-WithError "FUSU temporal training failed." }
}
if ($Stage -in @("joint", "all")) {
    & $PipelineScript -Stage joint
    if ($LASTEXITCODE -ne 0) { Stop-WithError "Joint training failed." }
}
if ($Stage -in @("mti", "all")) {
    & $PipelineScript -Stage mti
    if ($LASTEXITCODE -ne 0) { Stop-WithError "MTI calibration failed." }
}
if ($Stage -in @("register", "all")) {
    & $PipelineScript -Stage register
    if ($LASTEXITCODE -ne 0) { Stop-WithError "Candidate registration failed." }
}
if ($Stage -in @("overlay", "all")) {
    & $PipelineScript -Stage overlay
    if ($LASTEXITCODE -ne 0) { Stop-WithError "Final overlay failed." }
}
if ($Stage -in @("tongzhou", "all")) {
    $TongzhouCli = Join-Path $Project "tools\surface_state_v1_tongzhou_cli.py"
    if (-not (Test-Path -LiteralPath $TongzhouCli)) {
        Stop-WithError "Tongzhou acceptance CLI is not installed."
    }
    Invoke-Python @($TongzhouCli, "--project", $Project)
}

$Full20 = Join-Path $Project "outputs\pretraining\change_v2\20260727_001029\checkpoints\best_v2_multisource_change.pth"
$RSBuilding = Join-Path $Project "outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth"
if ((Get-FileHash $Full20 -Algorithm SHA256).Hash.ToLower() -ne "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621") {
    Stop-WithError "Full20 hash changed."
}
if ((Get-FileHash $RSBuilding -Algorithm SHA256).Hash.ToLower() -ne "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7") {
    Stop-WithError "RSBuilding hash changed."
}
Write-Host "Stage completed. WHU Test reads=0; SECOND Test reads=0."
Stop-Transcript | Out-Null
