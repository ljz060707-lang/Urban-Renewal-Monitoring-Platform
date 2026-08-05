param(
    [ValidateSet("Prepare", "Launch", "Status", "Finalize", "Resume")]
    [string]$Mode = "Status",
    [string]$RunDir = "",
    [string]$UcsRoot = "D:\Data Set\URS\600 manually delineated UCS labels part2",
    [ValidateSet("cuda", "cuda:0")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
$Worker = Join-Path $ProjectRoot "tools\construction_training_worker.py"
$Config = Join-Path $ProjectRoot "configs\urrs\construction\ucs386_task_candidate.yaml"
if (-not (Test-Path -LiteralPath $Python)) { throw "项目 Python 不存在: $Python" }
if (-not (Test-Path -LiteralPath $Worker)) { throw "training worker 不存在: $Worker" }
$env:PYTHONUTF8 = "1"

function Write-Utf8NoBomAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $target = [System.IO.Path]::GetFullPath($Path)
    $directory = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporary = Join-Path $directory ("." + [System.IO.Path]::GetFileName($target) + "." + [guid]::NewGuid().ToString("N") + ".tmp")
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = [System.IO.File]::Open($temporary, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $bytes = $encoding.GetBytes($Content)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
    if (Test-Path -LiteralPath $target) {
        [System.IO.File]::Replace($temporary, $target, $null, $true)
    }
    else {
        [System.IO.File]::Move($temporary, $target)
    }
}

function Find-LatestRun {
    $trainingRoot = Join-Path $ProjectRoot "outputs\training\construction_activity_v1"
    if (-not (Test-Path -LiteralPath $trainingRoot)) { throw "尚未 Prepare: $trainingRoot" }
    $candidate = Get-ChildItem -LiteralPath $trainingRoot -Directory |
        Where-Object { $_.Name -like "URRS-Construction-UCS386-v1.0-task-candidate_*" -and (Test-Path -LiteralPath (Join-Path $_.FullName "PREPARE_RECEIPT.json")) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $candidate) { throw "未找到包含 PREPARE_RECEIPT.json 的 UCS386 run directory" }
    return $candidate.FullName
}

function Resolve-Run {
    if ([string]::IsNullOrWhiteSpace($RunDir)) { return Find-LatestRun }
    $resolved = [System.IO.Path]::GetFullPath($RunDir)
    if (-not (Test-Path -LiteralPath $resolved)) { throw "RunDir 不存在: $resolved" }
    return $resolved
}

function Start-DetachedTraining([bool]$ResumeTraining) {
    $resolvedRun = Resolve-Run
    $runConfig = Join-Path $resolvedRun "run_config.yaml"
    if (-not (Test-Path -LiteralPath $runConfig)) { throw "run_config.yaml 不存在: $runConfig" }
    $logDir = Join-Path $resolvedRun "logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stdoutPath = Join-Path $logDir ($(if ($ResumeTraining) { "training_resume_stdout.log" } else { "detached_stdout.log" }))
    $stderrPath = Join-Path $logDir ($(if ($ResumeTraining) { "training_resume_stderr.log" } else { "detached_stderr.log" }))
    $argumentList = @(
        $Worker,
        "--mode", "train",
        "--project-root", $ProjectRoot,
        "--run-dir", $resolvedRun,
        "--config", $runConfig
    )
    if ($ResumeTraining) { $argumentList += "--resume" }
    $commandText = '"' + $Python + '" ' + (($argumentList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
    $commandBytes = [Text.Encoding]::UTF8.GetBytes($commandText)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $commandHash = -join ($sha256.ComputeHash($commandBytes) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $sha256.Dispose()
    }
    $startedAt = [DateTimeOffset]::UtcNow.ToString("o")
    $process = Start-Process -FilePath $Python -ArgumentList $argumentList -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
    $originalPid = 54708
    $originalReceiptPath = Join-Path $resolvedRun "TRAINING_LAUNCH_RECEIPT.json"
    if (Test-Path -LiteralPath $originalReceiptPath) {
        try {
            $originalText = [System.IO.File]::ReadAllText($originalReceiptPath, (New-Object System.Text.UTF8Encoding($false))).TrimStart([char]0xFEFF)
            $originalReceipt = $originalText | ConvertFrom-Json
            if ($null -ne $originalReceipt.pid) { $originalPid = [int]$originalReceipt.pid }
        }
        catch {
            $originalPid = 54708
        }
    }
    $resumeCheckpoint = Join-Path $resolvedRun "checkpoints\last.pth"
    $receiptName = if ($ResumeTraining) { "TRAINING_RESUME_RECEIPT.json" } else { "TRAINING_LAUNCH_RECEIPT.json" }
    $receiptStatus = if ($ResumeTraining) { "DETACHED_RESUME_PROCESS_STARTED" } else { "DETACHED_PROCESS_STARTED" }
    $launchReceipt = [ordered]@{
        status = $receiptStatus
        training_process_started = $true
        detached = $true
        pid = $process.Id
        new_pid = $process.Id
        run_directory = $resolvedRun
        python = $Python
        worker = $Worker
        stdout = $stdoutPath
        stderr = $stderrPath
        command = $commandText
        command_sha256 = $commandHash
        resume = $ResumeTraining
        original_pid = if ($ResumeTraining) { $originalPid } else { $null }
        failure_reason = if ($ResumeTraining) { "utf8_bom_json_decode_error" } else { $null }
        resume_checkpoint = if ($ResumeTraining) { $resumeCheckpoint } else { $null }
        resume_epoch = if ($ResumeTraining) { 0 } else { $null }
        resume_global_step = if ($ResumeTraining) { 0 } else { $null }
        resume_origin = if ($ResumeTraining) { "partial_checkpoint_incomplete_restart_epoch0" } else { $null }
        partial_uncheckpointed_steps_discarded = if ($ResumeTraining) { $true } else { $false }
        same_run_directory = $true
        new_training_run_created = $false
        started_at = $startedAt
        test_pixel_reads = 0
        locked_test_used_for_selection = $false
    }
    $receiptText = $launchReceipt | ConvertTo-Json -Depth 8
    Write-Utf8NoBomAtomic -Path (Join-Path $resolvedRun $receiptName) -Content $receiptText
    $pidReceipt = [ordered]@{
        pid = $process.Id
        original_pid = if ($ResumeTraining) { $originalPid } else { $null }
        run_directory = $resolvedRun
        started_at = $startedAt
        detached = $true
    }
    Write-Utf8NoBomAtomic -Path (Join-Path $resolvedRun "TRAINING_PID.json") -Content ($pidReceipt | ConvertTo-Json -Depth 8)
    $receiptText
}

switch ($Mode) {
    "Prepare" {
        & $Python $Worker --mode prepare --project-root $ProjectRoot --config $Config --ucs-root $UcsRoot
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "Launch" {
        Start-DetachedTraining $false
    }
    "Resume" {
        Start-DetachedTraining $true
    }
    "Status" {
        $resolvedRun = Resolve-Run
        & $Python $Worker --mode status --project-root $ProjectRoot --run-dir $resolvedRun
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "Finalize" {
        $resolvedRun = Resolve-Run
        $runConfig = Join-Path $resolvedRun "run_config.yaml"
        & $Python $Worker --mode finalize --project-root $ProjectRoot --run-dir $resolvedRun --config $runConfig
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
