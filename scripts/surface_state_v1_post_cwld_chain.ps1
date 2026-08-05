param(
    [int]$CwldControllerPid,
    [string]$CwldRunDir = "outputs\training\surface_state_v1\cwld_20260729_165008"
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project ".venv-gpu\Scripts\python.exe"
$PipelineCli = Join-Path $Project "tools\surface_state_v1_final_pipeline_cli.py"
$CwldRun = if ([IO.Path]::IsPathRooted($CwldRunDir)) { $CwldRunDir } else { Join-Path $Project $CwldRunDir }
$LogRoot = Join-Path $Project ("outputs\pipeline\surface_state_v1_post_cwld\" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $LogRoot | Out-Null
Start-Transcript -Path (Join-Path $LogRoot "chain.log") | Out-Null

while (Get-Process -Id $CwldControllerPid -ErrorAction SilentlyContinue) {
    Write-Host "Waiting for CWLD controller PID $CwldControllerPid..."
    Start-Sleep -Seconds 30
}

$FinalReceipt = Join-Path $CwldRun "CWLD_FINALIZATION_RECEIPT.json"
if (-not (Test-Path -LiteralPath $FinalReceipt)) {
    throw "CWLD controller exited without finalization receipt: $FinalReceipt"
}
$Final = Get-Content -LiteralPath $FinalReceipt -Raw | ConvertFrom-Json
if ($Final.status -ne "success" -or $Final.candidate_status -ne "cwld_validation_candidate") {
    throw "CWLD final candidate gate did not pass; downstream GPU stages were not started."
}

$OtherGpuTraining = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match "^python" -and
        $_.CommandLine -match "surface_state_v1_(runtime|experiment|final_pipeline)_cli"
    })
if ($OtherGpuTraining.Count -gt 0) {
    throw "Another SurfaceState GPU pipeline is active; downstream stages were not started."
}

$PipelineDir = Join-Path $Project ("outputs\pipeline\surface_state_v1_final\" + (Get-Date -Format "yyyyMMdd_HHmmss"))
& $Python $PipelineCli --cwld-run $CwldRun --pipeline-dir $PipelineDir
if ($LASTEXITCODE -ne 0) {
    throw "SurfaceState final pipeline failed with exit code $LASTEXITCODE"
}
Write-Host "SurfaceState final pipeline completed: $PipelineDir"
Stop-Transcript | Out-Null
