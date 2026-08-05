[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [int]$Port = 8765,
    [switch]$PrepareQueues,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$server = Join-Path $repo 'tools\urrs_lite_review\server.py'
$prep = Join-Path $repo 'tools\urrs_lite_review\prepare_review_queues.py'
$runtimePython = 'C:\Users\ljz52\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $runtimePython = $cmd.Source } else { throw 'Bundled Python was not found.' }
}

& $runtimePython $prep -Status | Out-Host
${backupRoot} = Join-Path $repo 'outputs\urrs\review\lite_review\backups'
$today = Get-Date -Format 'yyyyMMdd'
$todayBackup = Get-ChildItem -LiteralPath $backupRoot -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "$today*" } | Select-Object -First 1
if (-not $todayBackup) { & $runtimePython $prep -Backup | Out-Host }
if ($PrepareQueues) { & $runtimePython $prep -PrepareQueues | Out-Host }
if ($Status) { exit 0 }

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalAddress -eq '127.0.0.1' }
if (-not $existing) {
    $args = @($server, '--host', '127.0.0.1', '--port', "$Port")
    Start-Process -WindowStyle Hidden -FilePath $runtimePython -ArgumentList $args -WorkingDirectory $repo | Out-Null
    Start-Sleep -Milliseconds 700
}
$health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 10
if ($health.StatusCode -ne 200) { throw 'URRS Lite Review server did not pass local health check.' }
Write-Output "URRS Lite Review ready: http://127.0.0.1:$Port"
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port" | Out-Null }
