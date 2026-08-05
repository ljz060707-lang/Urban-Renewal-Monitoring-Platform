param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string[]]$Command,
    [int]$TimeoutSeconds = 3600
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
$runner = Join-Path $ProjectRoot "tools\codex_windows_fallback_runner.py"
if (-not (Test-Path -LiteralPath $python)) { throw "Project Python not found: $python" }
if (-not (Test-Path -LiteralPath $runner)) { throw "Fallback runner not found: $runner" }
$args = @($runner, "--cwd", $ProjectRoot, "--log", $LogPath, "--timeout", $TimeoutSeconds, "--") + $Command
& $python @args
exit $LASTEXITCODE
