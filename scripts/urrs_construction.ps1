param(
    [ValidateSet("discover","audit","pixel-smoke","ucs-train","joint-train","parcel","temporal-labels","temporal","mti-eval","tongzhou","review","status")]
    [string]$Stage = "status",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "项目 Python 不存在: $Python" }
$env:PYTHONUTF8 = "1"

switch ($Stage) {
    "discover" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction discover }
    "audit" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction audit }
    "pixel-smoke" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction smoke --device $Device }
    "ucs-train" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction train --stage ucs --device $Device }
    "joint-train" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction train --stage joint --device $Device }
    "parcel" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction parcel train }
    "temporal-labels" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction temporal pseudo-label }
    "temporal" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction temporal train }
    "mti-eval" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction mti evaluate }
    "tongzhou" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction tongzhou run }
    "review" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") review serve --open-browser }
    "status" { & $Python (Join-Path $ProjectRoot "tools\urrs.py") construction status }
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
