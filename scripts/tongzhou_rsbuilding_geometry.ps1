param(
    [ValidateSet(
        "diagnose",
        "smoke",
        "infer",
        "audit",
        "vectorize",
        "alignment",
        "overlay",
        "review",
        "status",
        "all"
    )]
    [string]$Stage = "status"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"
$Tool = Join-Path $Project "tools\infer_rsbuilding_static_geometry.py"
$GeometryRoot = Join-Path $Project "outputs\inference\beijing_tongzhou_building_geometry"
$ExpectedHash = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
$Checkpoint = Join-Path $Project "outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "RSBuilding Python missing: $Python"
}
if (-not (Test-Path -LiteralPath $Tool)) {
    throw "Geometry tool missing: $Tool"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Frozen checkpoint missing: $Checkpoint"
}
$ActualHash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $ExpectedHash) {
    throw "Frozen checkpoint SHA256 mismatch: $ActualHash"
}

New-Item -ItemType Directory -Force -Path $GeometryRoot | Out-Null
$Log = Join-Path $GeometryRoot ("pipeline_{0}_{1}.log" -f $Stage, (Get-Date -Format "yyyyMMdd_HHmmss"))

function Invoke-Geometry {
    param([string]$Command)
    "[$(Get-Date -Format o)] $Command" | Tee-Object -FilePath $Log -Append
    & $Python $Tool $Command 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Geometry stage failed: $Command (exit $LASTEXITCODE)"
    }
}

function Require-JsonStatus {
    param(
        [string]$Path,
        [string]$Property,
        [string]$Expected
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required receipt missing: $Path"
    }
    $Receipt = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($Receipt.$Property -ne $Expected) {
        throw "Receipt gate failed: $Path -> $Property=$($Receipt.$Property)"
    }
}

switch ($Stage) {
    "diagnose" {
        Invoke-Geometry "discover"
        Invoke-Geometry "diagnose"
    }
    "smoke" {
        Invoke-Geometry "smoke"
    }
    "infer" {
        $Smoke = Get-ChildItem -LiteralPath $GeometryRoot -Recurse -Filter "SMOKE_RECEIPT.json" |
            Sort-Object LastWriteTime | Select-Object -Last 1
        if (-not $Smoke) {
            throw "Passing 2023 smoke receipt is required before infer."
        }
        Require-JsonStatus $Smoke.FullName "status" "pass"
        Invoke-Geometry "infer-all"
    }
    "audit" {
        Invoke-Geometry "audit"
    }
    "vectorize" {
        Require-JsonStatus (Join-Path $GeometryRoot "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json") "geometry_numerical_status" "pass"
        Invoke-Geometry "vectorize"
    }
    "alignment" {
        Invoke-Geometry "alignment"
    }
    "overlay" {
        & (Join-Path $Project ".venv-gpu\Scripts\python.exe") `
            (Join-Path $Project "tools\surface_state_v1_geometry_overlay_cli.py") run
        if ($LASTEXITCODE -ne 0) {
            throw "Final geometry overlay failed."
        }
    }
    "review" {
        & (Join-Path $Project ".venv-gpu\Scripts\python.exe") `
            (Join-Path $Project "tools\surface_state_v1_geometry_overlay_cli.py") review
        if ($LASTEXITCODE -ne 0) {
            throw "Final review packet failed."
        }
    }
    "status" {
        Invoke-Geometry "status"
    }
    "all" {
        Invoke-Geometry "discover"
        Invoke-Geometry "diagnose"
        Invoke-Geometry "smoke"
        Invoke-Geometry "infer-all"
        Invoke-Geometry "audit"
        Invoke-Geometry "vectorize"
        Invoke-Geometry "alignment"
        & (Join-Path $Project ".venv-gpu\Scripts\python.exe") `
            (Join-Path $Project "tools\surface_state_v1_geometry_overlay_cli.py") all
        if ($LASTEXITCODE -ne 0) {
            throw "Final geometry overlay/review failed."
        }
    }
}

$FinalHash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($FinalHash -ne $ExpectedHash) {
    throw "Frozen checkpoint changed during pipeline: $FinalHash"
}
