param(
    [string]$ProjectRoot = "D:\Codex\projects\Documents-Codex\2026-07-18\urban-renewal-phase1-starter-v0-2\outputs\urban-renewal-phase1-starter",
    [string]$UcsRoot = "D:\Data Set\URS",
    [int]$SampleSize = 32,
    [string]$RgbMapping = "",
    [ValidateSet("", "confirmed_training_allowed", "confirmed_research_only", "confirmed_training_prohibited", "missing_license_information", "ambiguous_requires_human_confirmation")]
    [string]$LicenseDecision = ""
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "tools\audit_ucs_readiness.py"

if (-not (Test-Path $Python)) { throw "Project Python not found: $Python" }
if (-not (Test-Path $Script)) { throw "Audit script not found: $Script" }

$Args = @(
    $Script,
    "--project-root", $ProjectRoot,
    "--ucs-root", $UcsRoot,
    "--output-root", "outputs\urrs\audits",
    "--seed-root", "outputs\urrs\construction",
    "--sample-size", "$SampleSize"
)
if ($RgbMapping) { $Args += @("--rgb-mapping", $RgbMapping) }
if ($LicenseDecision) { $Args += @("--license-decision", $LicenseDecision) }

& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "Audit failed with exit code $LASTEXITCODE" }
