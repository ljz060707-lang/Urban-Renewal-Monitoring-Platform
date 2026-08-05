param(
    [ValidateSet("audit", "pair", "smoke", "validate", "package", "qa", "diagnostics", "finalize")]
    [string]$Stage = "smoke",
    [string]$RunDir = "",
    [string]$Config = "",
    [string]$GlobfpRoot = "D:\Data Set\3D-GloBFP",
    [string]$WhuRoot = "D:\Data Set\WHU_Building"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Project Python not found: $Python" }
if (-not $Config) { $Config = Join-Path $ProjectRoot "configs\building_foundation_v2_smoke.yaml" }
switch ($Stage) {
    "audit" { & $Python "$ProjectRoot\tools\audit_globfp_dataset.py" --root $GlobfpRoot --output "$ProjectRoot\outputs\pretraining\building_foundation_v2_audit" --no-hash }
    "pair" { if (-not $RunDir) { throw "-RunDir is required for pair" }; & $Python "$ProjectRoot\tools\pair_globfp_with_local_imagery.py" --globfp-root $GlobfpRoot --audit-root "$ProjectRoot\outputs\pretraining\building_foundation_v2_audit\01_globfp_audit" --imagery-root "D:\Data Set\Satellite\nanjing_jiangbei 20" --imagery-root "D:\Data Set\Satellite\beijing_tongzhou" --output "$RunDir\02_pairing" }
    "smoke" { & $Python "$ProjectRoot\tools\train_building_foundation_v2.py" --config $Config --data-root $WhuRoot --stage smoke --device cuda }
    "validate" { if (-not $RunDir) { throw "-RunDir is required for validate" }; & $Python "$ProjectRoot\tools\validate_building_foundation_v2.py" --run-dir $RunDir --device cuda }
    "package" { if (-not $RunDir) { throw "-RunDir is required for package" }; & $Python "$ProjectRoot\tools\package_building_foundation_v2.py" --run-dir $RunDir --baseline-checkpoint "$ProjectRoot\outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth" }
    "qa" { if (-not $RunDir) { throw "-RunDir is required for qa" }; & $Python "$ProjectRoot\tools\create_building_v2_qa_queue.py" --run-dir $RunDir }
    "diagnostics" { if (-not $RunDir) { throw "-RunDir is required for diagnostics" }; & $Python "$ProjectRoot\tools\build_building_v2_diagnostics.py" --run-dir $RunDir }
    "finalize" { if (-not $RunDir) { throw "-RunDir is required for finalize" }; & $Python "$ProjectRoot\tools\finalize_building_v2_run.py" --run-dir $RunDir }
}
if ($LASTEXITCODE -ne 0) { throw "Building Foundation v2 stage failed: $Stage" }
