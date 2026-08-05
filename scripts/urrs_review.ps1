param([string]$OverlayDir="outputs\integration\urban_renewal_overlay_with_geometry\20260801_060105",[int]$Port=8765,[switch]$NoBrowser)
$ErrorActionPreference="Stop"
$Project=Split-Path -Parent $PSScriptRoot
$Python=Join-Path $Project ".venv-gpu\Scripts\python.exe"
$ReviewDir=Join-Path $Project "outputs\urrs\review\tongzhou_20260801"
if(-not (Test-Path $Python)){throw "Project Python not found: $Python"}
& $Python (Join-Path $Project "tools\urrs_review.py") serve --review-dir $ReviewDir --port $Port $(if($NoBrowser){@()}else{"--open-browser"})
