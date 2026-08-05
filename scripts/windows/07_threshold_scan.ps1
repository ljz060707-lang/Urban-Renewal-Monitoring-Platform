param([string]$Run = "20260721_232528")
$ErrorActionPreference = "Stop"
python run.py change-threshold-scan --run $Run --config configs/change_public_s2looking_windows.yaml --split validation --device cuda:0
