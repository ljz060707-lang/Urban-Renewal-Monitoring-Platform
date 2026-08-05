param([string]$Run = "20260721_232528")
$ErrorActionPreference = "Stop"
python run.py change-retrain --run $Run --config configs/change_public_s2looking_hard_finetune.yaml --from-review --device cuda:0
