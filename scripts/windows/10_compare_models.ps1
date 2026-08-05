param([string]$BaselineRun = "20260721_232528", [string]$CandidateRun = "latest")
$ErrorActionPreference = "Stop"
python run.py change-compare --baseline-run $BaselineRun --candidate-run $CandidateRun --config configs/change_public_s2looking_hard_finetune.yaml --split validation --device cuda:0
