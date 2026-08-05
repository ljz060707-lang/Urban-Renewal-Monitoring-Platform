param([string]$Run = "20260721_232528", [string]$Model = "glm-4.6v-flash", [int]$MaxSamples = 100)
$ErrorActionPreference = "Stop"
if (-not $env:ZAI_API_KEY -and -not $env:GLM_API_KEY) { throw "未设置 ZAI_API_KEY 或 GLM_API_KEY" }
python run.py change-ai-review --run $Run --config configs/change_public_s2looking_windows.yaml --model $Model --max-samples $MaxSamples --resume
