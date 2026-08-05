param([ValidateSet("audit","smoke","temporal-smoke")][string]$Stage="audit")
$ErrorActionPreference="Stop"; $Proj=Split-Path -Parent $PSScriptRoot; $Py=Join-Path $Proj ".venv-gpu\Scripts\python.exe"; $Tool=Join-Path $Proj "tools\surface_state_v1_cli.py"
if($Stage -eq "audit"){ & $Py $Tool cwld-audit --manifest "$Proj\outputs\data_audit\cwld\CWLD_MANIFEST.csv" --output "$Proj\outputs\data_audit\cwld"; & $Py $Tool fusu-classify --manifest "$Proj\outputs\data_audit\fusu\FUSU_MANIFEST.csv" --output "$Proj\outputs\data_audit\fusu" }
if($Stage -eq "smoke"){ $out=Join-Path $Proj ("outputs\training\surface_state_v1_smoke_"+(Get-Date -Format yyyyMMdd_HHmmss)); & $Py $Tool cwld-smoke --config "$Proj\extensions\surface_state_v1\configs\cwld_smoke.yaml" --output $out }
if($Stage -eq "temporal-smoke"){ $out=Join-Path $Proj ("outputs\training\temporal_surface_state_v1_smoke_"+(Get-Date -Format yyyyMMdd_HHmmss)); & $Py $Tool temporal-smoke --output $out }
