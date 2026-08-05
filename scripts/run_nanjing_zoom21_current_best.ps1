[CmdletBinding()]
param(
    [ValidateSet('Audit','Native','ScaleNormalized','Compare','BuildReport','All','Status')]
    [string]$Mode = 'Audit',
    [string]$SourceRoot = 'D:\Data Set\Satellite\nanjing_jiangbei',
    [string]$Registry = '',
    [string]$RunRoot = ''
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv-gpu\Scripts\python.exe'
$Tool = Join-Path $ProjectRoot 'tools\run_nanjing_zoom21_current_best.py'
$args = @('--project-root', $ProjectRoot, '--source-root', $SourceRoot, '--mode', $Mode)
if ($Registry) { $args += @('--registry', $Registry) }
if ($RunRoot) { $args += @('--run-root', $RunRoot) }
& $Python $Tool @args
exit $LASTEXITCODE
