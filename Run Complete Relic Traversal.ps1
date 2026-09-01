$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project "python311\python.exe"
$runtime = Join-Path $project "relic_full_traversal.py"

if (-not (Test-Path -LiteralPath $python)) { throw "Bundled Python is missing: $python" }
if (-not (Test-Path -LiteralPath $runtime)) { throw "Traversal runtime is missing: $runtime" }

& $python $runtime
$exitCode = $LASTEXITCODE
Write-Output "PROCESS_EXIT_CODE=$exitCode"
exit $exitCode
