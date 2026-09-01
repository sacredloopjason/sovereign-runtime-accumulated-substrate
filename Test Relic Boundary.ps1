$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project "python311\python.exe"
$probe = Join-Path $project "relic_boundary_probe.py"

if (-not (Test-Path -LiteralPath $python)) { throw "Bundled Python is missing: $python" }
if (-not (Test-Path -LiteralPath $probe)) { throw "Boundary probe is missing: $probe" }

& $python $probe
$exitCode = $LASTEXITCODE
Write-Output "PROCESS_EXIT_CODE=$exitCode"
exit $exitCode
