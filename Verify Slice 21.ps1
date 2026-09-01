$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project "python311\python.exe"
$runtime = Join-Path $project "slice21_runtime.py"

if (-not (Test-Path -LiteralPath $python)) { throw "Bundled Python is missing: $python" }
if (-not (Test-Path -LiteralPath $runtime)) { throw "Slice 21 runtime is missing: $runtime" }

& $python $runtime --verify
if ($LASTEXITCODE -ne 0) { throw "Slice 21 verification failed with exit code $LASTEXITCODE" }
