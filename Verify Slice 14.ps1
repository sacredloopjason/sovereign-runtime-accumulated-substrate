$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "python311\python.exe"
$runtime = Join-Path $root "slice14_runtime.py"
& $python $runtime --verify --seed "SLICE6_SEED"
exit $LASTEXITCODE
