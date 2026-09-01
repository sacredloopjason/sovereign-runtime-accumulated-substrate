$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "python311\python.exe"
$analysis = Join-Path $root "kv_causal_response_dimensionality.py"
& $python $analysis
exit $LASTEXITCODE
