$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "python311\python.exe"
$runtime = Join-Path $root "cross_boundary_attention_relational_invariance.py"
& $python $runtime
exit $LASTEXITCODE
