$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "python311\python.exe"
$runtime = Join-Path $root "attention_relational_response_experiment.py"
& $python $runtime
exit $LASTEXITCODE
