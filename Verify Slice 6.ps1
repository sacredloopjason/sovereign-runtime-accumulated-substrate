$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$runtimeDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $runtimeDirectory "python311\python.exe"
$runtime = Join-Path $runtimeDirectory "slice6_runtime.py"

& $python $runtime --verify
exit $LASTEXITCODE
