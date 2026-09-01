$ErrorActionPreference = "Stop"
$runtimeDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $runtimeDirectory "python311\python.exe"
$runtime = Join-Path $runtimeDirectory "slice5_runtime.py"

& $python $runtime --interactive
exit $LASTEXITCODE
