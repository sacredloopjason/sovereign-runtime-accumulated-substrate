$ErrorActionPreference = "Stop"
$runtimeDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $runtimeDirectory "python311\python.exe"
$runtime = Join-Path $runtimeDirectory "slice4_runtime.py"

& $python $runtime --interactive
exit $LASTEXITCODE
