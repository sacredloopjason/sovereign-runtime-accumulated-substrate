$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project "python311\python.exe"
& $python (Join-Path $project "originating_standing_wave.py")
exit $LASTEXITCODE
