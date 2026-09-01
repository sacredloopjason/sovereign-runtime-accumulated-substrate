$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "python311\python.exe"
$verifier = Join-Path $root "slice17_existing_evidence_verifier.py"
& $python $verifier
exit $LASTEXITCODE
