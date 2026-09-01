param(
    [Parameter(Mandatory = $true)]
    [string]$Seed
)

$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "python311\python.exe"
$runtime = Join-Path $root "slice12_runtime.py"
& $python $runtime --verify --seed $Seed
exit $LASTEXITCODE
