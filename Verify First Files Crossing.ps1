param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDirectory
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentCommit = 'b3b80827a9dae75230181256f8f6df5baba2640d'
$python = Join-Path $project 'python311\python.exe'
$verifier = Join-Path $project 'verify_first_files_crossing.py'
$fixture = Join-Path $project 'fixtures\first_files_source.txt'
$evidenceDirectory = Join-Path $project 'evidence\first_files_crossing'
$evidence = Join-Path $evidenceDirectory 'acceptance_evidence.json'
$integrityReceipt = Join-Path $evidenceDirectory 'post_execution_integrity.json'

if ((git -C $project rev-parse HEAD) -ne $parentCommit) {
    throw 'The checkout is not at the authoritative parent commit.'
}
if (-not (Test-Path -LiteralPath $ModelDirectory -PathType Container)) {
    throw 'ModelDirectory must identify the authoritative local model directory.'
}
if (-not (Test-Path -LiteralPath $fixture -PathType Leaf)) {
    throw 'The one bounded fixture is absent.'
}

$parentPaths = @(git -C $project ls-tree -r --name-only $parentCommit)
if ($LASTEXITCODE -ne 0 -or $parentPaths.Count -eq 0) {
    throw 'Could not enumerate the authoritative parent files.'
}

function Measure-ParentFiles {
    $result = [ordered]@{}
    foreach ($relativePath in $parentPaths) {
        $path = Join-Path $project $relativePath
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Authoritative parent file is absent: $relativePath"
        }
        $result[$relativePath] = [ordered]@{
            bytes = (Get-Item -LiteralPath $path).Length
            sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    return $result
}

$before = Measure-ParentFiles
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONHASHSEED = '0'
& $python $verifier --model-dir $ModelDirectory --fixture $fixture --evidence-dir $evidenceDirectory
if ($LASTEXITCODE -ne 0) {
    throw "First FILES crossing verifier exited with code $LASTEXITCODE."
}

# This comparison intentionally occurs after every Python/model action.
$after = Measure-ParentFiles
$modified = @(
    foreach ($relativePath in $parentPaths) {
        if (
            $before[$relativePath].bytes -ne $after[$relativePath].bytes -or
            $before[$relativePath].sha256 -ne $after[$relativePath].sha256
        ) {
            $relativePath
        }
    }
)
$runtimeEvidence = Get-Content -LiteralPath $evidence -Raw | ConvertFrom-Json
$integrity = [ordered]@{
    schema = 'FIRST_FILES_MODULE_CROSSING_POST_EXECUTION_INTEGRITY_V1'
    authoritative_parent_commit = $parentCommit
    checked_after_all_python_model_execution = $true
    authoritative_parent_file_count = $parentPaths.Count
    modified_authoritative_files = $modified
    prior_authoritative_files_modified_after_execution = if ($modified.Count -eq 0) { 'NO' } else { 'YES' }
    runtime_outcome = $runtimeEvidence.outcome
}
[System.IO.File]::WriteAllText(
    $integrityReceipt,
    (($integrity | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)

Write-Output "PRIOR_AUTHORITATIVE_FILES_MODIFIED_AFTER_EXECUTION: $($integrity.prior_authoritative_files_modified_after_execution)"
Write-Output "POST_EXECUTION_INTEGRITY_RECEIPT: $integrityReceipt"
if ($modified.Count -ne 0) {
    throw "Authoritative parent files changed during execution: $($modified -join ', ')"
}
if ($runtimeEvidence.outcome -ne 'FIRST_FILES_MODULE_CROSSING_OPERATIONAL') {
    throw "Runtime outcome was $($runtimeEvidence.outcome)."
}
