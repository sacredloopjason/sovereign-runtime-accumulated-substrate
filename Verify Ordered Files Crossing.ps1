param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDirectory
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentCommit = '62056f32866ee51c36af3bd185476b17c3f5dcaf'
$python = Join-Path $project 'python311\python.exe'
$verifier = Join-Path $project 'verify_ordered_files_crossing.py'
$fixture1 = Join-Path $project 'fixtures\ordered_source_1.txt'
$fixture2 = Join-Path $project 'fixtures\ordered_source_2.txt'
$evidenceDirectory = Join-Path $project 'evidence\ordered_files_crossing'
$evidence = Join-Path $evidenceDirectory 'acceptance_evidence.json'
$integrityReceipt = Join-Path $evidenceDirectory 'post_execution_integrity.json'

if ((git -C $project rev-parse HEAD) -ne $parentCommit) {
    throw 'The checkout is not at the authoritative parent commit.'
}
if (-not (Test-Path -LiteralPath $ModelDirectory -PathType Container)) {
    throw 'ModelDirectory must identify the authoritative local model directory.'
}
foreach ($fixture in @($fixture1, $fixture2)) {
    if (-not (Test-Path -LiteralPath $fixture -PathType Leaf)) {
        throw "Bounded fixture is absent: $fixture"
    }
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
& $python $verifier --model-dir $ModelDirectory --fixture-1 $fixture1 --fixture-2 $fixture2 --evidence-dir $evidenceDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Ordered FILES crossing verifier exited with code $LASTEXITCODE."
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
    schema = 'ORDERED_MULTI_ARTIFACT_FILES_CROSSING_POST_EXECUTION_INTEGRITY_V1'
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
if ($runtimeEvidence.outcome -ne 'ORDERED_MULTI_ARTIFACT_FILES_CROSSING_OPERATIONAL') {
    throw "Runtime outcome was $($runtimeEvidence.outcome)."
}
