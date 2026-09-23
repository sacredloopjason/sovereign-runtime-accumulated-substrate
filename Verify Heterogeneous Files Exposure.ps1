param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDirectory
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentCommit = 'a0939d388c1f8493ec5d5e75d4a95ac831547510'
$python = Join-Path $project 'python311\python.exe'
$verifier = Join-Path $project 'verify_heterogeneous_files_exposure.py'
$textFixture = Join-Path $project 'fixtures\heterogeneous_text_source.txt'
$binaryFixture = Join-Path $project 'fixtures\heterogeneous_binary_source.bin'
$evidenceDirectory = Join-Path $project 'evidence\heterogeneous_files_exposure'
$evidence = Join-Path $evidenceDirectory 'acceptance_evidence.json'
$receipt = Join-Path $evidenceDirectory 'completion_receipt.txt'
$integrityReceipt = Join-Path $evidenceDirectory 'post_execution_integrity.json'

if ((git -C $project rev-parse HEAD) -ne $parentCommit) {
    throw 'The checkout is not at the authoritative parent commit.'
}
foreach ($path in @($python, $verifier, $textFixture, $binaryFixture)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required bounded input is absent: $path"
    }
}
if (-not (Test-Path -LiteralPath $ModelDirectory -PathType Container)) {
    throw 'ModelDirectory must identify the authoritative local model directory.'
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
& $python $verifier --model-dir $ModelDirectory --text-fixture $textFixture --binary-fixture $binaryFixture --evidence-dir $evidenceDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Heterogeneous FILES verifier exited with code $LASTEXITCODE."
}

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
    schema = 'HETEROGENEOUS_FILES_EXPOSURE_POST_EXECUTION_INTEGRITY_V1'
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
Add-Content -LiteralPath $receipt -Value "`nPRIOR_AUTHORITATIVE_FILES_MODIFIED_AFTER_EXECUTION: $($integrity.prior_authoritative_files_modified_after_execution)" -Encoding utf8

Write-Output "PRIOR_AUTHORITATIVE_FILES_MODIFIED_AFTER_EXECUTION: $($integrity.prior_authoritative_files_modified_after_execution)"
Write-Output "POST_EXECUTION_INTEGRITY_RECEIPT: $integrityReceipt"
if ($modified.Count -ne 0) {
    throw "Authoritative parent files changed during execution: $($modified -join ', ')"
}
if ($runtimeEvidence.outcome -ne 'HETEROGENEOUS_FILES_EXPOSURE_OPERATIONAL') {
    throw "Runtime outcome was $($runtimeEvidence.outcome)."
}
