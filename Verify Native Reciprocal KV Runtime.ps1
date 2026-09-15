$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentCommit = '106a79b148e939444df4c891c64f7c35a89699f2'
$runtime = Join-Path $project 'native_reciprocal_kv_runtime.py'
$python = Join-Path $project 'python311\python.exe'
$evidenceDirectory = Join-Path $project 'evidence\native_reciprocal_kv_runtime'
$evidence = Join-Path $evidenceDirectory 'acceptance_evidence.json'
$integrityReceipt = Join-Path $evidenceDirectory 'post_execution_integrity.json'

if ((git -C $project rev-parse HEAD) -ne $parentCommit) {
    throw 'The checkout is not at the authoritative parent commit.'
}

$modelDirectory = $env:SOVEREIGN_NATIVE_MODEL_DIR
if ([string]::IsNullOrWhiteSpace($modelDirectory)) {
    throw 'SOVEREIGN_NATIVE_MODEL_DIR must identify the authoritative local Slice 1 model directory.'
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
& $python $runtime --model-dir $modelDirectory --evidence-dir $evidenceDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Native reciprocal runtime verifier exited with code $LASTEXITCODE."
}

# This comparison intentionally occurs only after every Python/model action is complete.
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
    schema = 'NATIVE_RECIPROCAL_POST_EXECUTION_INTEGRITY_V1'
    authoritative_parent_commit = $parentCommit
    checked_after_all_python_model_execution = $true
    authoritative_parent_file_count = $parentPaths.Count
    modified_authoritative_files = $modified
    prior_authoritative_files_modified_after_execution = if ($modified.Count -eq 0) { 'NO' } else { 'YES' }
    runtime_outcome = $runtimeEvidence.outcome
}
New-Item -ItemType Directory -Path $evidenceDirectory -Force | Out-Null
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
if ($runtimeEvidence.outcome -ne 'NATIVE_RECIPROCAL_KV_RUNTIME_OPERATIONAL') {
    throw "Runtime outcome was $($runtimeEvidence.outcome)."
}
