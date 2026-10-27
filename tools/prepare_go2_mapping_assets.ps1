[CmdletBinding()]
param(
    [string]$WeightsPath = '',
    [string]$ModelManifest = '',
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $WeightsPath) { $WeightsPath = Join-Path $workspaceRoot 'models\depth-anything-v2-small-hf-5426e4f' }
if (-not $ModelManifest) { $ModelManifest = Join-Path $workspaceRoot 'mapping\config\pc_model_manifest.json' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $workspaceRoot 'output\assets' }

function Get-CanonicalTreeDigest([string]$Root) {
    $rootPath = (Resolve-Path -LiteralPath $Root).Path
    $files = @(Get-ChildItem -LiteralPath $rootPath -File -Recurse | Sort-Object FullName)
    if ($files.Count -eq 0) { throw "Weights directory is empty: $rootPath" }
    $sha = [System.Security.Cryptography.IncrementalHash]::CreateHash([System.Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($rootPath.Length).TrimStart('\', '/') -replace '\\', '/'
            $nameBytes = [System.Text.Encoding]::UTF8.GetBytes($relative)
            $payload = [System.IO.File]::ReadAllBytes($file.FullName)
            $sha.AppendData([BitConverter]::GetBytes([uint32]$nameBytes.Length))
            $sha.AppendData($nameBytes)
            $sha.AppendData([BitConverter]::GetBytes([uint64]$payload.Length))
            $sha.AppendData($payload)
        }
        return ([BitConverter]::ToString($sha.GetHashAndReset())).Replace('-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
}

$manifest = Get-Content -LiteralPath $ModelManifest -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or $manifest.asset_type -ne 'pc_model_weights') { throw 'Unsupported PC model manifest.' }
$WeightsPath = (Resolve-Path -LiteralPath $WeightsPath).Path
$actualDigest = Get-CanonicalTreeDigest $WeightsPath
if ($actualDigest -cne [string]$manifest.directory_sha256) { throw "Weights digest mismatch. expected=$($manifest.directory_sha256) actual=$actualDigest" }
$actualFiles = @(Get-ChildItem -LiteralPath $WeightsPath -File -Recurse)
if ($actualFiles.Count -ne [int]$manifest.file_count) { throw 'Weights file count does not match the manifest.' }
foreach ($entry in @($manifest.files)) {
    $filePath = Join-Path $WeightsPath ([string]$entry.path)
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) { throw "Weights file missing: $($entry.path)" }
    $fileHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
    if ($fileHash -cne [string]$entry.sha256) { throw "Weights file digest mismatch: $($entry.path)" }
    if ((Get-Item -LiteralPath $filePath).Length -ne [int64]$entry.bytes) { throw "Weights file size mismatch: $($entry.path)" }
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$verification = [ordered]@{
    schema_version = 1
    verified_at = [DateTime]::UtcNow.ToString('o')
    model_manifest = (Resolve-Path -LiteralPath $ModelManifest).Path
    weights_path = $WeightsPath
    directory_sha256 = $actualDigest
    file_count = $actualFiles.Count
    local_files_only = $true
}
$verificationPath = Join-Path $OutputDirectory 'pc-model-assets.verified.json'
[System.IO.File]::WriteAllText($verificationPath, ($verification | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))
Write-Output "PC_MODEL_ASSETS_READY digest=$actualDigest path=$WeightsPath verification=$verificationPath"
