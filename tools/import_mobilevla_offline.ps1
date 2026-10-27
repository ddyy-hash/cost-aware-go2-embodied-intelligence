[CmdletBinding()]
param(
    [string]$SourcePath = '',
    [string]$OutputDirectory = '',
    [string]$ManifestPath = ''
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $SourcePath) { $SourcePath = Join-Path $workspaceRoot 'models\mobilevla-r1' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $workspaceRoot 'output\assets' }
if (-not $ManifestPath) { $ManifestPath = Join-Path $workspaceRoot 'mapping\config\mobilevla_r1_manifest.json' }

function Get-TreeDigest([string]$Root) {
    $rootPath = (Resolve-Path -LiteralPath $Root).Path
    $files = @(Get-ChildItem -LiteralPath $rootPath -File -Recurse | Sort-Object FullName)
    if ($files.Count -eq 0) { throw "MobileVLA source directory is empty: $rootPath" }
    $sha = [System.Security.Cryptography.IncrementalHash]::CreateHash([System.Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($rootPath.Length).TrimStart('\', '/') -replace '\\', '/'
            $nameBytes = [System.Text.Encoding]::UTF8.GetBytes($relative)
            $payload = [System.IO.File]::ReadAllBytes($file.FullName)
            $fileHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
            if ($fileHash.Length -ne 64) { throw "Unable to hash MobileVLA file: $relative" }
            $sha.AppendData([BitConverter]::GetBytes([uint32]$nameBytes.Length))
            $sha.AppendData($nameBytes)
            $sha.AppendData([BitConverter]::GetBytes([uint64]$payload.Length))
            $sha.AppendData($payload)
        }
        return @{ digest = ([BitConverter]::ToString($sha.GetHashAndReset())).Replace('-', '').ToLowerInvariant(); count = $files.Count; bytes = ($files | Measure-Object Length -Sum).Sum }
    } finally { $sha.Dispose() }
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or $manifest.model_id -ne 'MobileVLA-R1' -or $manifest.offline_only -ne $true) {
    throw 'Unsupported MobileVLA manifest.'
}
if ((Test-Path -LiteralPath $SourcePath -PathType Leaf) -and [System.IO.Path]::GetExtension($SourcePath) -eq '.zip') {
    $extractRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('mobilevla-import-' + [Guid]::NewGuid().ToString('N'))
    try {
        Expand-Archive -LiteralPath $SourcePath -DestinationPath $extractRoot -Force
        $SourcePath = $extractRoot
        while (@(Get-ChildItem -LiteralPath $SourcePath -Directory).Count -eq 1 -and @(Get-ChildItem -LiteralPath $SourcePath -File).Count -eq 0) {
            $SourcePath = (Get-ChildItem -LiteralPath $SourcePath -Directory)[0].FullName
        }
        $tree = Get-TreeDigest $SourcePath
        $importedPath = Join-Path $workspaceRoot 'models\mobilevla-r1'
        if (Test-Path -LiteralPath $importedPath) { throw "MobileVLA checkpoint destination already exists: $importedPath" }
        New-Item -ItemType Directory -Force -Path $importedPath | Out-Null
        Copy-Item -LiteralPath (Join-Path $SourcePath '*') -Destination $importedPath -Recurse -Force
        $SourcePath = $importedPath
    } finally { Remove-Item -LiteralPath $extractRoot -Recurse -Force -ErrorAction SilentlyContinue }
} else {
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) { throw "MobileVLA local checkpoint is not staged: $SourcePath" }
    $tree = Get-TreeDigest $SourcePath
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$record = [ordered]@{
    schema_version = 1
    model_id = $manifest.model_id
    upstream_commit = $manifest.upstream_commit
    checkpoint_path = (Resolve-Path -LiteralPath $SourcePath).Path
    directory_sha256 = $tree.digest
    file_count = $tree.count
    total_bytes = $tree.bytes
    local_files_only = $true
    action_boundary = $manifest.action_boundary
    imported_at = [DateTime]::UtcNow.ToString('o')
}
$recordPath = Join-Path $OutputDirectory 'mobilevla-r1-assets.verified.json'
[System.IO.File]::WriteAllText($recordPath, ($record | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
Write-Output "MOBILEVLA_OFFLINE_IMPORT_READY path=$($record.checkpoint_path) digest=$($record.directory_sha256) record=$recordPath"
