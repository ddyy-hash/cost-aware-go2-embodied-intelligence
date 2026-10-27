[CmdletBinding(DefaultParameterSetName = 'Measured')]
param(
    [Parameter(ParameterSetName = 'Measured')]
    [switch]$Measured,
    [Parameter(Mandatory, ParameterSetName = 'TexturedTile')]
    [string]$TileRevisionId,
    [Parameter(Mandatory, ParameterSetName = 'TexturedTile')]
    [string]$AppearanceRevisionId,
    [Parameter(Mandatory, ParameterSetName = 'TexturedTile')]
    [string]$LodId,
    [Parameter(ParameterSetName = 'TexturedTile')]
    [string]$RgbdAugmentedRevisionId,
    [Parameter(Mandatory, ParameterSetName = 'TexturedBuilding')]
    [string]$AssemblyId,
    [Parameter(Mandatory, ParameterSetName = 'TexturedBuilding')]
    [string]$BuildingAppearanceRevisionId,
    [Parameter(Mandatory, ParameterSetName = 'TexturedBuilding')]
    [string]$LodByMemberJson,
    [string]$DashboardUrl = 'http://127.0.0.1:8765',
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\output\exports')
)

$ErrorActionPreference = 'Stop'
$dashboardBase = [Uri]::new($DashboardUrl.TrimEnd('/') + '/')

if ($PSCmdlet.ParameterSetName -ne 'Measured') {
    $isTile = $PSCmdlet.ParameterSetName -eq 'TexturedTile'
    $request = if ($isTile) {
        [ordered]@{
            tile_revision_id = $TileRevisionId
            appearance_revision_id = $AppearanceRevisionId
            lod_id = $LodId
            rgbd_augmented_revision_id = if ($RgbdAugmentedRevisionId) { $RgbdAugmentedRevisionId } else { $null }
        }
    } else {
        [ordered]@{
            assembly_id = $AssemblyId
            building_appearance_revision_id = $BuildingAppearanceRevisionId
            lod_by_member = $LodByMemberJson | ConvertFrom-Json -AsHashtable
        }
    }
    $endpoint = if ($isTile) {
        '/api/reconstruction/export-textured-tile'
    } else {
        '/api/reconstruction/export-textured-building'
    }
    $published = Invoke-RestMethod -Method Post -Uri ([Uri]::new($dashboardBase, $endpoint)) `
        -ContentType 'application/json' -Body ($request | ConvertTo-Json -Depth 8 -Compress) -TimeoutSec 300
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
    $glbTarget = Join-Path $resolvedOutput $published.glb_name
    $provenanceTarget = Join-Path $resolvedOutput $published.provenance_name
    Invoke-WebRequest -Uri ([Uri]::new($dashboardBase, $published.glb_url)) -OutFile $glbTarget -TimeoutSec 300
    Invoke-WebRequest -Uri ([Uri]::new($dashboardBase, $published.provenance_url)) -OutFile $provenanceTarget -TimeoutSec 60
    $glbSha = (Get-FileHash -LiteralPath $glbTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    $provenanceSha = (Get-FileHash -LiteralPath $provenanceTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($glbSha -ne $published.glb_sha256 -or $provenanceSha -ne $published.provenance_sha256) {
        throw 'Downloaded textured export checksum mismatch.'
    }
    [ordered]@{
        artifact_kind = $published.artifact_kind
        glb_path = $glbTarget
        provenance_path = $provenanceTarget
        glb_sha256 = $glbSha
        provenance_sha256 = $provenanceSha
    } | ConvertTo-Json -Depth 5
    return
}

$publishUri = [Uri]::new($dashboardBase, '/api/reconstruction/publish-measured')
$published = Invoke-RestMethod -Method Post -Uri $publishUri -TimeoutSec 180

if ($published.artifact_kind -ne 'MEASURED_SURFACE_PREVIEW') {
    throw "Unexpected reconstruction artifact kind: $($published.artifact_kind)"
}
if ($published.ply_sha256 -notmatch '^[0-9a-f]{64}$') {
    throw 'Published reconstruction response has no valid PLY SHA-256.'
}
foreach ($artifactName in @($published.ply_name, $published.metadata_name)) {
    if ($artifactName -notmatch '^[a-z0-9][a-z0-9_-]*\.(?:ply|json)$') {
        throw "Published reconstruction returned an unsafe artifact name: $artifactName"
    }
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$plyTarget = Join-Path $resolvedOutput $published.ply_name
$metadataTarget = Join-Path $resolvedOutput $published.metadata_name
$nonce = [Guid]::NewGuid().ToString('N')
$plyPartial = Join-Path $resolvedOutput ".$($published.ply_name).$nonce.partial"
$metadataPartial = Join-Path $resolvedOutput ".$($published.metadata_name).$nonce.partial"

function Commit-ContentAddressedFile {
    param(
        [Parameter(Mandatory)] [string]$PartialPath,
        [Parameter(Mandatory)] [string]$TargetPath,
        [Parameter(Mandatory)] [string]$ExpectedSha256
    )
    $actual = (Get-FileHash -LiteralPath $PartialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256) {
        throw "Downloaded artifact checksum mismatch: $([System.IO.Path]::GetFileName($TargetPath))"
    }
    if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
        $existing = (Get-FileHash -LiteralPath $TargetPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -ne $ExpectedSha256) {
            throw "Content-addressed target conflicts with different data: $TargetPath"
        }
        Remove-Item -LiteralPath $PartialPath -Force
        return
    }
    Move-Item -LiteralPath $PartialPath -Destination $TargetPath
}

try {
    Invoke-WebRequest -Uri ([Uri]::new($dashboardBase, $published.ply_url)) -OutFile $plyPartial -TimeoutSec 180
    Invoke-WebRequest -Uri ([Uri]::new($dashboardBase, $published.metadata_url)) -OutFile $metadataPartial -TimeoutSec 30

    $metadata = Get-Content -LiteralPath $metadataPartial -Raw | ConvertFrom-Json
    if ($metadata.ply_sha256 -ne $published.ply_sha256) {
        throw 'Downloaded metadata does not identify the published PLY.'
    }
    $metadataSha256 = (Get-FileHash -LiteralPath $metadataPartial -Algorithm SHA256).Hash.ToLowerInvariant()
    Commit-ContentAddressedFile -PartialPath $plyPartial -TargetPath $plyTarget -ExpectedSha256 $published.ply_sha256
    Commit-ContentAddressedFile -PartialPath $metadataPartial -TargetPath $metadataTarget -ExpectedSha256 $metadataSha256
} finally {
    foreach ($partial in @($plyPartial, $metadataPartial)) {
        if (Test-Path -LiteralPath $partial -PathType Leaf) {
            Remove-Item -LiteralPath $partial -Force
        }
    }
}

[ordered]@{
    artifact_kind = $published.artifact_kind
    quality_label = $published.quality_label
    delivery_authorized = [bool]$published.delivery_authorized
    quality_blockers = @($published.quality_blockers)
    ply_path = $plyTarget
    metadata_path = $metadataTarget
    ply_sha256 = $published.ply_sha256
    derived_cache_key = $published.derived_cache_key
    vertex_count = $published.vertex_count
    triangle_count = $published.triangle_count
} | ConvertTo-Json -Depth 5
