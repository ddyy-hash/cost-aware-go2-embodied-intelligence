[CmdletBinding()]
param(
    [string]$OutputDirectory = '',
    [hashtable]$DeployParameters = @{}
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $workspaceRoot 'output\robot-release'
}

$archive = Get-ChildItem -LiteralPath $OutputDirectory -Filter 'go2-mapping-robot-*.tar.gz' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $archive) {
    throw "No prepared robot bundle found in: $OutputDirectory"
}

$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive.FullName).Hash.ToLowerInvariant()
$expectedName = "go2-mapping-robot-$archiveHash.tar.gz"
if ($archive.Name -ne $expectedName) {
    throw "Prepared robot bundle name does not match its SHA-256: $($archive.Name)"
}

$deployer = Join-Path $PSScriptRoot 'deploy_go2_mapping_robot.ps1'
if (-not (Test-Path -LiteralPath $deployer -PathType Leaf)) {
    throw "Atomic robot deployer is missing: $deployer"
}

$forwardedParameters = @{}
foreach ($key in $DeployParameters.Keys) {
    if ($key -notin @('PreparedArchive', 'UseTailscaleProxy')) {
        $forwardedParameters[$key] = $DeployParameters[$key]
    }
}

& $deployer -PreparedArchive $archive.FullName -UseTailscaleProxy $true @forwardedParameters
