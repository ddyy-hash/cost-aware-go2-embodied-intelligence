[CmdletBinding()]
param(
    [ValidateRange(1, 60)]
    [int]$TtlMinutes = 15,
    [string]$OutputFile = (Join-Path (Join-Path $PSScriptRoot '..') 'mapping-data\secrets\go2-room-scan-operator-token.json')
)

$ErrorActionPreference = 'Stop'
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputFile)
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$secretsRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot 'mapping-data\secrets'))
if (-not $resolvedOutput.StartsWith($secretsRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Room scan operator token must stay below mapping-data\secrets.'
}

[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($resolvedOutput)) | Out-Null
$bytes = New-Object byte[] 32
$random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $random.GetBytes($bytes)
} finally {
    $random.Dispose()
}
$token = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
$issued = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$payload = [ordered]@{
    schema_version = 1
    token = $token
    issued_at_unix = $issued
    expires_at_unix = $issued + ($TtlMinutes * 60)
}
$json = $payload | ConvertTo-Json -Compress
[System.IO.File]::WriteAllText($resolvedOutput, $json + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
Write-Output "ROOM_SCAN_OPERATOR_TOKEN_FILE=$resolvedOutput"
Write-Output "ROOM_SCAN_OPERATOR_TOKEN_EXPIRES_AT=$($payload.expires_at_unix)"
Write-Output "ROOM_SCAN_OPERATOR_TOKEN=$token"
