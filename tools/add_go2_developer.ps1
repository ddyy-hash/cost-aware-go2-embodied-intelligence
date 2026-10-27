[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string[]]$Node,

    [Parameter(Mandatory)]
    [ValidatePattern('^[a-z_][a-z0-9_-]{0,30}$')]
    [string]$UserName,

    [Parameter(Mandatory)]
    [string]$PublicKeyFile,

    [string]$AdminUser = 'unitree'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PublicKeyFile)) {
    throw "Public key not found: $PublicKeyFile"
}

foreach ($robot in $Node) {
    $target = "${AdminUser}@${robot}"
    $remoteKey = "/tmp/${UserName}.pub"
    Write-Host "Adding $UserName to $robot..."
    & scp $PublicKeyFile "${target}:${remoteKey}"
    if ($LASTEXITCODE -ne 0) { throw "Key upload failed for $robot" }
    & ssh -t $target "sudo /usr/local/sbin/go2-developer add '$UserName' '$remoteKey'; rm -f '$remoteKey'"
    if ($LASTEXITCODE -ne 0) { throw "Developer setup failed for $robot" }
}

