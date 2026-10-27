[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Domain,
    [Parameter(Mandatory)]
    [string]$ServerHost,
    [string]$ServerUser = 'root',
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$PrivateKeyFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [Parameter(Mandatory)]
    [string]$DerperBinary,
    [ValidateRange(900, 9999)]
    [int]$RegionId = 902,
    [ValidatePattern('^[a-z0-9-]+$')]
    [string]$RegionCode = 'private-cn',
    [string]$RegionName = 'Go2 Private China'
)

$ErrorActionPreference = 'Stop'
foreach ($command in @('ssh.exe', 'scp.exe', 'curl.exe')) { Get-Command $command -ErrorAction Stop | Out-Null }
foreach ($file in @($PrivateKeyFile, $DerperBinary)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file not found: $file" }
}
if ($Domain -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]{1,251}[A-Za-z0-9]$') { throw 'Invalid DERP domain.' }

$installer = Join-Path $PSScriptRoot 'install_private_derp.sh'
$remote = "${ServerUser}@${ServerHost}"
$sshArgs = @('-i', $PrivateKeyFile, '-p', "$SshPort", '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12')
$releaseId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$remoteRoot = "/tmp/go2-private-derp-$releaseId"
& ssh.exe @sshArgs $remote "install -d -m 0700 '$remoteRoot'"
if ($LASTEXITCODE -ne 0) { throw 'Could not create the staged DERP directory.' }
& scp.exe -i $PrivateKeyFile -P $SshPort -o BatchMode=yes $installer "${remote}:${remoteRoot}/install_private_derp.sh"
if ($LASTEXITCODE -ne 0) { throw 'Could not upload the DERP installer.' }
& scp.exe -i $PrivateKeyFile -P $SshPort -o BatchMode=yes $DerperBinary "${remote}:${remoteRoot}/derper"
if ($LASTEXITCODE -ne 0) { throw 'Could not upload the DERP binary.' }
& ssh.exe @sshArgs $remote "chmod 0700 '$remoteRoot/install_private_derp.sh' '$remoteRoot/derper' && sudo '$remoteRoot/install_private_derp.sh' '$Domain' '$remoteRoot/derper'"
if ($LASTEXITCODE -ne 0) { throw 'Private DERP installation failed.' }

$response = & curl.exe --silent --show-error --http1.1 --max-time 8 -D - -o NUL `
    -H 'Connection: Upgrade' -H 'Upgrade: DERP' "https://${Domain}/derp" 2>&1 | Out-String
if ($response -notmatch '101 Switching Protocols' -or $response -notmatch 'Derp-Public-Key:') {
    throw "Private DERP TLS upgrade verification failed for $Domain."
}

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputRoot = Join-Path $workspaceRoot 'mapping-data\network'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$policyPath = Join-Path $outputRoot "private-derp-$RegionId.hujson"
$policy = @{
    derpMap = @{
        OmitDefaultRegions = $false
        Regions = @{
            "$RegionId" = @{
                RegionID = $RegionId
                RegionCode = $RegionCode
                RegionName = $RegionName
                Nodes = @(@{ Name = "${RegionId}a"; RegionID = $RegionId; HostName = $Domain; DERPPort = 443; STUNPort = 3478 })
            }
        }
    }
} | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($policyPath, $policy + "`n", [System.Text.UTF8Encoding]::new($false))
Write-Output "PRIVATE_DERP_DEPLOYED domain=$Domain region=$RegionId policy=$policyPath"
Write-Output 'Apply the generated derpMap in the Tailscale admin policy; official regions remain enabled.'
Write-Output 'Do not force this region until both PC and Go2 netcheck measurements show a material improvement.'
