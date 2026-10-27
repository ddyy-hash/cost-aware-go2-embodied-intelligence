[CmdletBinding()]
param(
    [ValidateSet('start', 'status', 'stop', 'rollback')]
    [string]$Action = 'start',
    [string]$AutoDLHost = 'connect.westb.seetacloud.com',
    [ValidateRange(1, 65535)]
    [int]$AutoDLPort = 35722,
    [string]$AutoDLUser = 'root',
    [string]$PrivateKeyFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [ValidatePattern('^[a-z0-9.-]*$')]
    [string]$PublicDerpHost = '',
    [ValidatePattern('^u[0-9]+$')]
    [string]$AutoDLPublicUser = 'u692021',
    [ValidatePattern('^[a-z0-9.-]+$')]
    [string]$AutoDLPublicDomain = 'westb.seetacloud.com',
    [ValidateRange(1, 65535)]
    [int]$PublicDerpPort = 8443,
    [ValidateRange(1, 9999)]
    [int]$PreferredDerpRegion = 0,
    [ValidateRange(1, 65535)]
    [int]$LocalProxyPort = 7897,
    [ValidateRange(1, 65535)]
    [int]$RemoteControlProxyPort = 17897,
    [string]$BulkTokenFile,
    [switch]$StopControlTunnel
)

$ErrorActionPreference = 'Stop'
foreach ($command in @('ssh.exe', 'scp.exe', 'tailscale.exe', 'curl.exe')) {
    Get-Command $command -ErrorAction Stop | Out-Null
}
if (-not (Test-Path -LiteralPath $PrivateKeyFile -PathType Leaf)) {
    throw "Private key not found: $PrivateKeyFile"
}

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BulkTokenFile = if ($BulkTokenFile) { $BulkTokenFile } else { Join-Path $workspaceRoot 'mapping-data\secrets\autodl-bulk-token' }
$remoteManagerSource = Join-Path $PSScriptRoot 'manage_autodl_derp.sh'
$gatewaySource = Join-Path $PSScriptRoot 'autodl_derp_bulk_gateway.py'
$receiverSource = Join-Path $workspaceRoot 'mapping\pc\ssh_tunnel_receiver.py'
$lifecycleSource = Join-Path $workspaceRoot 'mapping\pc\artifact_lifecycle.py'
$logRoot = Join-Path $workspaceRoot 'mapping-data\session-logs'
$derpPolicyRoot = Join-Path $workspaceRoot 'mapping-data\network'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$sshBase = @('-i', $PrivateKeyFile, '-p', "$AutoDLPort", '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12')
$remote = "${AutoDLUser}@${AutoDLHost}"
$tunnelNeedle = "127.0.0.1:${RemoteControlProxyPort}:127.0.0.1:${LocalProxyPort}"
$script:tailnetMapMatches = $false

function Get-ControlTunnelProcess {
    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match [regex]::Escape($tunnelNeedle) -and
        $_.CommandLine -match [regex]::Escape($AutoDLHost) -and
        $_.CommandLine -match "\s-N(?:\s|$)"
    }
}

function Stop-ControlTunnel {
    Get-ControlTunnelProcess | ForEach-Object {
        Stop-Process -Id $_.ProcessId -ErrorAction Stop
        Write-Output "Stopped AutoDL control proxy tunnel PID $($_.ProcessId)"
    }
}

function Start-ControlTunnel {
    if (Get-ControlTunnelProcess) { return }
    if (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port $LocalProxyPort -InformationLevel Quiet)) {
        throw "Local HTTP proxy is not listening on 127.0.0.1:$LocalProxyPort"
    }
    $stdout = Join-Path $logRoot 'autodl-control-tunnel.log'
    $stderr = Join-Path $logRoot 'autodl-control-tunnel.err.log'
    $args = @(
        '-i', $PrivateKeyFile,
        '-p', "$AutoDLPort",
        '-o', 'BatchMode=yes',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=3',
        '-N', '-R', $tunnelNeedle, $remote
    )
    $process = Start-Process -FilePath ssh.exe -ArgumentList $args -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Start-Sleep -Milliseconds 900
    if ($process.HasExited) {
        throw "AutoDL control proxy tunnel exited early; see $stderr"
    }
    Write-Output "Started AutoDL control proxy tunnel PID $($process.Id)"
}

function Invoke-RemoteManager([string]$RemoteAction) {
    & ssh.exe @sshBase $remote "/root/autodl-tmp/go2-derp/tools/manage_autodl_derp.sh $RemoteAction"
    if ($LASTEXITCODE -ne 0) { throw "Remote DERP action failed: $RemoteAction" }
}

function Get-RemoteContainerHostname {
    $hostnameOutput = & ssh.exe @sshBase $remote 'hostname'
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not read the current AutoDL container hostname'
    }
    $containerHostname = @(
        $hostnameOutput | ForEach-Object { "$_".Trim() } | Where-Object { $_ }
    ) | Select-Object -First 1
    if (-not $containerHostname) {
        throw 'AutoDL returned an empty container hostname'
    }
    return $containerHostname
}

function Resolve-PublicDerpEndpoint {
    if ($PublicDerpHost) {
        return [pscustomobject]@{
            HostName = $PublicDerpHost
            Port = $PublicDerpPort
            Url = "https://${PublicDerpHost}:${PublicDerpPort}"
            Source = 'explicit'
            ContainerHostname = ''
        }
    }

    $containerHostname = Get-RemoteContainerHostname
    $containerMatch = [regex]::Match(
        $containerHostname,
        '^autodl-container-([a-z0-9]+)-([a-z0-9]+)$'
    )
    if (-not $containerMatch.Success) {
        throw "Unsupported AutoDL container hostname format: $containerHostname. Use -PublicDerpHost to override discovery."
    }
    $compactContainerId = "$($containerMatch.Groups[1].Value)$($containerMatch.Groups[2].Value)"
    if ($compactContainerId.Length -lt 12) {
        throw "AutoDL container identifier is too short for public mapping discovery: $containerHostname"
    }
    $mappingId = $compactContainerId.Substring($compactContainerId.Length - 12).ToLowerInvariant()
    $mappingLabel = "$($mappingId.Substring(0, 4))-$($mappingId.Substring(4))"
    $hostName = "$AutoDLPublicUser-$mappingLabel.$AutoDLPublicDomain"
    return [pscustomobject]@{
        HostName = $hostName
        Port = $PublicDerpPort
        Url = "https://${hostName}:${PublicDerpPort}"
        Source = 'container-hostname'
        ContainerHostname = $containerHostname
    }
}

function Write-AutoDLDerpPolicyFragment([pscustomobject]$Endpoint) {
    New-Item -ItemType Directory -Force -Path $derpPolicyRoot | Out-Null
    $policyPath = Join-Path $derpPolicyRoot 'autodl-derp-901.hujson'
    $policy = [ordered]@{
        derpMap = [ordered]@{
            OmitDefaultRegions = $false
            Regions = [ordered]@{
                '901' = [ordered]@{
                    RegionID = 901
                    RegionCode = 'autodl-westb'
                    RegionName = 'AutoDL WestB'
                    Nodes = @(
                        [ordered]@{
                            Name = '901a'
                            RegionID = 901
                            HostName = $Endpoint.HostName
                            DERPPort = $Endpoint.Port
                            STUNPort = -1
                        },
                        [ordered]@{
                            Name = '901s'
                            RegionID = 901
                            HostName = 'stun.cloudflare.com'
                            DERPPort = -1
                            STUNPort = 3478
                        }
                    )
                }
            }
        }
    } | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($policyPath, $policy + "`n", [System.Text.UTF8Encoding]::new($false))
    return $policyPath
}

function Test-TailnetDerpMap([pscustomobject]$Endpoint) {
    $script:tailnetMapMatches = $false
    $derpMapJson = & tailscale.exe debug derp-map | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not inspect the local Tailscale DERP map'
    }
    try {
        $derpMap = $derpMapJson | ConvertFrom-Json
    } catch {
        throw "Local Tailscale DERP map was not valid JSON: $($_.Exception.Message)"
    }
    $region = $derpMap.Regions.PSObject.Properties['901'].Value
    $matchesCurrentEndpoint = $region -and @(
        $region.Nodes | Where-Object {
            $_.HostName -eq $Endpoint.HostName -and [int]$_.DERPPort -eq $Endpoint.Port
        }
    ).Count -gt 0
    if ($matchesCurrentEndpoint) {
        $script:tailnetMapMatches = $true
        Write-Output "AUTODL_DERP_MAP_MATCH region=901 endpoint=$($Endpoint.Url)"
        return
    }

    $policyPath = Write-AutoDLDerpPolicyFragment $Endpoint
    Write-Warning "Tailnet DERP region 901 does not reference $($Endpoint.Url). Apply the generated policy fragment before forcing region 901."
    Write-Output "AUTODL_DERP_POLICY_FRAGMENT region=901 endpoint=$($Endpoint.Url) path=$policyPath"
}

function Test-PublicDerpUpgrade([pscustomobject]$Endpoint) {
    $url = "$($Endpoint.Url)/derp"
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $response = & curl.exe --silent --show-error --http1.1 --max-time 3 `
            -D - -o NUL -H 'Connection: Upgrade' -H 'Upgrade: DERP' $url 2>&1 | Out-String
        $curlExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($response -notmatch '101 Switching Protocols' -or $response -notmatch 'Derp-Public-Key:') {
        throw "AutoDL public endpoint did not preserve the DERP HTTP upgrade: $url (curl exit $curlExitCode)"
    }
    # Keep DERP upgrades open after curl's bounded probe timeout.
    $global:LASTEXITCODE = 0
    Write-Output "DERP_UPGRADE_OK $url"
}

function Set-PreferredRegion([int]$RegionId) {
    & tailscale.exe debug force-prefer-derp $RegionId
    if ($LASTEXITCODE -ne 0) { throw "Could not set preferred DERP region to $RegionId" }
    & tailscale.exe debug rebind
    if ($LASTEXITCODE -ne 0) { throw 'Could not rebind the local Tailscale transport' }
}

if ($Action -eq 'start') {
    foreach ($source in @($remoteManagerSource, $gatewaySource, $receiverSource, $lifecycleSource)) {
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "AutoDL deployment source is missing: $source"
        }
    }
    if (-not (Test-Path -LiteralPath $BulkTokenFile -PathType Leaf)) {
        $tokenDirectory = Split-Path -Parent $BulkTokenFile
        New-Item -ItemType Directory -Force -Path $tokenDirectory | Out-Null
        $random = New-Object byte[] 32
        $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try { $rng.GetBytes($random) } finally { $rng.Dispose() }
        $token = [Convert]::ToBase64String($random).TrimEnd('=').Replace('+','-').Replace('/','_')
        [System.IO.File]::WriteAllText($BulkTokenFile, $token + "`n", (New-Object System.Text.ASCIIEncoding))
    }
    Start-ControlTunnel
    & ssh.exe @sshBase $remote 'mkdir -p /root/autodl-tmp/go2-derp/tools /root/autodl-tmp/go2-derp/state'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the AutoDL tools directory' }
    & scp.exe -i $PrivateKeyFile -P $AutoDLPort -o BatchMode=yes $remoteManagerSource `
        "${remote}:/root/autodl-tmp/go2-derp/tools/manage_autodl_derp.sh"
    if ($LASTEXITCODE -ne 0) { throw 'Could not upload the AutoDL DERP manager' }
    foreach ($upload in @(
        @($gatewaySource, 'autodl_derp_bulk_gateway.py'),
        @($receiverSource, 'ssh_tunnel_receiver.py'),
        @($lifecycleSource, 'artifact_lifecycle.py')
    )) {
        & scp.exe -i $PrivateKeyFile -P $AutoDLPort -o BatchMode=yes $upload[0] `
            "${remote}:/root/autodl-tmp/go2-derp/tools/$($upload[1])"
        if ($LASTEXITCODE -ne 0) { throw "Could not upload AutoDL component: $($upload[1])" }
    }
    & scp.exe -i $PrivateKeyFile -P $AutoDLPort -o BatchMode=yes $BulkTokenFile `
        "${remote}:/root/autodl-tmp/go2-derp/state/bulk-auth-token"
    if ($LASTEXITCODE -ne 0) { throw 'Could not upload the AutoDL bulk token' }
    & ssh.exe @sshBase $remote 'chmod 0755 /root/autodl-tmp/go2-derp/tools/manage_autodl_derp.sh'
    if ($LASTEXITCODE -ne 0) { throw 'Could not mark the AutoDL DERP manager executable' }
    & ssh.exe @sshBase $remote 'chmod 0600 /root/autodl-tmp/go2-derp/state/bulk-auth-token'
    if ($LASTEXITCODE -ne 0) { throw 'Could not protect the AutoDL bulk token' }
    Invoke-RemoteManager 'start'
    $publicEndpoint = Resolve-PublicDerpEndpoint
    Test-PublicDerpUpgrade $publicEndpoint
    Test-TailnetDerpMap $publicEndpoint
    Write-Output "AUTODL_DERP_PUBLIC_ENDPOINT url=$($publicEndpoint.Url) source=$($publicEndpoint.Source) container=$($publicEndpoint.ContainerHostname)"
    if ($PreferredDerpRegion -gt 0) {
        if (-not $script:tailnetMapMatches) {
            throw 'Refusing to force region 901 while the Tailnet DERP map is stale'
        }
        Set-PreferredRegion $PreferredDerpRegion
    } else {
        Write-Output 'Direct path remains preferred; AutoDL DERP is registered as fallback.'
    }
    Write-Output "AUTODL_DERP_READY region=$PreferredDerpRegion public=$($publicEndpoint.Url)"
} elseif ($Action -eq 'status') {
    Invoke-RemoteManager 'status'
    $publicEndpoint = Resolve-PublicDerpEndpoint
    Test-PublicDerpUpgrade $publicEndpoint
    Test-TailnetDerpMap $publicEndpoint
    Write-Output "AUTODL_DERP_PUBLIC_ENDPOINT url=$($publicEndpoint.Url) source=$($publicEndpoint.Source) container=$($publicEndpoint.ContainerHostname)"
    if ($PreferredDerpRegion -gt 0) {
        & tailscale.exe debug derp $PreferredDerpRegion
    } else {
        Write-Output 'No forced DERP region; automatic direct/DERP selection remains active.'
    }
} elseif ($Action -eq 'stop') {
    Invoke-RemoteManager 'stop'
    Set-PreferredRegion 0
    if ($StopControlTunnel) { Stop-ControlTunnel }
} else {
    Invoke-RemoteManager 'stop-all'
    Set-PreferredRegion 0
    Stop-ControlTunnel
    Write-Output 'AUTODL_DERP_ROLLED_BACK public DERP regions remain enabled in the tailnet policy'
}
