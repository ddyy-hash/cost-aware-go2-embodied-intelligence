[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$LocalProxyPort = 7897,

    [ValidateRange(1, 65535)]
    [int]$RemoteControlProxyPort = 17897,

    [string]$AutoDLHost = 'connect.westb.seetacloud.com',

    [ValidateRange(1, 65535)]
    [int]$AutoDLPort = 35722,

    [string]$AutoDLUser = 'root',

    [string]$PrivateKeyFile = "$env:USERPROFILE\.ssh\id_ed25519"
)

$ErrorActionPreference = 'Stop'
$logPath = Join-Path $PSScriptRoot '..\mapping-data\session-logs\autodl-control-tunnel-supervisor.log'
$logDirectory = Split-Path -Parent $logPath
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-Log([string]$Message) {
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $logPath -Encoding utf8
}

function Get-ControlTunnel {
    $needle = "127.0.0.1:${RemoteControlProxyPort}:127.0.0.1:${LocalProxyPort}"
    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match [regex]::Escape($needle) -and
        $_.CommandLine -match [regex]::Escape($AutoDLHost) -and
        $_.CommandLine -match '\s-N(?:\s|$)'
    }
}

if (-not (Test-Path -LiteralPath $PrivateKeyFile -PathType Leaf)) {
    throw "Private key not found: $PrivateKeyFile"
}

Write-Log 'control-tunnel supervisor started'
while ($true) {
    try {
        if (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port $LocalProxyPort -InformationLevel Quiet)) {
            Write-Log "waiting for local HTTP proxy on 127.0.0.1:$LocalProxyPort"
            Start-Sleep -Seconds 15
            continue
        }

        if (-not (Get-ControlTunnel)) {
            $args = @(
                '-i', $PrivateKeyFile,
                '-p', "$AutoDLPort",
                '-o', 'BatchMode=yes',
                '-o', 'ExitOnForwardFailure=yes',
                '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3',
                '-N', '-R', "127.0.0.1:${RemoteControlProxyPort}:127.0.0.1:${LocalProxyPort}",
                "${AutoDLUser}@${AutoDLHost}"
            )
            Start-Process -FilePath ssh.exe -ArgumentList $args -WindowStyle Hidden
            Write-Log "started control tunnel to ${AutoDLHost}:${AutoDLPort}"
            Start-Sleep -Seconds 5
        }
    } catch {
        Write-Log "supervisor error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 20
}
