[CmdletBinding()]
param(
    [ValidateSet('apply', 'install', 'monitor', 'status', 'remove')]
    [string]$Action = 'apply',

    [ValidateRange(900, 9999)]
    [int]$RegionId = 901,

    [ValidateRange(10, 300)]
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$taskName = 'GO2 AutoDL DERP Preference'
$tailscaleExe = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
$logDirectory = Join-Path $PSScriptRoot '..\mapping-data\session-logs'
$logPath = Join-Path $logDirectory 'autodl-derp-preference.log'

function Write-PreferenceLog([string]$Message) {
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $logPath -Encoding utf8
}

function Set-AutoDlDerpPreference {
    if (-not (Test-Path -LiteralPath $tailscaleExe -PathType Leaf)) {
        throw "Tailscale CLI not found: $tailscaleExe"
    }

    $ready = $false
    foreach ($attempt in 1..24) {
        & $tailscaleExe status --json *> $null
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 5
    }
    if (-not $ready) {
        throw 'Tailscale local API did not become ready within 120 seconds.'
    }

    & $tailscaleExe debug force-prefer-derp $RegionId
    if ($LASTEXITCODE -ne 0) { throw "Unable to set DERP preference to region $RegionId." }
    & $tailscaleExe debug rebind
    if ($LASTEXITCODE -ne 0) { throw 'Unable to rebind Tailscale after setting DERP preference.' }
    Write-PreferenceLog "preferred_region=$RegionId applied"
}

function Get-OfficialDerpPeers {
    $statusJson = (& $tailscaleExe status --json 2>$null | Out-String)
    if ($LASTEXITCODE -ne 0) { return @() }
    try { $status = $statusJson | ConvertFrom-Json } catch { return @() }
    $targets = @('go2edu', 'go2-2cb21a')
    return @($status.Peer.PSObject.Properties.Value | Where-Object {
        $targets -contains $_.HostName -and $_.Relay -and $_.Relay -ne 'autodl-westb'
    })
}

switch ($Action) {
    'apply' {
        Set-AutoDlDerpPreference
    }
    'monitor' {
        Write-PreferenceLog "monitor started region=$RegionId poll_seconds=$PollSeconds"
        while ($true) {
            try {
                $officialPeers = Get-OfficialDerpPeers
                if ($officialPeers.Count -gt 0) {
                    $names = ($officialPeers | ForEach-Object { "$($_.HostName):$($_.Relay)" }) -join ','
                    Write-PreferenceLog "official_relay_detected=$names; reapplying_region=$RegionId"
                    Set-AutoDlDerpPreference
                }
            } catch {
                Write-PreferenceLog "monitor_error=$($_.Exception.Message)"
            }
            Start-Sleep -Seconds $PollSeconds
        }
    }
    'install' {
        $scriptPath = $PSCommandPath
        $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $taskAction = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Action monitor -RegionId $RegionId -PollSeconds 30"
        $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
        $taskPrincipal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
        $taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Days 7) -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskSettings -Description 'Keep Go2 relay traffic on AutoDL DERP when Tailscale moves it to an official relay. It does not start the AutoDL control tunnel.' -Force -ErrorAction Stop | Out-Null
        Set-AutoDlDerpPreference
        Write-Output "Installed '$taskName' for DERP region $RegionId."
    }
    'status' {
        Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName, State
        if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 20 }
    }
    'remove' {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Output "Removed '$taskName'. Current Tailscale preference remains until Tailscale restarts."
    }
}
