[CmdletBinding()]
param(
    [string]$Node = '192.0.2.11',
    [string]$UserName = 'unitree',
    [string]$PrivateKeyFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Distro = 'Ubuntu-24.04',
    [ValidatePattern('^[a-z0-9-]+$')]
    [string]$PrivateRegionCode = 'autodl-westb',
    [ValidateRange(900, 9999)]
    [int]$PrivateRegionId = 901,
    [ValidateRange(0, 90)]
    [double]$MinimumImprovementPercent = 10,
    [switch]$ApplyPreference,
    [switch]$UseTailscaleProxy,
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
foreach ($command in @('tailscale.exe', 'ssh.exe', 'wsl.exe')) { Get-Command $command -ErrorAction Stop | Out-Null }
if (-not (Test-Path -LiteralPath $PrivateKeyFile -PathType Leaf)) { throw "Private key not found: $PrivateKeyFile" }
if ($Node -notmatch '^100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}$') { throw 'Node must be the Go2 Tailscale IPv4 address.' }

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutputDirectory = if ($OutputDirectory) { $OutputDirectory } else { Join-Path $workspaceRoot 'mapping-data\network' }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$pcPath = Join-Path $OutputDirectory "derp-netcheck-pc-$stamp.txt"
$robotPath = Join-Path $OutputDirectory "derp-netcheck-go2-$stamp.txt"
$decisionPath = Join-Path $OutputDirectory "derp-decision-$stamp.json"

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    # netcheck can emit harmless port-mapping diagnostics on stderr.
    $pcReport = (& tailscale.exe netcheck 2>&1 | Out-String)
    $pcExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $savedErrorActionPreference
}
if ($pcExitCode -ne 0) { throw 'PC tailscale netcheck failed.' }
[System.IO.File]::WriteAllText($pcPath, $pcReport, [System.Text.UTF8Encoding]::new($false))

$sshArgs = @('-i', $PrivateKeyFile, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12')
if ($UseTailscaleProxy) { $sshArgs += @('-o', 'ProxyCommand=tailscale.exe nc %h %p') }
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $robotReport = (& ssh.exe @sshArgs "${UserName}@${Node}" 'tailscale netcheck' 2>&1 | Out-String)
    $robotExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $savedErrorActionPreference
}
if ($robotExitCode -ne 0) { throw 'Go2 tailscale netcheck failed; no DERP preference was changed.' }
[System.IO.File]::WriteAllText($robotPath, $robotReport, [System.Text.UTF8Encoding]::new($false))

$status = (& tailscale.exe status --json | Out-String | ConvertFrom-Json)
$peer = $status.Peer.PSObject.Properties.Value | Where-Object { $_.TailscaleIPs -contains $Node } | Select-Object -First 1
$peerIsDirect = [bool]($peer -and $peer.CurAddr)
$linuxRoot = (& wsl.exe -d $Distro wslpath -a ($workspaceRoot -replace '\\', '/') 2>$null | Select-Object -First 1).Trim()
if (-not $linuxRoot) {
    $drive = $workspaceRoot.Substring(0, 1).ToLowerInvariant()
    $linuxRoot = '/mnt/' + $drive + ($workspaceRoot.Substring(2) -replace '\\', '/')
}
function Convert-ToWorkspaceRelativeUnixPath([string]$Path) {
    if (-not $Path.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the workspace: $Path"
    }
    return $Path.Substring($workspaceRoot.Length).TrimStart('\', '/') -replace '\\', '/'
}
$relativePc = Convert-ToWorkspaceRelativeUnixPath $pcPath
$relativeRobot = Convert-ToWorkspaceRelativeUnixPath $robotPath
$selectorArgs = @(
    '-d', $Distro, '--cd', $linuxRoot, '--',
    '/home/a/micromamba/bin/micromamba', 'run', '-n', 'go2-mapping-pc',
    'python', '-m', 'mapping.pc.go2_mapping_network.derp_selector',
    '--pc-netcheck', "$linuxRoot/$relativePc",
    '--robot-netcheck', "$linuxRoot/$relativeRobot",
    '--private-region', "$PrivateRegionCode=$PrivateRegionId",
    '--minimum-improvement-percent', "$MinimumImprovementPercent"
)
if ($peerIsDirect) { $selectorArgs += '--peer-direct' }
$decisionJson = (& wsl.exe @selectorArgs 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "DERP selector failed: $decisionJson" }
$decision = $decisionJson | ConvertFrom-Json
[System.IO.File]::WriteAllText($decisionPath, $decisionJson + "`n", [System.Text.UTF8Encoding]::new($false))

if ($ApplyPreference -and $decision.preferred_region_id -gt 0) {
    & tailscale.exe debug force-prefer-derp $decision.preferred_region_id
    if ($LASTEXITCODE -ne 0) { throw 'PC DERP preference failed.' }
    & ssh.exe @sshArgs "${UserName}@${Node}" "tailscale debug force-prefer-derp $($decision.preferred_region_id) && tailscale debug rebind"
    if ($LASTEXITCODE -ne 0) {
        & tailscale.exe debug force-prefer-derp 0 | Out-Null
        throw 'Go2 DERP preference failed; the PC preference was rolled back.'
    }
    & tailscale.exe debug rebind | Out-Null
    Write-Output "DERP_PREFERENCE_APPLIED region=$($decision.preferred_region_id); official DERP fallback remains enabled"
} else {
    Write-Output 'P2P/direct or automatic DERP selection remains active; no preference was changed.'
}
Write-Output "DERP_DECISION reason=$($decision.reason) region=$($decision.preferred_region_id) report=$decisionPath"
