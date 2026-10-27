[CmdletBinding()]
param(
    [string]$RobotAddress = '192.168.123.18',
    [string]$OwnerUser = 'unitree',
    [string]$PublicKeyFile = "$HOME/.ssh/id_ed25519.pub",
    [string]$TailscaleArchive = "$PSScriptRoot/../tailscale_1.98.2_arm64.tgz"
)

$ErrorActionPreference = 'Stop'
$installer = Join-Path $PSScriptRoot 'install_go2_edge_node.sh'
$developerTool = Join-Path $PSScriptRoot 'manage_go2_developer.sh'

foreach ($path in @($installer, $developerTool, $PublicKeyFile, $TailscaleArchive)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $path"
    }
}

$test = Test-NetConnection $RobotAddress -Port 22 -InformationLevel Quiet
if (-not $test) {
    throw "Cannot reach ${RobotAddress}:22. Connect the wired cable and give the PC a 192.168.123.x/24 address first."
}

$target = "${OwnerUser}@${RobotAddress}"
Write-Host "Uploading edge-node package to $target over the wired link..."
& scp $TailscaleArchive "${target}:/tmp/tailscale_arm64.tgz"
if ($LASTEXITCODE -ne 0) { throw 'Tailscale archive upload failed.' }
& scp $PublicKeyFile "${target}:/tmp/edge_owner.pub"
if ($LASTEXITCODE -ne 0) { throw 'Owner public-key upload failed.' }
& scp $installer "${target}:/tmp/install_go2_edge_node.sh"
if ($LASTEXITCODE -ne 0) { throw 'Installer upload failed.' }
& scp $developerTool "${target}:/tmp/manage_go2_developer.sh"
if ($LASTEXITCODE -ne 0) { throw 'Developer tool upload failed.' }

Write-Host 'Installing services. SSH and sudo may each request the robot maintenance password.'
& ssh -t $target "sudo bash /tmp/install_go2_edge_node.sh /tmp/tailscale_arm64.tgz '$OwnerUser' /tmp/edge_owner.pub /tmp/manage_go2_developer.sh"
if ($LASTEXITCODE -ne 0) { throw 'Robot edge-node installation failed.' }

Write-Host 'Open the Tailscale authorization URL printed above, then verify the new go2-* node in the admin console.'

