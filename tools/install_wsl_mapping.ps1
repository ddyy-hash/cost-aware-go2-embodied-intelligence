#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [ValidateSet('Enable', 'Provision')]
    [string]$Stage = 'Enable'
)

$ErrorActionPreference = 'Stop'

function Invoke-DismFeature {
    param([Parameter(Mandatory)][string]$FeatureName)

    & dism.exe /Online /Enable-Feature "/FeatureName:$FeatureName" /All /NoRestart
    if ($LASTEXITCODE -notin @(0, 1641, 3010)) {
        throw "Failed to enable Windows feature: $FeatureName"
    }
    if ($LASTEXITCODE -in @(1641, 3010)) {
        Write-Host "Windows feature enabled; restart is required: $FeatureName"
    }
}

if ($Stage -eq 'Enable') {
    Invoke-DismFeature -FeatureName 'Microsoft-Windows-Subsystem-Linux'
    Invoke-DismFeature -FeatureName 'VirtualMachinePlatform'
    Write-Host 'WSL platform features are enabled.'
    Write-Host 'Restart Windows, then run this script again with -Stage Provision.'
    Write-Host 'The installer does not restart the computer.'
    exit 0
}

$distro = 'Ubuntu-24.04'

& wsl.exe --install -d $distro
if ($LASTEXITCODE -ne 0) {
    throw "WSL distribution installation failed: $distro"
}

& wsl.exe --set-default-version 2
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to select WSL 2 as the default version.'
}

& wsl.exe --status
& wsl.exe --list --verbose
Write-Host "WSL provisioning finished with distribution: $distro"
