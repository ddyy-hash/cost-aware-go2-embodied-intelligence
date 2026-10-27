[CmdletBinding()]
param([string]$Distro = 'Ubuntu-24.04')

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxWorkspace = '/mnt/' + $workspaceRoot.Substring(0, 1).ToLowerInvariant() + ($workspaceRoot.Substring(2) -replace '\\', '/')

$distros = & wsl.exe --list --quiet
if ($LASTEXITCODE -ne 0 -or $distros -notcontains $Distro) {
    throw "WSL distribution '$Distro' is unavailable. Complete install_wsl_mapping.ps1 -Stage Provision first."
}

& wsl.exe -d $Distro -- bash "$linuxWorkspace/mapping/pc/start_bootstrap_wsl.sh"
if ($LASTEXITCODE -ne 0) { throw 'WSL mapping bootstrap launch failed.' }
