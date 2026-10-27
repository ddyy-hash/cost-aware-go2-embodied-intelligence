[CmdletBinding()]
param([string]$Distro = 'Ubuntu-24.04')

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxWorkspace = '/mnt/' + $workspaceRoot.Substring(0, 1).ToLowerInvariant() + ($workspaceRoot.Substring(2) -replace '\\', '/')

& wsl.exe -d $Distro -- bash -lc "export MAMBA_ROOT_PREFIX=\$HOME/micromamba && \$HOME/micromamba/bin/micromamba run -n go2-mapping-pc python '$linuxWorkspace/mapping/pc/verify_environment.py'"
exit $LASTEXITCODE
