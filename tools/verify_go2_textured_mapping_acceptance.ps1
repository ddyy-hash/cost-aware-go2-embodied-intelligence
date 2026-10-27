[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ContextPath,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\output\field-acceptance\validated-texture'),
    [string]$Distro = 'Ubuntu-24.04',
    [Alias('Restart-OwnedDashboard')]
    [switch]$RestartOwnedDashboard
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$contextPath = (Resolve-Path -LiteralPath $ContextPath).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$context = Get-Content -LiteralPath $contextPath -Raw | ConvertFrom-Json

function Convert-ToWslPath([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    return '/mnt/' + $resolved.Substring(0, 1).ToLowerInvariant() + ($resolved.Substring(2) -replace '\\', '/')
}

function Test-LowerSha([object]$Value) {
    return $Value -is [string] -and $Value -cmatch '^[0-9a-f]{64}$'
}

function Invoke-Operator([string[]]$Arguments) {
    $linuxWorkspace = Convert-ToWslPath $workspace
    $command = @(
        '-d', $Distro, '--cd', $linuxWorkspace, '--', 'env', 'PYTHONPATH=.',
        '/home/a/micromamba/bin/micromamba', 'run', '-n', 'go2-mapping-pc', 'python',
        '-m', 'mapping.pc.go2_mapping_appearance.field_acceptance_operator'
    ) + $Arguments
    & wsl.exe @command
    if ($LASTEXITCODE -ne 0) { throw 'Physical textured mapping operator validation failed.' }
}

function Get-OwnedDashboard([int]$ProcessId, [string]$ExpectedWorkspace) {
    $linuxWorkspace = Convert-ToWslPath $ExpectedWorkspace
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if (
        $null -eq $process -or
        $process.Name -ne 'wsl.exe' -or
        $process.CommandLine -notlike "*--cd $linuxWorkspace*" -or
        $process.CommandLine -notlike '*mapping/pc/start_dashboard_wsl.sh*'
    ) {
        throw 'Owned dashboard PID does not identify this worktree dashboard process.'
    }
    return $process
}

function Invoke-ThreeJsLoad([string]$DashboardUrl, [object[]]$Artifacts) {
    $commandShell = (Get-Command cmd.exe -ErrorAction Stop).Source
    Get-Command npx.cmd -ErrorAction Stop | Out-Null
    $session = 'go2-glb-' + [Guid]::NewGuid().ToString('N').Substring(0, 10)
    $results = [ordered]@{}
    try {
        foreach ($artifact in $Artifacts) {
            $name = [string]$artifact.artifact_name
            if ($name -notmatch '^(?:tile-[0-9a-f]{64}-[0-9a-f]{64}-[0-9a-f]{64}|building-[0-9a-f]{64}-[0-9a-f]{64})\.glb$') {
                throw "Unsafe GLB name in prepared evidence: $name"
            }
            $artifactPath = '/api/reconstruction/artifacts/' + $name
            $loaderUrl = $DashboardUrl.TrimEnd('/') + '/static/field-acceptance-glb.html?artifact=' + [Uri]::EscapeDataString($artifactPath)
            $code = @"
async (page) => {
  await page.goto('$loaderUrl');
  await page.waitForFunction(() => document.title !== 'THREEJS_LOADING', null, {timeout: 30000});
  if (await page.title() !== 'THREEJS_LOADED') {
    throw new Error(await page.locator('html').getAttribute('data-detail'));
  }
}
"@
            $run = @('/d', '/c', 'npx.cmd', '--yes', '--package', '@playwright/cli', 'playwright-cli', "-s=$session", 'run-code', $code)
            & $commandShell @run
            if ($LASTEXITCODE -ne 0) { throw "Three.js GLTFLoader rejected $name" }
            $results[$name] = $true
        }
    } finally {
        $close = @('/d', '/c', 'npx.cmd', '--yes', '--package', '@playwright/cli', 'playwright-cli', "-s=$session", 'close')
        & $commandShell @close 2>$null | Out-Null
    }
    return $results
}

if ($context.schema_version -ne 1) { throw 'Physical capture context schema_version must be 1.' }
if ($context.synthetic -ne $false) { throw 'Synthetic evidence cannot satisfy physical acceptance.' }
if ($context.evidence_kind -ne 'PHYSICAL_GO2_CAPTURE') { throw 'Physical acceptance requires PHYSICAL_GO2_CAPTURE.' }
if ($context.transport -ne 'TAILSCALE_DIRECT') { throw 'Physical acceptance requires TAILSCALE_DIRECT.' }
if ([System.IO.Path]::GetFullPath([string]$context.worktree_root) -ne $workspace) {
    throw 'Capture context belongs to a different worktree.'
}
$dashboardUri = $null
if (-not [Uri]::TryCreate([string]$context.dashboard_url, [UriKind]::Absolute, [ref]$dashboardUri) -or $dashboardUri.Scheme -notin @('http', 'https')) {
    throw 'Capture context requires an explicit HTTP(S) dashboard_url.'
}
if (-not $context.owned_dashboard_pid -or [int]$context.owned_dashboard_pid -le 0) {
    throw 'Capture context requires an owned dashboard PID.'
}
if ($context.depth_anything_weights_path) {
    if (-not (Test-Path -LiteralPath $context.depth_anything_weights_path -PathType Leaf)) {
        throw 'Capture context Depth Anything weights are unavailable.'
    }
    if (-not (Test-LowerSha $context.depth_anything_weights_sha256)) {
        throw 'Capture context Depth Anything weights SHA-256 is invalid.'
    }
}
if (-not $RestartOwnedDashboard) {
    throw 'Physical acceptance requires -RestartOwnedDashboard for exact pre/post identity verification.'
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
$linuxContext = Convert-ToWslPath $contextPath
$linuxOutput = Convert-ToWslPath $output
$linuxWorkspace = Convert-ToWslPath $workspace
Invoke-Operator @('prepare', '--context', $linuxContext, '--output', $linuxOutput, '--workspace', $linuxWorkspace)

if ($RestartOwnedDashboard) {
    $owned = Get-OwnedDashboard -ProcessId ([int]$context.owned_dashboard_pid) -ExpectedWorkspace $workspace
    Stop-Process -Id $owned.ProcessId -Force -ErrorAction Stop
    $startArguments = @{
        Mode = [string]$context.mode
        Node = [string]$context.robot_tailnet_ip
        Distro = if ($context.distro) { [string]$context.distro } else { $Distro }
        CalibrationManifest = [string]$context.calibration_manifest_path
        SessionEvidence = [string]$context.session_evidence_path
    }
    if ($context.depth_anything_weights_path) {
        $startArguments.DepthAnythingWeights = [string]$context.depth_anything_weights_path
        $startArguments.DepthAnythingWeightsSha256 = [string]$context.depth_anything_weights_sha256
    }
    & (Join-Path $PSScriptRoot 'start_go2_pc_stack.ps1') @startArguments
    if ($LASTEXITCODE -ne 0) { throw 'Owned dashboard restart failed.' }
}

$preparedPath = Join-Path $output 'prepared-evidence.json'
$prepared = Get-Content -LiteralPath $preparedPath -Raw | ConvertFrom-Json
$artifacts = @($prepared.local_exports) + @($prepared.building_export)
$browserResults = Invoke-ThreeJsLoad -DashboardUrl ([string]$context.dashboard_url) -Artifacts $artifacts
$browserResultsPath = Join-Path $output 'threejs-results.json'
[System.IO.File]::WriteAllText(
    $browserResultsPath,
    (($browserResults | ConvertTo-Json -Compress) + [Environment]::NewLine),
    (New-Object System.Text.UTF8Encoding($false))
)

Invoke-Operator @(
    'finalize', '--context', $linuxContext, '--output', $linuxOutput, '--workspace', $linuxWorkspace,
    '--browser-results', (Convert-ToWslPath $browserResultsPath), '--restart-performed'
)
Write-Output 'LIVE_TEXTURED_MAPPING_ACCEPTED'
