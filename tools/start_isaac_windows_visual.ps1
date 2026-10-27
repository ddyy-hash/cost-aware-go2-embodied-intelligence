[CmdletBinding()]
param(
    [string]$IsaacRoot = $env:ISAAC_SIM_ROOT,
    [ValidateSet("flat", "up", "down")]
    [string]$Terrain = "up",
    [int]$Steps = 1000,
    [double]$RootHeight = 0.20,
    [double]$StartX = -0.80,
    [double]$Vx = 0.60,
    [double]$Vy = 0.0,
    [ValidateSet("analytic", "raycast")]
    [string]$HeightmapMode = "analytic",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

$roots = @()
if ($IsaacRoot) {
    $roots += $IsaacRoot
}
$roots += @(
    "C:\isaac-sim",
    "C:\isaac_sim",
    (Join-Path $env:LOCALAPPDATA "ov\pkg\isaac-sim-6.0.1"),
    (Join-Path $env:LOCALAPPDATA "ov\pkg\isaac-sim-5.1.0"),
    (Join-Path $env:LOCALAPPDATA "ov\pkg\isaac-sim-4.5.0")
)

$python = $null
foreach ($root in $roots) {
    if (-not $root) { continue }
    $candidate = Join-Path $root "python.bat"
    if (Test-Path -LiteralPath $candidate) {
        $python = (Resolve-Path -LiteralPath $candidate).Path
        break
    }
}
if (-not $python) {
    throw "Native Isaac Sim was not found. Install it on Windows or set ISAAC_SIM_ROOT to the directory containing python.bat. WSL Isaac Sim is headless-only for this project."
}
if ($Steps -le 0) {
    throw "Steps must be positive."
}

if (-not $Output) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Output = Join-Path $repo "output\isaac-sim\windows-visual-$Terrain-$stamp.json"
}

$arguments = @(
    (Join-Path $repo "tools\run_isaac_go2_pgtt_smoke.py"),
    "--visual",
    "--usd", (Join-Path $repo "output\isaac-sim\go2_driven\go2.usd\go2\go2.usda"),
    "--policy", (Join-Path $repo "output\robot-release\pgtt-go2\policy_level03_run0.npz"),
    "--robot-root", "/Go2",
    "--terrain", $Terrain,
    "--steps", $Steps.ToString(),
    "--root-height", $RootHeight.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--start-x", $StartX.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--vx", $Vx.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--vy", $Vy.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--heightmap-mode", $HeightmapMode,
    "--output", $Output
)

Push-Location $repo
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
