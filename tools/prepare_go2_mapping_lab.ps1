[CmdletBinding()]
param(
    [string]$Node = '192.0.2.11',
    [ValidateRange(1, 65535)]
    [int]$ControlPort = 8788,
    [string]$ControlTokenFile,
    [Parameter(Mandatory = $true)]
    [string]$CalibrationManifest,
    [Parameter(Mandatory = $true)]
    [string]$SessionEvidence,
    [Parameter(Mandatory = $true)]
    [string]$DepthAnythingWeights,
    [Parameter(Mandatory = $true)]
    [string]$DepthAnythingWeightsSha256File,
    [string]$OutputRoot,
    [string]$Distro = 'Ubuntu-24.04'
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ControlTokenFile = if ($ControlTokenFile) {
    $ControlTokenFile
} else {
    Join-Path $workspaceRoot 'mapping-data\secrets\go2-session-control-token'
}
$OutputRoot = if ($OutputRoot) { $OutputRoot } else { Join-Path $workspaceRoot 'output\lab-preflight' }

function Invoke-DirectRestMethod {
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Get', 'Post')]
        [string]$Method,
        [Parameter(Mandatory)]
        [string]$Uri,
        [hashtable]$Headers,
        [ValidateRange(1, 300)]
        [int]$TimeoutSec = 90
    )
    $request = [System.Net.HttpWebRequest][System.Net.WebRequest]::Create($Uri)
    $request.Method = $Method.ToUpperInvariant()
    $request.Proxy = $null
    $request.Timeout = $TimeoutSec * 1000
    if ($Headers) {
        foreach ($name in $Headers.Keys) {
            $request.Headers[$name] = [string]$Headers[$name]
        }
    }
    $response = $null
    $reader = $null
    try {
        $response = [System.Net.HttpWebResponse]$request.GetResponse()
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        return ($reader.ReadToEnd() | ConvertFrom-Json)
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
    }
}

function Test-TailnetIPv4([string]$Address) {
    $parsed = $null
    if (-not [System.Net.IPAddress]::TryParse($Address, [ref]$parsed)) { return $false }
    $bytes = $parsed.GetAddressBytes()
    return $bytes.Length -eq 4 -and $bytes[0] -eq 100 -and $bytes[1] -ge 64 -and $bytes[1] -le 127
}

function Get-ContentTreeSha256([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (Test-Path -LiteralPath $resolved -PathType Leaf) {
        return (Get-FileHash -Algorithm SHA256 -LiteralPath $resolved).Hash.ToLowerInvariant()
    }
    $lines = Get-ChildItem -LiteralPath $resolved -File -Recurse | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($resolved.Length).TrimStart('\', '/') -replace '\\', '/'
        $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        "$digest  $relative"
    }
    if (-not $lines) { throw 'Depth Anything weights directory is empty.' }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes((($lines -join "`n") + "`n"))
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($bytes)) -replace '-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
}

if (-not (Test-TailnetIPv4 $Node)) { throw 'Node must be an explicit robot Tailscale IPv4 address.' }
foreach ($path in @($ControlTokenFile, $CalibrationManifest, $SessionEvidence, $DepthAnythingWeightsSha256File)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required lab file not found: $path" }
}
if (-not (Test-Path -LiteralPath $DepthAnythingWeights)) { throw 'Depth Anything weights path is missing.' }
$token = (Get-Content -LiteralPath $ControlTokenFile -Raw).Trim()
if ($token.Length -lt 32 -or $token.Length -gt 512) { throw 'Session-control token must contain 32-512 characters.' }

$calibration = Get-Content -LiteralPath $CalibrationManifest -Raw | ConvertFrom-Json
if ($calibration.status -ne 'measured' -or -not $calibration.calibration_id) {
    throw 'Calibration manifest is not measured.'
}
if (-not $calibration.frame_chain.color_camera_info -or -not $calibration.frame_chain.depth_camera_info) {
    throw 'Calibration manifest is missing camera intrinsics.'
}
if (-not $calibration.frame_chain.t_c_l.sha256 -or -not $calibration.frame_chain.t_d_c.sha256) {
    throw 'Calibration manifest is missing measured extrinsics.'
}
if (-not $calibration.clock_fits.lidar -or -not $calibration.clock_fits.color -or -not $calibration.clock_fits.depth) {
    throw 'Calibration manifest is missing synchronized clock fits.'
}
if ($calibration.holdout_evidence.t_d_c_validated -ne $true) {
    throw 'Calibration holdout did not validate the depth-to-color transform.'
}

$evidence = Get-Content -LiteralPath $SessionEvidence -Raw | ConvertFrom-Json
$depthEncoding = if ($evidence.depth_encoding) { $evidence.depth_encoding } else { $evidence.camera.depth_encoding }
$depthScale = if ($evidence.depth_scale_m) { $evidence.depth_scale_m } else { $evidence.camera.depth_scale_m }
if ($depthEncoding -ne '16UC1' -or $null -eq $depthScale -or [double]$depthScale -le 0.0) {
    throw 'Session evidence must contain a verified positive 16UC1 depth scale.'
}

$expectedWeightsSha = ((Get-Content -LiteralPath $DepthAnythingWeightsSha256File -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
if ($expectedWeightsSha -notmatch '^[0-9a-f]{64}$') { throw 'Depth Anything weights SHA file is invalid.' }
$actualWeightsSha = Get-ContentTreeSha256 $DepthAnythingWeights
if ($actualWeightsSha -ne $expectedWeightsSha) { throw 'Depth Anything weights SHA-256 mismatch.' }

$workspaceDrive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($workspaceRoot).Substring(0, 1))
if ($workspaceDrive.Free -lt 50GB) { throw 'PC storage requires at least 50 GiB free for lab capture.' }
& wsl.exe -d $Distro -- test -x /home/a/micromamba/bin/micromamba
if ($LASTEXITCODE -ne 0) { throw 'Owned WSL micromamba environment is unavailable.' }

Get-Command tailscale.exe -ErrorAction Stop | Out-Null
$probe = & tailscale.exe ping --timeout=10s $Node 2>&1
if (($probe | Out-String) -notmatch 'pong from') { throw "Tailscale cannot reach $Node." }
$headers = @{ Authorization = "Bearer $token" }
$preflight = Invoke-DirectRestMethod -Method Get -Uri "http://$($Node):$ControlPort/v1/preflight" -Headers $headers
if ($preflight.ready -ne $true) { throw "Robot preflight failed: $($preflight.report)" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$report = [ordered]@{
    schema_version = 1
    state = 'LAB_PREPARE_OK'
    synthetic = $false
    robot_tailnet_ip = $Node
    calibration_id = $calibration.calibration_id
    calibration_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationManifest).Hash.ToLowerInvariant()
    session_evidence_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $SessionEvidence).Hash.ToLowerInvariant()
    depth_encoding = $depthEncoding
    depth_scale_m = [double]$depthScale
    depth_anything_weights_sha256 = $actualWeightsSha
    robot_preflight = $preflight
}
$reportPath = Join-Path $OutputRoot 'lab-preflight.json'
$temporary = "$reportPath.$([Guid]::NewGuid().ToString('N')).partial"
try {
    [System.IO.File]::WriteAllText(
        $temporary,
        (($report | ConvertTo-Json -Depth 10 -Compress) + "`n"),
        (New-Object System.Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $temporary -Destination $reportPath -Force
} finally {
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
}
Write-Output "LAB_PREPARE_OK report=$reportPath calibration=$($calibration.calibration_id) model=$actualWeightsSha"
