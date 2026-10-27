[CmdletBinding()]
param(
    [ValidateSet('start','status','stop')]
    [string]$Action = 'start',
    [ValidateRange(60, 3600)]
    [int]$DurationSeconds = 600,
    [string]$Node = '192.0.2.11',
    [string]$RequestId,
    [string]$AcceptanceContextPath = (Join-Path $PSScriptRoot '..\mapping-data\sessions\room-scan-context.json'),
    [switch]$WaitForCompletion
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifest = (Resolve-Path (Join-Path $root 'mapping\config\calibration_manifest.json')).Path
$controlToken = Join-Path $root 'mapping-data\secrets\go2-session-control-token'
$args = @{
    Action = $Action
    Mode = 'full'
    CoordinateScope = 'SESSION_ODOM_LOCAL'
    DurationSeconds = $DurationSeconds
    Node = $Node
    CalibrationManifest = $manifest
    ControlTokenFile = $controlToken
    GeometryOnlyCalibration = $true
    AcceptanceContextPath = $AcceptanceContextPath
}
if ($RequestId) { $args.RequestId = $RequestId }
if ($WaitForCompletion) { $args.WaitForCompletion = $true }
& (Join-Path $PSScriptRoot 'start_go2_mapping_session_tailnet.ps1') @args
if ($LASTEXITCODE -ne 0) { throw 'Room scan session control failed.' }
