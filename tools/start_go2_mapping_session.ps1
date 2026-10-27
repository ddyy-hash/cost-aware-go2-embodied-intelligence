[CmdletBinding()]
param(
    [ValidateSet('start', 'status', 'stop')]
    [string]$Action = 'start',
    [ValidateSet('baseline', 'full')]
    [string]$Mode = 'full',
    [ValidateSet('GLOBAL_MAP', 'SESSION_ODOM_LOCAL')]
    [string]$CoordinateScope = 'GLOBAL_MAP',
    [ValidateRange(1, 86400)]
    [int]$DurationSeconds = 300,
    [ValidateRange(60, 86400)]
    [int]$CompletionTimeoutSeconds = 7200,
    [string]$Node = '192.0.2.11',
    [ValidateRange(1, 65535)]
    [int]$ControlPort = 8788,
    [string]$ControlTokenFile,
    [string]$CalibrationManifest = (Join-Path $PSScriptRoot '..\mapping\config\calibration_manifest.json'),
    [string]$SessionEvidence,
    [string]$OperatorForcedCalibrationOverride,
    [string]$SessionId,
    [string]$RequestId,
    [string]$MissionId,
    [string]$TaskId,
    [ValidateRange(1, 2147483647)]
    [int]$TaskRevision,
    [string]$LeaseId,
    [string]$AcceptanceContextPath,
    [string]$Distro = 'Ubuntu-24.04',
    [switch]$UncalibratedCapture,
    [switch]$GeometryOnlyCalibration,
    [switch]$WaitForCompletion
)

$ErrorActionPreference = 'Stop'
$arguments = @{
    Action = $Action
    Mode = $Mode
    CoordinateScope = $CoordinateScope
    DurationSeconds = $DurationSeconds
    CompletionTimeoutSeconds = $CompletionTimeoutSeconds
    Node = $Node
    ControlPort = $ControlPort
    Distro = $Distro
}
foreach ($entry in @{
    ControlTokenFile = $ControlTokenFile
    CalibrationManifest = $CalibrationManifest
    SessionEvidence = $SessionEvidence
    OperatorForcedCalibrationOverride = $OperatorForcedCalibrationOverride
    SessionId = $SessionId
    RequestId = $RequestId
    MissionId = $MissionId
    TaskId = $TaskId
    TaskRevision = $TaskRevision
    LeaseId = $LeaseId
    AcceptanceContextPath = $AcceptanceContextPath
}.GetEnumerator()) {
    if ($entry.Value) { $arguments[$entry.Key] = $entry.Value }
}
if ($WaitForCompletion) { $arguments.WaitForCompletion = $true }
if ($UncalibratedCapture) { $arguments.UncalibratedCapture = $true }
if ($GeometryOnlyCalibration) { $arguments.GeometryOnlyCalibration = $true }
$compatibilityState = [ordered]@{
    transport = 'TAILSCALE_DIRECT'
    as_built_authority = $false
    formal_runtime_override = [bool]$OperatorForcedCalibrationOverride
    measurement_authority = -not $OperatorForcedCalibrationOverride
    runtime_authority = if ($OperatorForcedCalibrationOverride) { 'OPERATOR_FORCED_FORMAL_RUNTIME' } else { 'MEASURED' }
}
& (Join-Path $PSScriptRoot 'start_go2_mapping_session_tailnet.ps1') @arguments
if ($LASTEXITCODE -ne 0) { throw 'Tailnet mapping session control failed.' }
