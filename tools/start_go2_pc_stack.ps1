[CmdletBinding()]
param(
    [ValidateSet('full', 'baseline')]
    [string]$Mode = 'full',
    [ValidateSet('direct', 'autodl')]
    [string]$HighFidelityTransport = 'direct',
    [string]$Node = '192.0.2.11',
    [string]$UserName = 'unitree',
    [string]$PrivateKeyFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Distro = 'Ubuntu-24.04',
    [string]$CalibrationManifest,
    [string]$SessionEvidence,
    [string]$OperatorForcedCalibrationOverride,
    [switch]$AllowOperatorForcedCalibrationOverride,
    [string]$RoomScanOperatorTokenFile,
    [string]$SessionControlTokenFile,
    [string]$DerivedSurfaceJson,
    [string]$IngestSessionId,
    [ValidateSet('RAW_LIDAR_WITH_RECORDED_LIO','RECORDED_LIO_REGISTERED_CLOUD','RECORDED_LIO_SESSION_ODOM_CLOUD')]
    [string]$ReconstructionSourceMode = 'RAW_LIDAR_WITH_RECORDED_LIO',
    [switch]$AllowDashboardRestart,
    [string]$DepthAnythingWeights,
    [string]$DepthAnythingWeightsSha256,
    [ValidateSet('transformers', 'da3')]
    [string]$DepthAnythingBackend = 'transformers',
    [string]$DepthAnythingModelName = '',
    [ValidateSet('cpu', 'cuda')]
    [string]$DepthAnythingDevice = 'cuda',
    [ValidateSet('float16', 'bfloat16', 'float32')]
    [string]$DepthAnythingPrecision = 'float16',
    [ValidateRange(0, 9999)]
    [int]$PreferredDerpRegion = 0,
    [string]$BulkEndpoint = '',
    [string]$BulkTokenFile,
    [ValidateRange(1, 8)]
    [int]$BulkPullWorkers = 4,
    [bool]$UseTailscaleProxy = $false
)

$ErrorActionPreference = 'Stop'
Get-Command tailscale.exe -ErrorAction Stop | Out-Null
$tailnetIpProcess = [System.Diagnostics.Process]::new()
$tailnetIpProcess.StartInfo.FileName = (Get-Command tailscale.exe -ErrorAction Stop).Source
$tailnetIpProcess.StartInfo.Arguments = 'ip -4'
$tailnetIpProcess.StartInfo.UseShellExecute = $false
$tailnetIpProcess.StartInfo.RedirectStandardOutput = $true
$tailnetIpProcess.StartInfo.RedirectStandardError = $true
$tailnetIpProcess.StartInfo.CreateNoWindow = $true
try {
    if (-not $tailnetIpProcess.Start()) {
        throw 'tailscale.exe did not start.'
    }
    $tailnetIpOutput = $tailnetIpProcess.StandardOutput.ReadToEnd()
    $tailnetIpError = $tailnetIpProcess.StandardError.ReadToEnd()
    $tailnetIpProcess.WaitForExit()
    $tailnetIpResult = [pscustomobject]@{
        ExitCode = $tailnetIpProcess.ExitCode
        Output = $tailnetIpOutput
        Error = $tailnetIpError
    }
} finally {
    $tailnetIpProcess.Dispose()
}
if ($tailnetIpResult.ExitCode -ne 0) {
    $tailnetIpDiagnostic = "$($tailnetIpResult.Output)`n$($tailnetIpResult.Error)".Trim()
    if ($tailnetIpDiagnostic -match '(?i)access is denied|E_ACCESS_DENIED') {
        throw "Tailscale local API access was denied. Run this PC-stack maintenance command from a normal interactive desktop administrator token. Details: $tailnetIpDiagnostic"
    }
    throw "Unable to query the PC Tailscale IPv4 address. Details: $tailnetIpDiagnostic"
}
$tailnetIp = ([string]$tailnetIpResult.Output).Trim()
if ($tailnetIp -notmatch '^100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}$') {
    throw "Unable to determine the PC Tailscale IPv4 address: $tailnetIp"
}
if ($Node -notmatch '^100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}$') {
    throw 'Direct high-fidelity transport requires -Node to be the robot Tailscale IPv4 address.'
}
$receiverHost = $tailnetIp
$receiverHealthUri = "http://${receiverHost}:8000/health"

function Set-PreferredDerpRegion {
    try {
        $derpMap = (& tailscale.exe debug derp-map | Out-String | ConvertFrom-Json)
        if ($derpMap.Regions.PSObject.Properties.Name -notcontains "$PreferredDerpRegion") {
            throw "Tailnet DERP map does not contain region $PreferredDerpRegion"
        }
        & tailscale.exe debug force-prefer-derp $PreferredDerpRegion
        if ($LASTEXITCODE -ne 0) { throw 'force-prefer-derp failed' }
        & tailscale.exe debug rebind
        if ($LASTEXITCODE -ne 0) { throw 'Tailscale rebind failed' }
        Write-Output "Preferred DERP region: $PreferredDerpRegion (official regions remain fallback)"
    } catch {
        Write-Warning "Custom DERP preference was not applied; continuing with official fallback: $($_.Exception.Message)"
    }
}

if ($PreferredDerpRegion -gt 0 -and $HighFidelityTransport -eq 'autodl') {
    Write-Output 'Deferred AutoDL DERP preference until runtime mapping verification completes.'
} elseif ($PreferredDerpRegion -gt 0) {
    Set-PreferredDerpRegion
} else {
    Write-Output 'Direct path remains preferred; custom and official DERP regions are automatic fallback.'
}
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defaultDepthAnythingWeights = Join-Path $workspaceRoot 'models\depth-anything-v2-small-hf-5426e4f'
$defaultDepthAnythingManifest = Join-Path $workspaceRoot 'mapping\config\pc_model_manifest.json'
$assetPrepareScript = Join-Path $workspaceRoot 'tools\prepare_go2_mapping_assets.ps1'
if (-not $DepthAnythingWeights) {
    $DepthAnythingWeights = $defaultDepthAnythingWeights
}
if (-not $DepthAnythingWeightsSha256) {
    $DepthAnythingWeightsSha256 = '37a91a613d005c92f71e6b0caf2790c112ab06a7794fd2fa275521be14510137'
}
if ((Resolve-Path -LiteralPath $DepthAnythingWeights -ErrorAction SilentlyContinue).Path -eq
    (Resolve-Path -LiteralPath $defaultDepthAnythingWeights -ErrorAction SilentlyContinue).Path) {
    if (-not (Test-Path -LiteralPath $defaultDepthAnythingManifest -PathType Leaf)) {
        throw "Verified PC model manifest is missing: $defaultDepthAnythingManifest"
    }
}
$receiverScript = Join-Path $workspaceRoot 'mapping\pc\ssh_tunnel_receiver.py'
$bulkPullerScript = Join-Path $workspaceRoot 'mapping\pc\autodl_spool_puller.py'
$BulkTokenFile = if ($BulkTokenFile) { $BulkTokenFile } else { Join-Path $workspaceRoot 'mapping-data\secrets\autodl-bulk-token' }
$dataRoot = Join-Path $workspaceRoot 'mapping-data'
$RoomScanOperatorTokenFile = if ($RoomScanOperatorTokenFile) { $RoomScanOperatorTokenFile } else { Join-Path $dataRoot 'secrets\go2-room-scan-operator-token.json' }
$SessionControlTokenFile = if ($SessionControlTokenFile) { $SessionControlTokenFile } else { Join-Path $dataRoot 'secrets\go2-session-control-token' }
$inboxRoot = Join-Path $dataRoot 'inbox'
$logRoot = Join-Path $dataRoot 'session-logs'
$runtimeRoot = Join-Path $dataRoot 'pc-stack'
if ((Resolve-Path -LiteralPath $DepthAnythingWeights -ErrorAction SilentlyContinue).Path -eq
    (Resolve-Path -LiteralPath $defaultDepthAnythingWeights -ErrorAction SilentlyContinue).Path) {
    & $assetPrepareScript -WeightsPath $DepthAnythingWeights -ModelManifest $defaultDepthAnythingManifest `
        -OutputDirectory (Join-Path $runtimeRoot 'assets')
}
$automaticReconstructionRoot = Join-Path $workspaceRoot 'output\reconstruction\derived'
$automaticReconstructionStatusPath = Join-Path $runtimeRoot 'automatic-reconstruction-current.json'
$automaticReconstructionScript = Join-Path $workspaceRoot 'mapping\pc\start_session_reconstruction_wsl.sh'
$automaticReconstructionLinuxPidPath = Join-Path $runtimeRoot 'automatic-reconstruction-linux.pid'
$automaticReconstructionSensorFrameTransformId = 'PROVISIONAL_GO2_UTLIDAR_PITCH180_v2'
$automaticReconstructionPointerProtocol = 'automatic-reconstruction-pointer-v1'
$sessionOdomLocalRoot = Join-Path $workspaceRoot 'output\reconstruction\session-odom-local'
$sessionOdomLocalStatusPath = Join-Path $runtimeRoot 'session-odom-local-current.json'
$dashboardInternalPort = 8767
$statusPath = Join-Path $dataRoot 'session-status.json'
$transferStatusPath = Join-Path $dataRoot 'progressive-transfer-status.json'
$qualityGatesPath = Join-Path $workspaceRoot 'mapping\config\quality_gates.json'
$captureLabel = if ($Mode -eq 'full') { 'lidar-imu-rgbd' } else { 'rgbd-baseline' }
$ingestRoot = Join-Path $inboxRoot $captureLabel
New-Item -ItemType Directory -Force -Path $inboxRoot,$ingestRoot,$logRoot,$runtimeRoot,$automaticReconstructionRoot,$sessionOdomLocalRoot | Out-Null

 $requiredPaths = @($receiverScript, $qualityGatesPath, $automaticReconstructionScript)
# AutoDL is opt-in; the direct path must remain usable with no cloud token.
# if ($HighFidelityTransport -eq 'autodl')
if ($HighFidelityTransport -eq 'autodl') { $requiredPaths += $bulkPullerScript }
foreach ($required in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required PC mapping path is missing: $required"
    }
}

if ($Mode -eq 'full' -and $HighFidelityTransport -eq 'autodl') {
    $autodlManagerOutput = @(
        & (Join-Path $PSScriptRoot 'manage_autodl_derp.ps1') -Action start -BulkTokenFile $BulkTokenFile
    )
    $autodlManagerExitCode = $LASTEXITCODE
    $autodlManagerOutput | Write-Output
    if ($autodlManagerExitCode -ne 0) {
        throw "AutoDL DERP manager failed with exit code $autodlManagerExitCode"
    }
    $endpointRecord = @(
        $autodlManagerOutput | Where-Object { "$_" -match '^AUTODL_DERP_PUBLIC_ENDPOINT url=(\S+)' }
    ) | Select-Object -Last 1
    if (-not $endpointRecord -or "$endpointRecord" -notmatch '^AUTODL_DERP_PUBLIC_ENDPOINT url=(\S+)') {
        throw 'Could not read the runtime AutoDL public endpoint from the verified manager output'
    }
    $runtimeBulkEndpoint = $Matches[1]
    if ($BulkEndpoint -and $BulkEndpoint.TrimEnd('/') -ne $runtimeBulkEndpoint.TrimEnd('/')) {
        throw "Configured -BulkEndpoint does not match the current AutoDL public mapping: $runtimeBulkEndpoint"
    }
    $BulkEndpoint = $runtimeBulkEndpoint
    if ($PreferredDerpRegion -gt 0) {
        $expectedMapMatch = "^AUTODL_DERP_MAP_MATCH region=901 endpoint=$([Regex]::Escape($runtimeBulkEndpoint))$"
        if (-not @($autodlManagerOutput | Where-Object { "$_" -match $expectedMapMatch }).Count) {
            throw 'Current AutoDL region 901 map did not match the discovered endpoint; refusing to force the stale mapping.'
        }
        Set-PreferredDerpRegion
    }
    if (-not (Test-Path -LiteralPath $BulkTokenFile -PathType Leaf)) {
        throw "AutoDL bulk token was not created: $BulkTokenFile"
    }
}
if ($OperatorForcedCalibrationOverride -and -not $AllowOperatorForcedCalibrationOverride) {
    throw 'OperatorForcedCalibrationOverride requires explicit AllowOperatorForcedCalibrationOverride.'
}
if ($AllowOperatorForcedCalibrationOverride -and -not $OperatorForcedCalibrationOverride) {
    throw 'AllowOperatorForcedCalibrationOverride requires an override file.'
}
foreach ($evidence in @($CalibrationManifest, $SessionEvidence, $OperatorForcedCalibrationOverride)) {
    if ($evidence -and -not (Test-Path -LiteralPath $evidence -PathType Leaf)) {
        throw "Quality evidence file not found: $evidence"
    }
}
if ($CalibrationManifest) { $CalibrationManifest = (Resolve-Path -LiteralPath $CalibrationManifest).Path }
if ($SessionEvidence) { $SessionEvidence = (Resolve-Path -LiteralPath $SessionEvidence).Path }
if ($OperatorForcedCalibrationOverride) {
    $OperatorForcedCalibrationOverride = (Resolve-Path -LiteralPath $OperatorForcedCalibrationOverride).Path
    $operatorOverridePayload = Get-Content -LiteralPath $OperatorForcedCalibrationOverride -Raw | ConvertFrom-Json
    $operatorOverrideCalibrationId = [string]$operatorOverridePayload.calibration_id
    if ($operatorOverrideCalibrationId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
        throw 'OperatorForcedCalibrationOverride does not contain a canonical calibration_id.'
    }
}
if ($DerivedSurfaceJson) {
    if (-not (Test-Path -LiteralPath $DerivedSurfaceJson -PathType Leaf)) {
        throw "Derived surface JSON was not found: $DerivedSurfaceJson"
    }
    $DerivedSurfaceJson = (Resolve-Path -LiteralPath $DerivedSurfaceJson).Path
    if ([System.IO.Path]::GetFileName($DerivedSurfaceJson) -ne 'measured-surface.json') {
        throw 'Derived surface input must name measured-surface.json.'
    }
}
if ($IngestSessionId -and $IngestSessionId -notmatch '^segment-[0-9]{8}T[0-9]{6}Z$') {
    throw 'IngestSessionId must match segment-YYYYMMDDTHHMMSSZ.'
}
if ($IngestSessionId -and $Mode -ne 'full') {
    throw 'IngestSessionId is valid only for full LiDAR mapping sessions.'
}
if ($DepthAnythingWeights) {
    $DepthAnythingWeights = (Resolve-Path -LiteralPath $DepthAnythingWeights).Path
    if (-not $DepthAnythingWeightsSha256 -or $DepthAnythingWeightsSha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'Depth Anything weights SHA-256 must be explicit lowercase hex.'
    }
}
$dashboardConfig = [ordered]@{
    schema_version = 1
    room_scan_runtime_revision = 3
    ingest_root = $ingestRoot
    lifecycle_root = $inboxRoot
    status_path = $statusPath
    transfer_status_path = $transferStatusPath
    quality_gates = $qualityGatesPath
    calibration_manifest = [string]$CalibrationManifest
    session_evidence = [string]$SessionEvidence
    operator_forced_calibration_override = [string]$OperatorForcedCalibrationOverride
    allow_operator_forced_calibration = [bool]$AllowOperatorForcedCalibrationOverride
    derived_surface_json = [string]$DerivedSurfaceJson
    automatic_reconstruction_status_path = $automaticReconstructionStatusPath
    automatic_reconstruction_root = $automaticReconstructionRoot
    sensor_frame_transform_id = $automaticReconstructionSensorFrameTransformId
    automatic_reconstruction_pointer_protocol = $automaticReconstructionPointerProtocol
    session_odom_local_status_path = $sessionOdomLocalStatusPath
    session_odom_local_root = $sessionOdomLocalRoot
    depth_anything_weights = [string]$DepthAnythingWeights
    depth_anything_weights_sha256 = [string]$DepthAnythingWeightsSha256
    depth_anything_device = $DepthAnythingDevice
    depth_anything_precision = $DepthAnythingPrecision
    depth_anything_backend = $DepthAnythingBackend
    depth_anything_model_name = $DepthAnythingModelName
    depth_anything_manifest = $defaultDepthAnythingManifest
    depth_anything_local_files_only = $true
    direct_rgb_texture = $true
}
$dashboardConfigJson = $dashboardConfig | ConvertTo-Json -Compress
$dashboardConfigHasher = [System.Security.Cryptography.SHA256]::Create()
try {
    $dashboardConfigBytes = $dashboardConfigHasher.ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes($dashboardConfigJson)
    )
} finally {
    $dashboardConfigHasher.Dispose()
}
$dashboardConfigSha256 = ([BitConverter]::ToString($dashboardConfigBytes)).Replace('-', '').ToLowerInvariant()

function Convert-ToWslPath([string]$Path) {
    if (-not $Path) { return '' }
    $resolved = [System.IO.Path]::GetFullPath($Path)
    return '/mnt/' + $resolved.Substring(0, 1).ToLowerInvariant() + ($resolved.Substring(2) -replace '\\', '/')
}

function Test-Receiver {
    try {
        $health = Invoke-RestMethod -Uri $receiverHealthUri -Proxy $null -TimeoutSec 3
        return $health.progressive_protocol -eq 'sha256-chunk-v1' -and `
            $health.transport -eq 'tailnet-direct-or-loopback' -and `
            $health.capacity_probe -eq $true
    } catch {
        return $false
    }
}

function Test-ReceiverLoopback {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Proxy $null -TimeoutSec 3
        return $health.progressive_protocol -eq 'sha256-chunk-v1' -and `
            $health.transport -eq 'tailnet-direct-or-loopback' -and `
            $health.capacity_probe -eq $true
    } catch {
        return $false
    }
}

function Test-Dashboard {
    try {
        $readiness = Invoke-RestMethod -Uri "http://127.0.0.1:$dashboardInternalPort/api/stack/readiness" -Proxy $null -TimeoutSec 3
        $derivedReady = $true
        $operatorOverrideReady = $true
        if ($DerivedSurfaceJson) {
            $derivedStatus = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$dashboardInternalPort/api/reconstruction/measured-surface/status" `
                -Proxy $null -TimeoutSec 3
            $derivedReady = (
                $derivedStatus.available -eq $true -and
                $derivedStatus.source_mode -eq 'DERIVED_READ_ONLY' -and
                $derivedStatus.read_only -eq $true
            )
        }
        if ($OperatorForcedCalibrationOverride) {
            $runtimeCalibration = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$dashboardInternalPort/api/runtime/calibration-status" `
                -Proxy $null -TimeoutSec 3
            $operatorOverrideReady = (
                $runtimeCalibration.calibration_id -eq $operatorOverrideCalibrationId -and
                $runtimeCalibration.formal_runtime_override -eq $true -and
                $runtimeCalibration.measurement_authority -eq $false -and
                $runtimeCalibration.runtime_authority -eq 'OPERATOR_FORCED_FORMAL' -and
                $runtimeCalibration.runtime_projection_authorized -eq $true -and
                @($runtimeCalibration.camera_from_lidar_row_major).Count -eq 16
            )
        }
        return (
            $readiness.ready -eq $true -and
            $readiness.dashboard_protocol -eq 'rgbd-producer-epoch-v2' -and
            $readiness.live_protocol -eq 3 -and
            $readiness.automatic_reconstruction_pointer_protocol -eq $automaticReconstructionPointerProtocol -and
            $readiness.runtime_config_sha256 -eq $dashboardConfigSha256 -and
            $readiness.capabilities.building_render_manifest -eq $true -and
            $readiness.capabilities.camera_mjpeg -eq $true -and
            $readiness.capabilities.field_acceptance_evidence -eq $true -and
            $readiness.capabilities.textured_tile_export -eq $true -and
            $readiness.capabilities.textured_building_export -eq $true -and
            $derivedReady -and
            $operatorOverrideReady
        )
    } catch {
        return $false
    }
}

$status = [ordered]@{
    source = 'PC STACK READY / WAITING FOR ROBOT SCAN'
    robot = @{ position = @(0.0, 0.0, 0.0); yaw = 0.0; speed_mps = 0.0 }
    mapping = @{ as_built_authority = $false; points = 0; coverage_m2 = 0.0; update_hz = 0.0; residual_cm = 0.0 }
    transport = @{ goodput_mbps = 0.0; latency_ms = 0; buffer_gib = 0.0 }
    bim = @{ state = 'unregistered'; support_groups = 0 }
}
[System.IO.File]::WriteAllText(
    $statusPath,
    ($status | ConvertTo-Json -Depth 4),
    (New-Object System.Text.UTF8Encoding($false))
)

if (-not (Test-Receiver) -or -not (Test-ReceiverLoopback)) {
    $portOccupied = Test-NetConnection -ComputerName $receiverHost -Port 8000 -InformationLevel Quiet
    if ($portOccupied) {
        $receiverPattern = [Regex]::Escape($receiverScript)
        $ownedReceiver = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -match $receiverPattern -and $_.CommandLine -match 'ssh_tunnel_receiver\.py'
        }
        if (-not $ownedReceiver) {
            throw 'Port 8000 is occupied by a process that is not the owned Go2 receiver.'
        }
        $ownedReceiver | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Start-Sleep -Milliseconds 500
    }
    $python = (Get-Command python -ErrorAction Stop).Source
    $receiverLog = Join-Path $logRoot 'tailnet-artifact-receiver.log'
    $receiverError = Join-Path $logRoot 'tailnet-artifact-receiver.err.log'
    $receiver = Start-Process -FilePath $python -ArgumentList @(
        $receiverScript, '--root', $inboxRoot, '--status-path', $transferStatusPath, '--port', '8000',
        '--bind', $tailnetIp,
        '--bind', '127.0.0.1',
        '--allow-client', $tailnetIp, '--allow-client', $Node,
        '--allow-client', '127.0.0.1', '--allow-client', '::1'
    ) -PassThru -RedirectStandardOutput $receiverLog -RedirectStandardError $receiverError -WindowStyle Hidden
    [System.IO.File]::WriteAllText((Join-Path $runtimeRoot 'receiver.pid'), [string]$receiver.Id)
    for ($attempt = 0; $attempt -lt 30 -and ((-not (Test-Receiver)) -or (-not (Test-ReceiverLoopback))); $attempt++) {
        Start-Sleep -Milliseconds 200
    }
    if ((-not (Test-Receiver)) -or (-not (Test-ReceiverLoopback))) { throw "PC receiver did not start on both Tailnet and loopback. See $receiverError" }
}

if ($Mode -eq 'full' -and $HighFidelityTransport -eq 'autodl') {
    $pullerPattern = [Regex]::Escape($bulkPullerScript)
    $puller = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match $pullerPattern -and $_.CommandLine -match [Regex]::Escape($BulkEndpoint)
    } | Select-Object -First 1
    if (-not $puller) {
        $python = (Get-Command python -ErrorAction Stop).Source
        $pullerLog = Join-Path $logRoot 'autodl-spool-puller.log'
        $pullerError = Join-Path $logRoot 'autodl-spool-puller.err.log'
        $puller = Start-Process -FilePath $python -ArgumentList @(
            $bulkPullerScript, '--endpoint', $BulkEndpoint, '--token-file', $BulkTokenFile,
            '--root', $inboxRoot, '--status-path', (Join-Path $dataRoot 'autodl-pull-status.json'),
            '--workers', "$BulkPullWorkers", '--interval', '1.0'
        ) -PassThru -RedirectStandardOutput $pullerLog -RedirectStandardError $pullerError -WindowStyle Hidden
        Start-Sleep -Milliseconds 750
        if ($puller.HasExited) { throw "AutoDL spool puller exited early. See $pullerError" }
    }
}

$linuxWorkspace = Convert-ToWslPath $workspaceRoot
$linuxInboxRoot = Convert-ToWslPath $inboxRoot
$linuxIngest = Convert-ToWslPath $ingestRoot
$linuxAutomaticReconstructionRoot = Convert-ToWslPath $automaticReconstructionRoot
$linuxAutomaticReconstructionStatusPath = Convert-ToWslPath $automaticReconstructionStatusPath
$linuxAutomaticReconstructionLinuxPidPath = Convert-ToWslPath $automaticReconstructionLinuxPidPath
$linuxSessionOdomLocalRoot = Convert-ToWslPath $sessionOdomLocalRoot
$linuxSessionOdomLocalStatusPath = Convert-ToWslPath $sessionOdomLocalStatusPath
$dashboardLinuxPidPath = Join-Path $runtimeRoot 'dashboard-linux.pid'
$linuxDashboardPidPath = Convert-ToWslPath $dashboardLinuxPidPath
if (-not (Test-Dashboard)) {
    $dashboardPortOccupied = Test-NetConnection -ComputerName 127.0.0.1 -Port $dashboardInternalPort -InformationLevel Quiet
    if ($dashboardPortOccupied) {
        if (-not $AllowDashboardRestart) {
            throw "The owned Go2 dashboard configuration differs. Preserve the running service by default; rerun with -AllowDashboardRestart to replace it."
        }
        $dashboardPidPath = Join-Path $runtimeRoot 'dashboard-wsl.pid'
        $ownedDashboardPid = if (Test-Path -LiteralPath $dashboardPidPath -PathType Leaf) {
            [int](Get-Content -Raw -LiteralPath $dashboardPidPath)
        } else { 0 }
        $ownedDashboard = if ($ownedDashboardPid -gt 0) {
            Get-CimInstance Win32_Process -Filter "ProcessId=$ownedDashboardPid" -ErrorAction SilentlyContinue
        } else { $null }
        $ownedLinuxDashboardPid = if (Test-Path -LiteralPath $dashboardLinuxPidPath -PathType Leaf) {
            [int](Get-Content -Raw -LiteralPath $dashboardLinuxPidPath)
        } else { 0 }
        if (
            $null -eq $ownedDashboard -or
            $ownedDashboard.Name -ne 'wsl.exe'
        ) {
            throw 'Port 8765 is occupied by a process that is not the owned Go2 dashboard.'
        }
        if ($ownedLinuxDashboardPid -le 0) {
            throw 'Owned Go2 dashboard has no Linux PID ownership record.'
        }
        $windowsCommandLineAvailable = -not [string]::IsNullOrWhiteSpace([string]$ownedDashboard.CommandLine)
        if ($windowsCommandLineAvailable -and (
            $ownedDashboard.CommandLine -notlike "*--cd $linuxWorkspace*" -or
            $ownedDashboard.CommandLine -notlike '*mapping/pc/start_dashboard_wsl.sh*'
        )) {
            throw 'Port 8765 is occupied by a process that is not the owned Go2 dashboard.'
        }
        & wsl.exe -d $Distro --cd $linuxWorkspace -- bash mapping/pc/start_dashboard_wsl.sh `
            verify-owned "$ownedLinuxDashboardPid" "$dashboardInternalPort"
        if ($LASTEXITCODE -ne 0) {
            throw 'dashboard Linux ownership does not match this worktree and process group.'
        }
        if (-not $windowsCommandLineAvailable) {
            Write-Output 'Windows command line is unavailable; accepted the independently verified WSL dashboard owner record.'
        }
        & wsl.exe -d $Distro --cd $linuxWorkspace -- bash mapping/pc/start_dashboard_wsl.sh `
            stop-owned "$ownedLinuxDashboardPid" "$dashboardInternalPort"
        if ($LASTEXITCODE -ne 0) {
            throw "Owned Go2 dashboard did not release its WSL listener on port $dashboardInternalPort."
        }
        Stop-Process -Id $ownedDashboardPid -Force -ErrorAction SilentlyContinue
        if (Test-NetConnection -ComputerName 127.0.0.1 -Port $dashboardInternalPort -InformationLevel Quiet) {
            throw "Owned Go2 dashboard stopped, but port $dashboardInternalPort is still occupied."
        }
    }
    $linuxStatus = Convert-ToWslPath $statusPath
    $linuxTransfer = Convert-ToWslPath $transferStatusPath
    $linuxLifecycle = Convert-ToWslPath $inboxRoot
    $linuxQuality = Convert-ToWslPath $qualityGatesPath
    $linuxCalibration = Convert-ToWslPath $CalibrationManifest
    $linuxEvidence = Convert-ToWslPath $SessionEvidence
    $linuxOperatorForcedCalibrationOverride = Convert-ToWslPath $OperatorForcedCalibrationOverride
    $linuxDerivedSurface = Convert-ToWslPath $DerivedSurfaceJson
    $dashboardLog = Join-Path $logRoot 'dashboard.log'
    $dashboardError = Join-Path $logRoot 'dashboard.err.log'
    $dashboardArgs = @(
        '-d', $Distro, '--cd', $linuxWorkspace, '--', 'env',
        "GO2_MAPPING_SESSION_STATUS=$linuxStatus",
        "GO2_MAPPING_TRANSFER_STATUS=$linuxTransfer",
        "GO2_MAPPING_INGEST_ROOT=$linuxIngest",
        "GO2_MAPPING_LIFECYCLE_ROOT=$linuxLifecycle",
        "GO2_MAPPING_QUALITY_GATES=$linuxQuality",
        "GO2_MAPPING_CALIBRATION_MANIFEST=$linuxCalibration",
        "GO2_MAPPING_SESSION_EVIDENCE=$linuxEvidence",
        "GO2_MAPPING_OPERATOR_FORCED_CALIBRATION_OVERRIDE=$linuxOperatorForcedCalibrationOverride",
        "GO2_MAPPING_ALLOW_OPERATOR_FORCED_CALIBRATION=$([int][bool]$AllowOperatorForcedCalibrationOverride)",
        "GO2_MAPPING_DERIVED_SURFACE_JSON=$linuxDerivedSurface",
        "GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_STATUS_PATH=$linuxAutomaticReconstructionStatusPath",
        "GO2_MAPPING_AUTOMATIC_RECONSTRUCTION_ROOT=$linuxAutomaticReconstructionRoot",
        "GO2_MAPPING_SESSION_ODOM_LOCAL_STATUS_PATH=$linuxSessionOdomLocalStatusPath",
        "GO2_MAPPING_SESSION_ODOM_LOCAL_ROOT=$linuxSessionOdomLocalRoot",
        "GO2_MAPPING_ROOM_SCAN_OPERATOR_TOKEN_FILE=$(Convert-ToWslPath $RoomScanOperatorTokenFile)",
        "GO2_MAPPING_SESSION_CONTROL_TOKEN_FILE=$(Convert-ToWslPath $SessionControlTokenFile)",
        "GO2_MAPPING_ROOM_SCAN_ROBOT_IP=$Node",
        "GO2_MAPPING_PC_TAILNET_IP=$tailnetIp",
        "GO2_MAPPING_SESSION_CONTROL_PORT=8788",
        "GO2_MAPPING_RECEIVER_HEALTH_URL=http://127.0.0.1:8000/health",
        "GO2_MAPPING_SENSOR_FRAME_TRANSFORM_ID=$automaticReconstructionSensorFrameTransformId",
        "GO2_MAPPING_DEPTH_ANYTHING_ENABLED=$([int]([bool]$DepthAnythingWeights))",
        "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_PATH=$(Convert-ToWslPath $DepthAnythingWeights)",
        "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_SHA256=$DepthAnythingWeightsSha256",
        "GO2_MAPPING_DEPTH_ANYTHING_DEVICE=$DepthAnythingDevice",
        "GO2_MAPPING_DEPTH_ANYTHING_PRECISION=$DepthAnythingPrecision",
        "GO2_MAPPING_DEPTH_ANYTHING_BACKEND=$DepthAnythingBackend",
        "GO2_MAPPING_DEPTH_ANYTHING_MODEL_NAME=$DepthAnythingModelName",
        'GO2_MAPPING_DIRECT_RGB_TEXTURE=1',
        'GO2_MAPPING_GEOMETRY_ONLY_CALIBRATION=1',
        "GO2_MAPPING_RUNTIME_CONFIG_SHA256=$dashboardConfigSha256",
        "GO2_MAPPING_DASHBOARD_PID_FILE=$linuxDashboardPidPath",
        "GO2_MAPPING_DASHBOARD_PORT=$dashboardInternalPort",
        'bash', 'mapping/pc/start_dashboard_wsl.sh'
    )
    $dashboard = Start-Process -FilePath wsl.exe -ArgumentList $dashboardArgs -PassThru `
        -RedirectStandardOutput $dashboardLog -RedirectStandardError $dashboardError -WindowStyle Hidden
    [System.IO.File]::WriteAllText((Join-Path $runtimeRoot 'dashboard-wsl.pid'), [string]$dashboard.Id)
    for ($attempt = 0; $attempt -lt 160 -and -not (Test-Dashboard); $attempt++) {
        Start-Sleep -Milliseconds 250
    }
    if (-not (Test-Dashboard)) { throw "PC dashboard/backend did not start. See $dashboardError" }
}

if ($IngestSessionId) {
    # The owned WSL launcher executes tools/run_go2_automatic_reconstruction.py.
    $workerPidPath = Join-Path $runtimeRoot 'automatic-reconstruction-wsl.pid'
    $workerLog = Join-Path $logRoot "automatic-reconstruction-$IngestSessionId.log"
    $workerError = Join-Path $logRoot "automatic-reconstruction-$IngestSessionId.err.log"
    $worker = $null
    $workerReady = $false
    if (Test-Path -LiteralPath $workerPidPath -PathType Leaf) {
        $ownedWorkerPid = [int](Get-Content -Raw -LiteralPath $workerPidPath)
        $ownedWorker = Get-CimInstance Win32_Process -Filter "ProcessId=$ownedWorkerPid" -ErrorAction SilentlyContinue
        if (
            $ownedWorker -and
            $ownedWorker.Name -eq 'wsl.exe' -and
            $ownedWorker.CommandLine -like "*--cd $linuxWorkspace*" -and
            $ownedWorker.CommandLine -like '*mapping/pc/start_session_reconstruction_wsl.sh*'
        ) {
            if ($ownedWorker.CommandLine -like "*$IngestSessionId*") {
                $worker = $ownedWorker
                $ownedLinuxWorkerPid = if (Test-Path -LiteralPath $automaticReconstructionLinuxPidPath -PathType Leaf) { [int](Get-Content -Raw -LiteralPath $automaticReconstructionLinuxPidPath) } else { 0 }
                if ($ownedLinuxWorkerPid -le 0) { throw 'Owned reconstruction worker has no Linux PID ownership record.' }
                & wsl.exe -d $Distro --cd $linuxWorkspace -- bash mapping/pc/start_session_reconstruction_wsl.sh verify-owned "$ownedLinuxWorkerPid" "$IngestSessionId"
                if ($LASTEXITCODE -ne 0) { throw 'Owned reconstruction worker does not match its session ownership record.' }
                $workerReady = $true
            } else {
                $ownedLinuxWorkerPid = if (Test-Path -LiteralPath $automaticReconstructionLinuxPidPath -PathType Leaf) { [int](Get-Content -Raw -LiteralPath $automaticReconstructionLinuxPidPath) } else { 0 }
                if ($ownedLinuxWorkerPid -le 0) { throw 'Owned reconstruction worker has no Linux PID ownership record.' }
                & wsl.exe -d $Distro --cd $linuxWorkspace -- bash mapping/pc/start_session_reconstruction_wsl.sh stop-owned "$ownedLinuxWorkerPid"
                if ($LASTEXITCODE -ne 0) { throw 'Owned reconstruction worker did not stop cleanly.' }
                Stop-Process -Id $ownedWorkerPid -Force -ErrorAction SilentlyContinue
            }
        } elseif ($ownedWorker) {
            throw 'Automatic reconstruction PID is occupied by a process that is not the owned session worker.'
        }
    }
    if (-not $worker) {
        $workerArgs = @(
            '-d', $Distro, '--cd', $linuxWorkspace, '--', 'env',
            "GO2_MAPPING_RECONSTRUCTION_PID_FILE=$linuxAutomaticReconstructionLinuxPidPath",
            'bash', 'mapping/pc/start_session_reconstruction_wsl.sh', 'run', $IngestSessionId,
            $linuxInboxRoot, $linuxAutomaticReconstructionRoot, $linuxAutomaticReconstructionStatusPath,
            $automaticReconstructionSensorFrameTransformId
            $ReconstructionSourceMode
        )
        $worker = Start-Process -FilePath wsl.exe -ArgumentList $workerArgs -PassThru `
            -RedirectStandardOutput $workerLog -RedirectStandardError $workerError -WindowStyle Hidden
        [System.IO.File]::WriteAllText($workerPidPath, [string]$worker.Id)
    }
    for ($attempt = 0; $attempt -lt 40 -and -not $workerReady; $attempt++) {
        $worker.Refresh()
        if ($worker.HasExited) { break }
        $ownedLinuxWorkerPid = if (Test-Path -LiteralPath $automaticReconstructionLinuxPidPath -PathType Leaf) { [int](Get-Content -Raw -LiteralPath $automaticReconstructionLinuxPidPath) } else { 0 }
        if ($ownedLinuxWorkerPid -gt 0) {
            & wsl.exe -d $Distro --cd $linuxWorkspace -- bash mapping/pc/start_session_reconstruction_wsl.sh verify-owned "$ownedLinuxWorkerPid" "$IngestSessionId"
            if ($LASTEXITCODE -eq 0) { $workerReady = $true; break }
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $workerReady) {
        if (-not $worker.HasExited) { Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue }
        throw "Automatic reconstruction worker did not become ready. See $workerError"
    }
    Write-Output "Automatic reconstruction: session=$IngestSessionId final_mode=high_fidelity_batch_reconstruction"
}

& tailscale.exe serve --bg --yes --tcp=8765 "tcp://127.0.0.1:$dashboardInternalPort"
if ($LASTEXITCODE -ne 0) { throw 'Tailnet dashboard publication failed.' }
$serveStatus = (& tailscale.exe serve status | Out-String)
$expectedServeSource = "tcp://${tailnetIp}:8765"
$expectedServeTarget = "tcp://127.0.0.1:$dashboardInternalPort"
if (
    $LASTEXITCODE -ne 0 -or
    $serveStatus -notmatch [Regex]::Escape($expectedServeSource) -or
    $serveStatus -notmatch [Regex]::Escape($expectedServeTarget)
) {
    throw 'Tailnet dashboard publication does not expose the expected TCP route.'
}

$liveEndpoint = "ws://${tailnetIp}:8765/ws/live/keyframe"
$liveHealthUri = "http://${tailnetIp}:8765/api/stack/readiness"

Write-Output 'GO2_PC_STACK_READY'
Write-Output "Dashboard: http://127.0.0.1:$dashboardInternalPort (Tailnet: http://${tailnetIp}:8765)"
Write-Output "High-fidelity inbox: $ingestRoot"
Write-Output "GO2_MAPPING_LIVE_ENDPOINT=$liveEndpoint"
Write-Output "Live preview transport: tailnet-direct endpoint=$liveEndpoint"
$highFidelityEndpoint = if ($HighFidelityTransport -eq 'autodl') { $BulkEndpoint } else { "http://${tailnetIp}:8000" }
Write-Output "High-fidelity transport: $HighFidelityTransport"
Write-Output "High-fidelity endpoint: $highFidelityEndpoint"
Write-Output "Robot start: bash /home/unitree/go2_mapping/current/tools/start_go2_robot_scan.sh start"
