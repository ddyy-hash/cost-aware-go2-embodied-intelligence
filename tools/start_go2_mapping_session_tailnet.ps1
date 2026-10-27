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
    [string]$CalibrationManifest,
    [string]$SessionEvidence,
    [string]$OperatorForcedCalibrationOverride,
    [string]$DepthAnythingWeights,
    [string]$DepthAnythingWeightsSha256,
    [ValidateSet('transformers', 'da3')]
    [string]$DepthAnythingBackend = 'transformers',
    [string]$DepthAnythingModelName = '',
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
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ControlTokenFile = if ($ControlTokenFile) {
    $ControlTokenFile
} else {
    Join-Path $workspaceRoot 'mapping-data\secrets\go2-session-control-token'
}

function Invoke-DirectRestMethod {
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Get', 'Post')]
        [string]$Method,
        [Parameter(Mandatory)]
        [string]$Uri,
        [hashtable]$Headers,
        [string]$ContentType,
        [string]$Body,
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
    if ($PSBoundParameters.ContainsKey('Body')) {
        $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
        if ($PSBoundParameters.ContainsKey('ContentType')) { $request.ContentType = $ContentType }
        $request.ContentLength = $bodyBytes.Length
        $requestStream = $request.GetRequestStream()
        try { $requestStream.Write($bodyBytes, 0, $bodyBytes.Length) }
        finally { $requestStream.Dispose() }
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

function Invoke-SessionStatusWithRetry {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,
        [Parameter(Mandatory)]
        [hashtable]$Headers,
        [ValidateRange(1, 8)]
        [int]$Attempts = 5
    )
    $delaySeconds = 1
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            return Invoke-DirectRestMethod -Method Get -Uri $Uri -Headers $Headers
        } catch {
            $statusCode = $null
            if ($null -ne $_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch { $statusCode = $null }
            }
            $transient = $null -eq $statusCode -or $statusCode -in @(408, 429, 500, 502, 503, 504)
            if (-not $transient) { throw }
            if ($attempt -eq $Attempts) {
                throw "Robot session status retry exhausted after $Attempts attempts: $($_.Exception.Message)"
            }
            Start-Sleep -Seconds $delaySeconds
            $delaySeconds = [Math]::Min(8, $delaySeconds * 2)
        }
    }
}

function Test-TailnetIPv4([string]$Address) {
    $parsed = $null
    if (-not [System.Net.IPAddress]::TryParse($Address, [ref]$parsed)) { return $false }
    $bytes = $parsed.GetAddressBytes()
    return $bytes.Length -eq 4 -and $bytes[0] -eq 100 -and $bytes[1] -ge 64 -and $bytes[1] -le 127
}

if (-not (Test-TailnetIPv4 $Node)) {
    throw 'Node must be the explicit robot Tailscale IPv4 address in 100.64.0.0/10.'
}
if (-not (Test-Path -LiteralPath $ControlTokenFile -PathType Leaf)) {
    throw "Session-control token file not found: $ControlTokenFile"
}
$token = (Get-Content -LiteralPath $ControlTokenFile -Raw).Trim()
if ($token.Length -lt 32 -or $token.Length -gt 512) {
    throw 'Session-control token must contain 32-512 characters.'
}
$headers = @{ Authorization = "Bearer $token" }
$controlBase = "http://${Node}:$ControlPort"

if ($Action -ne 'start') {
    if (-not $RequestId -or $RequestId -notmatch '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$') {
        throw 'A canonical lowercase UUIDv4 RequestId is required for status or stop.'
    }
    $uri = "$controlBase/v1/sessions/$RequestId"
    $result = if ($Action -eq 'stop') {
        Invoke-DirectRestMethod -Method Post -Uri "$uri/stop" -Headers $headers -ContentType 'application/json' -Body '{}'
    } else {
        Invoke-DirectRestMethod -Method Get -Uri $uri -Headers $headers
    }
    $result | ConvertTo-Json -Depth 8
    return
}

Get-Command tailscale.exe -ErrorAction Stop | Out-Null
$pcTailnetIp = (& tailscale.exe ip -4 | Select-Object -First 1).Trim()
if (-not (Test-TailnetIPv4 $pcTailnetIp)) {
    throw "Unable to determine the PC Tailscale IPv4 address: $pcTailnetIp"
}
$localLiveHealthUrl = 'http://127.0.0.1:8767/api/stack/readiness'
$localRuntimeCalibrationUrl = 'http://127.0.0.1:8767/api/runtime/calibration-status'
$liveHealthUrl = "http://${pcTailnetIp}:8765/api/stack/readiness"
$receiverHealthUrl = "http://${pcTailnetIp}:8000/health"
if ($UncalibratedCapture -and $OperatorForcedCalibrationOverride) {
    throw 'UncalibratedCapture and OperatorForcedCalibrationOverride are mutually exclusive.'
}
if ($UncalibratedCapture) {
    if ($Mode -ne 'full') { throw 'UncalibratedCapture requires full mode.' }
    if ($CalibrationManifest -or $SessionEvidence) {
        throw 'UncalibratedCapture uses only the versioned provisional capture manifest.'
    }
    if ($DepthAnythingWeights -or $DepthAnythingWeightsSha256) {
        throw 'UncalibratedCapture cannot use Depth Anything weights.'
    }
    $CalibrationManifest = Join-Path $workspaceRoot 'mapping\config\calibration_capture_manifest.json'
} elseif ($OperatorForcedCalibrationOverride) {
    if ($Mode -ne 'full') { throw 'OperatorForcedCalibrationOverride requires full mode.' }
    if (-not (Test-Path -LiteralPath $OperatorForcedCalibrationOverride -PathType Leaf)) {
        throw "OperatorForcedCalibrationOverride was not found: $OperatorForcedCalibrationOverride"
    }
    if (-not $CalibrationManifest) {
        $CalibrationManifest = Join-Path $workspaceRoot 'mapping\config\calibration_capture_manifest.json'
    }
} else {
    if (-not $CalibrationManifest -or -not (Test-Path -LiteralPath $CalibrationManifest -PathType Leaf)) {
        throw 'A measured calibration manifest is required.'
    }
    if (-not $GeometryOnlyCalibration -and (-not $SessionEvidence -or -not (Test-Path -LiteralPath $SessionEvidence -PathType Leaf))) {
        throw 'Session evidence is required.'
    }
}
$CalibrationManifest = (Resolve-Path -LiteralPath $CalibrationManifest).Path
if ($SessionEvidence) { $SessionEvidence = (Resolve-Path -LiteralPath $SessionEvidence).Path }
if ($OperatorForcedCalibrationOverride) {
    $OperatorForcedCalibrationOverride = (Resolve-Path -LiteralPath $OperatorForcedCalibrationOverride).Path
}
$calibration = Get-Content -LiteralPath $CalibrationManifest -Raw | ConvertFrom-Json
$calibrationId = [string]$calibration.calibration_id
if ($calibrationId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw 'Calibration manifest does not contain a canonical calibration_id.'
}
if ($UncalibratedCapture -or $OperatorForcedCalibrationOverride) {
    if (
        $calibration.status -ne 'provisional' -or
        $calibration.purpose -ne 'calibration_capture' -or
        $calibrationId -ne 'PROVISIONAL_AUTOCAL_PENDING_v1' -or
        $calibration.acceptance_authority -ne $false
    ) {
        throw 'The versioned provisional capture manifest is invalid.'
    }
    if ($OperatorForcedCalibrationOverride) {
        $operatorOverride = Get-Content -LiteralPath $OperatorForcedCalibrationOverride -Raw | ConvertFrom-Json
        if (
            $operatorOverride.schema_version -ne 1 -or
            $operatorOverride.kind -ne 'GO2_OPERATOR_FORCED_CALIBRATION_OVERRIDE' -or
            $operatorOverride.status -ne 'FORMAL_OPERATOR_FORCED_RUNTIME' -or
            $operatorOverride.calibration_id -ne $calibrationId -or
            $operatorOverride.formal_runtime_override -ne $true -or
            $operatorOverride.operator_acknowledged -ne $true -or
            $operatorOverride.measurement_authority -ne $false -or
            $operatorOverride.runtime_appearance_authority -ne $true
        ) {
            throw 'OperatorForcedCalibrationOverride does not match the formal runtime contract.'
        }
    }
} elseif ($calibration.status -ne 'measured') {
    throw 'Formal mapping requires calibration status measured.'
}
$calibrationSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $CalibrationManifest).Hash.ToLowerInvariant()
$evidenceSha = if ($SessionEvidence) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $SessionEvidence).Hash.ToLowerInvariant()
} else { $null }
$SessionId = if ($SessionId) { $SessionId } else { 'segment-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') }
if ($SessionId -notmatch '^segment-[0-9]{8}T[0-9]{6}Z$') {
    throw 'SessionId must match segment-YYYYMMDDTHHMMSSZ.'
}
if ($MissionId -and $MissionId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw 'MissionId must be canonical.'
}
if ($TaskId -and $TaskId -notmatch '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$') {
    throw 'TaskId must be canonical.'
}
if ($LeaseId -and $LeaseId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw 'LeaseId must be canonical.'
}

$pcStackArgs = @{
    Mode = $Mode
    Node = $Node
    Distro = $Distro
}
if (-not $UncalibratedCapture) {
    $pcStackArgs.CalibrationManifest = $CalibrationManifest
    if ($SessionEvidence) { $pcStackArgs.SessionEvidence = $SessionEvidence }
}
if ($CoordinateScope -eq 'SESSION_ODOM_LOCAL' -and $Mode -ne 'full') {
    throw 'SESSION_ODOM_LOCAL requires full mode.'
}
if ($Mode -eq 'full' -and $CoordinateScope -eq 'GLOBAL_MAP') {
    $pcStackArgs.IngestSessionId = $SessionId
    $pcStackArgs.ReconstructionSourceMode = 'RAW_LIDAR_WITH_RECORDED_LIO'
} elseif ($Mode -eq 'full' -and $CoordinateScope -eq 'SESSION_ODOM_LOCAL') {
    $pcStackArgs.IngestSessionId = $SessionId
    $pcStackArgs.ReconstructionSourceMode = 'RECORDED_LIO_SESSION_ODOM_CLOUD'
}
if ($OperatorForcedCalibrationOverride) {
    $pcStackArgs.OperatorForcedCalibrationOverride = $OperatorForcedCalibrationOverride
    $pcStackArgs.AllowOperatorForcedCalibrationOverride = $true
}
if ($DepthAnythingWeights) {
    if (-not $DepthAnythingWeightsSha256 -or $DepthAnythingWeightsSha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'Depth Anything requires an explicit lowercase weights SHA-256.'
    }
    if (-not (Test-Path -LiteralPath $DepthAnythingWeights)) {
        throw 'Depth Anything weights path does not exist.'
    }
    $pcStackArgs.DepthAnythingWeights = (Resolve-Path -LiteralPath $DepthAnythingWeights).Path
    $pcStackArgs.DepthAnythingWeightsSha256 = $DepthAnythingWeightsSha256
    $pcStackArgs.DepthAnythingBackend = $DepthAnythingBackend
    $pcStackArgs.DepthAnythingModelName = $DepthAnythingModelName
    $DepthAnythingWeights = $pcStackArgs.DepthAnythingWeights
}
& (Join-Path $PSScriptRoot 'start_go2_pc_stack.ps1') @pcStackArgs
if ($LASTEXITCODE -ne 0) { throw 'The local PC mapping stack failed to start.' }
$liveHealth = Invoke-DirectRestMethod -Method Get -Uri $localLiveHealthUrl
$receiverHealth = Invoke-DirectRestMethod -Method Get -Uri $receiverHealthUrl
if ($receiverHealth.progressive_protocol -ne 'sha256-chunk-v1' -or $receiverHealth.capacity_probe -ne $true) {
    throw 'The direct Tailnet receiver is not ready for verified high-fidelity capture.'
}
if (
    $liveHealth.ready -ne $true -or
    $liveHealth.live_protocol -ne 3 -or
    $liveHealth.capabilities.camera_mjpeg -ne $true -or
    $liveHealth.capabilities.building_render_manifest -ne $true
) {
    throw 'The local dashboard stack readiness contract is incomplete.'
}
if ($OperatorForcedCalibrationOverride) {
    $runtimeCalibration = Invoke-DirectRestMethod -Method Get -Uri $localRuntimeCalibrationUrl
    if (
        $runtimeCalibration.calibration_id -ne $calibrationId -or
        $runtimeCalibration.formal_runtime_override -ne $true -or
        $runtimeCalibration.measurement_authority -ne $false -or
        $runtimeCalibration.runtime_authority -ne 'OPERATOR_FORCED_FORMAL' -or
        $runtimeCalibration.runtime_projection_authorized -ne $true -or
        @($runtimeCalibration.camera_from_lidar_row_major).Count -ne 16
    ) {
        throw 'OPERATOR_FORCED_RUNTIME_NOT_ACTIVE: Dashboard did not load the requested formal operator override.'
    }
}

$preflight = Invoke-DirectRestMethod -Method Get -Uri "$controlBase/v1/preflight" -Headers $headers
if ($preflight.ready -ne $true) {
    throw "Robot Tailnet preflight is not ready: $($preflight.report)"
}

$RequestId = if ($RequestId) { $RequestId } else { [Guid]::NewGuid().ToString('D').ToLowerInvariant() }
if ($RequestId -notmatch '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$') {
    throw 'RequestId must be a canonical lowercase UUIDv4.'
}
$request = [ordered]@{
    request_id = $RequestId
    mode = $Mode
    duration_seconds = $DurationSeconds
    session_id = $SessionId
    calibration_id = $calibrationId
    calibration_sha256 = $calibrationSha
    pc_tailnet_ip = $pcTailnetIp
    coordinate_scope = $CoordinateScope
}
if ($MissionId) { $request.mission_id = $MissionId }
if ($TaskId) { $request.task_id = $TaskId }
if ($PSBoundParameters.ContainsKey('TaskRevision')) { $request.task_revision = $TaskRevision }
if ($LeaseId) { $request.lease_id = $LeaseId }
$requestJson = $request | ConvertTo-Json -Compress
$started = Invoke-DirectRestMethod -Method Post -Uri "$controlBase/v1/sessions" -Headers $headers -ContentType 'application/json' -Body $requestJson
if ($started.request_id -ne $RequestId -or $started.state -ne 'RUNNING') {
    throw 'Robot session control did not return the exact running request.'
}

$statusUri = "$controlBase/v1/sessions/$RequestId"
$status = Invoke-SessionStatusWithRetry -Uri $statusUri -Headers $headers
if ($status.request_id -ne $RequestId) { throw 'Robot session status identity changed.' }
if ($WaitForCompletion) {
    $deadline = [DateTime]::UtcNow.AddSeconds($DurationSeconds + $CompletionTimeoutSeconds)
    while ($status.state -eq 'RUNNING' -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 2
        $status = Invoke-SessionStatusWithRetry -Uri $statusUri -Headers $headers
        if ($status.request_id -ne $RequestId) { throw 'Robot session status identity changed.' }
    }
    if ($status.state -eq 'RUNNING') { throw 'Timed out waiting for the exact robot session request.' }
}

$context = [ordered]@{
    schema_version = 1
    evidence_kind = if ($UncalibratedCapture) { 'PHYSICAL_GO2_CALIBRATION_CAPTURE' } else { 'PHYSICAL_GO2_CAPTURE' }
    transport = 'TAILSCALE_DIRECT'
    acceptance_authority = -not $UncalibratedCapture
    formal_runtime_override = [bool]$OperatorForcedCalibrationOverride
    measurement_authority = -not $UncalibratedCapture -and -not $OperatorForcedCalibrationOverride
    geometry_only_calibration = [bool]$GeometryOnlyCalibration
    runtime_authority = if ($OperatorForcedCalibrationOverride) { 'OPERATOR_FORCED_FORMAL_RUNTIME' } elseif ($UncalibratedCapture) { 'NONE' } else { 'MEASURED' }
    capture_purpose = if ($UncalibratedCapture) { 'calibration_capture' } else { 'mapping' }
    control_uri = $controlBase
    request_id = $RequestId
    session_id = $SessionId
    mission_id = $MissionId
    task_id = $TaskId
    task_revision = if ($PSBoundParameters.ContainsKey('TaskRevision')) { $TaskRevision } else { $null }
    lease_id = $LeaseId
    mode = $Mode
    coordinate_scope = $CoordinateScope
    duration_seconds = $DurationSeconds
    pc_tailnet_ip = $pcTailnetIp
    robot_tailnet_ip = $Node
    calibration_id = $calibrationId
    calibration_status = if ($OperatorForcedCalibrationOverride) { 'FORMAL_OPERATOR_FORCED_RUNTIME' } else { [string]$calibration.status }
    calibration_manifest_status = [string]$calibration.status
    calibration_sha256 = $calibrationSha
    operator_forced_calibration_override_path = $OperatorForcedCalibrationOverride
    operator_forced_calibration_override_sha256 = if ($OperatorForcedCalibrationOverride) { (Get-FileHash -Algorithm SHA256 -LiteralPath $OperatorForcedCalibrationOverride).Hash.ToLowerInvariant() } else { $null }
    session_evidence_sha256 = $evidenceSha
    calibration_manifest_path = $CalibrationManifest
    session_evidence_path = $SessionEvidence
    dashboard_url = "http://${pcTailnetIp}:8765"
    dashboard_pid_path = (Join-Path $workspaceRoot 'mapping-data\pc-stack\dashboard-wsl.pid')
    owned_dashboard_pid = $null
    worktree_root = $workspaceRoot
    distro = $Distro
    depth_anything_weights_path = $DepthAnythingWeights
    depth_anything_weights_sha256 = $DepthAnythingWeightsSha256
    state = $status.state
    pid = $status.pid
    synthetic = $false
}
$dashboardPidPath = [string]$context.dashboard_pid_path
if (Test-Path -LiteralPath $dashboardPidPath -PathType Leaf) {
    $context.owned_dashboard_pid = [int](Get-Content -Raw -LiteralPath $dashboardPidPath)
}
if (-not $context.owned_dashboard_pid) {
    throw 'The PC stack did not publish its owned dashboard PID.'
}
if ($AcceptanceContextPath) {
    $contextPath = [System.IO.Path]::GetFullPath($AcceptanceContextPath)
    $contextDirectory = Split-Path -Parent $contextPath
    New-Item -ItemType Directory -Force -Path $contextDirectory | Out-Null
    $temporary = "$contextPath.$([Guid]::NewGuid().ToString('N')).partial"
    try {
        $contextJson = ($context | ConvertTo-Json -Depth 8 -Compress) + "`n"
        [System.IO.File]::WriteAllText($temporary, $contextJson, (New-Object System.Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temporary -Destination $contextPath -Force
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}
$context | ConvertTo-Json -Depth 8
Write-Output "MAPPING_SESSION_CONTROLLED transport=TAILSCALE_DIRECT request=$RequestId session=$SessionId state=$($status.state)"
