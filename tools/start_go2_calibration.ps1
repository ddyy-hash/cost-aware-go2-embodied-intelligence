[CmdletBinding()]
param(
    [ValidateRange(120, 86400)]
    [int]$DurationSeconds = 300,
    [ValidateRange(10, 300)]
    [int]$LiveReadinessTimeoutSeconds = 90,
    [string]$Distro = 'Ubuntu-24.04',
    [switch]$NoOpenDashboard
)

$ErrorActionPreference = 'Stop'
$script:WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:SessionEntrypoint = Join-Path $PSScriptRoot 'start_go2_mapping_session.ps1'
$script:RobotTailnetIp = '192.0.2.11'
$script:RobotControlPort = 8788
$script:PcTailnetIp = '192.0.2.10'
$script:DashboardBaseUrl = 'http://192.0.2.10:8765'
$script:DashboardMapUrl = 'http://192.0.2.10:8765/?view=map'
$script:CameraFreshnessMs = 5000

function Get-CalibrationUtcNow {
    return [DateTimeOffset]::UtcNow
}

function Wait-CalibrationPoll {
    Start-Sleep -Milliseconds 500
}

function New-CalibrationLaunchIdentity {
    $requestId = [Guid]::NewGuid().ToString('D').ToLowerInvariant()
    $sessionId = 'segment-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $launchKey = "$sessionId-$requestId"
    $directory = Join-Path $script:WorkspaceRoot "mapping-data\calibration-startup\$launchKey"
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    return [pscustomobject]@{
        request_id = $requestId
        session_id = $sessionId
        context_path = Join-Path $directory 'capture-context.json'
    }
}

function Invoke-CalibrationSessionEntrypoint {
    [CmdletBinding()]
    param(
        [ValidateSet('start', 'status')]
        [string]$Action,
        [string]$RequestId,
        [string]$SessionId,
        [int]$DurationSeconds,
        [string]$Distro,
        [string]$AcceptanceContextPath
    )

    $arguments = @{
        Action = $Action
        RequestId = $RequestId
        Node = $script:RobotTailnetIp
        ControlPort = $script:RobotControlPort
        Distro = $Distro
    }
    if ($Action -eq 'start') {
        $arguments.Mode = 'full'
        # Calibration observes one continuous local sensor/LIO session; it is
        # not a room or building map and must not require a global map frame.
        $arguments.CoordinateScope = 'SESSION_ODOM_LOCAL'
        $arguments.DurationSeconds = $DurationSeconds
        $arguments.SessionId = $SessionId
        $arguments.AcceptanceContextPath = $AcceptanceContextPath
        $arguments.UncalibratedCapture = $true
    }

    $output = & $script:SessionEntrypoint @arguments
    if (-not $?) {
        throw "SESSION_ENTRYPOINT_FAILED action=$Action request=$RequestId session=$SessionId"
    }
    return ($output -join "`n")
}

function Read-CalibrationAcceptanceContext {
    param(
        [string]$Path,
        [string]$RequestId,
        [string]$SessionId
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "ACCEPTANCE_CONTEXT_MISMATCH request=$RequestId session=$SessionId reason=missing"
    }
    try {
        return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    } catch {
        throw "ACCEPTANCE_CONTEXT_MISMATCH request=$RequestId session=$SessionId reason=invalid_json"
    }
}

function Assert-CalibrationAcceptanceContext {
    param(
        $Context,
        [string]$RequestId,
        [string]$SessionId,
        [int]$ExpectedDurationSeconds
    )

    $mismatches = New-Object System.Collections.Generic.List[string]
    if ([int]$Context.schema_version -ne 1) { $mismatches.Add('schema_version') }
    if ([string]$Context.request_id -cne $RequestId) { $mismatches.Add('request_id') }
    if ([string]$Context.session_id -cne $SessionId) { $mismatches.Add('session_id') }
    if ([string]$Context.state -cne 'RUNNING') { $mismatches.Add('state') }
    if ([string]$Context.mode -cne 'full') { $mismatches.Add('mode') }
    if ([int]$Context.duration_seconds -ne $ExpectedDurationSeconds) { $mismatches.Add('duration_seconds') }
    if ([string]$Context.capture_purpose -cne 'calibration_capture') { $mismatches.Add('capture_purpose') }
    if ([string]$Context.calibration_id -cne 'PROVISIONAL_AUTOCAL_PENDING_v1') {
        $mismatches.Add('calibration_id')
    }
    if ([string]$Context.calibration_status -cne 'provisional') { $mismatches.Add('calibration_status') }
    if ($Context.acceptance_authority -ne $false) { $mismatches.Add('acceptance_authority') }
    if ([string]$Context.pc_tailnet_ip -cne $script:PcTailnetIp) { $mismatches.Add('pc_tailnet_ip') }
    if ([string]$Context.robot_tailnet_ip -cne $script:RobotTailnetIp) { $mismatches.Add('robot_tailnet_ip') }

    if ($mismatches.Count -gt 0) {
        $fields = $mismatches -join ','
        throw "ACCEPTANCE_CONTEXT_MISMATCH request=$RequestId session=$SessionId fields=$fields"
    }
}

function Get-FormalDashboardJson {
    param([string]$Path)

    $allowedPaths = @(
        '/api/stack/readiness',
        '/api/camera/status',
        '/api/calibration/progress'
    )
    if ($allowedPaths -cnotcontains $Path) {
        throw "DASHBOARD_PATH_NOT_ALLOWED path=$Path"
    }

    $request = [System.Net.HttpWebRequest][System.Net.WebRequest]::Create("$script:DashboardBaseUrl$Path")
    $request.Method = 'GET'
    $request.Proxy = $null
    $request.Timeout = 5000
    $request.ReadWriteTimeout = 5000
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

function Test-CalibrationStackReady {
    param($Status)

    if ($null -eq $Status -or $null -eq $Status.capabilities) { return $false }
    return (
        $Status.ready -eq $true -and
        [int]$Status.live_protocol -eq 3 -and
        $Status.capabilities.camera_mjpeg -eq $true
    )
}

function Test-CalibrationCameraReady {
    param($Status)

    if ($null -eq $Status) { return $false }
    $names = @($Status.PSObject.Properties.Name)
    foreach ($required in @('state', 'sequence', 'age_ms', 'preview_fps', 'distance_measurement_ready')) {
        if ($names -notcontains $required -or $null -eq $Status.$required) {
            return $false
        }
    }

    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $styles = [System.Globalization.NumberStyles]::Float
    $sequence = 0L
    $ageMs = 0.0
    $previewFps = 0.0
    $sequenceText = [Convert]::ToString($Status.sequence, $culture)
    $ageText = [Convert]::ToString($Status.age_ms, $culture)
    $fpsText = [Convert]::ToString($Status.preview_fps, $culture)
    if (-not [long]::TryParse($sequenceText, [System.Globalization.NumberStyles]::Integer, $culture, [ref]$sequence)) {
        return $false
    }
    if (-not [double]::TryParse($ageText, $styles, $culture, [ref]$ageMs)) { return $false }
    if (-not [double]::TryParse($fpsText, $styles, $culture, [ref]$previewFps)) { return $false }

    return (
        [string]$Status.state -like 'LIVE_*' -and
        $sequence -ge 0 -and
        $ageMs -ge 0 -and
        $ageMs -le $script:CameraFreshnessMs -and
        $previewFps -gt 0 -and
        $Status.distance_measurement_ready -eq $true
    )
}

function Test-CalibrationProgressReady {
    param($Status)

    if ($null -eq $Status) { return $false }
    if (@($Status.PSObject.Properties.Name) -notcontains 'active_slot') { return $false }
    $validSlots = @('estimation_1', 'estimation_2', 'estimation_3', 'holdout')
    return ($validSlots -ccontains [string]$Status.active_slot)
}

function New-CalibrationTimeoutMessage {
    param(
        [string]$RequestId,
        [string]$SessionId,
        [string]$Phase,
        [string]$LastBlocker
    )

    $statusCommand = ".\tools\start_go2_mapping_session.ps1 -Action status -RequestId $RequestId"
    $stopCommand = ".\tools\start_go2_mapping_session.ps1 -Action stop -RequestId $RequestId"
    return (
        "CALIBRATION_READINESS_TIMEOUT phase=$Phase request=$RequestId session=$SessionId " +
        "last_blocker=$LastBlocker status_command=$statusCommand stop_command=$stopCommand"
    )
}

function Assert-CalibrationBeforeDeadline {
    param(
        [DateTimeOffset]$Deadline,
        [string]$RequestId,
        [string]$SessionId,
        [string]$Phase,
        [string]$LastBlocker
    )

    if ((Get-CalibrationUtcNow) -ge $Deadline) {
        throw (New-CalibrationTimeoutMessage `
            -RequestId $RequestId `
            -SessionId $SessionId `
            -Phase $Phase `
            -LastBlocker $LastBlocker)
    }
}

function Open-FormalDashboard {
    param([string]$Url)

    Start-Process -FilePath $Url | Out-Null
}

function Invoke-Go2CalibrationStartup {
    [CmdletBinding()]
    param(
        [ValidateRange(120, 86400)]
        [int]$DurationSeconds = 300,
        [ValidateRange(10, 300)]
        [int]$LiveReadinessTimeoutSeconds = 90,
        [string]$Distro = 'Ubuntu-24.04',
        [switch]$NoOpenDashboard
    )

    if ($LiveReadinessTimeoutSeconds -gt ($DurationSeconds - 30)) {
        throw 'LiveReadinessTimeoutSeconds must leave at least 30 seconds in the requested session.'
    }

    $launch = New-CalibrationLaunchIdentity
    $requestId = [string]$launch.request_id
    $sessionId = [string]$launch.session_id
    $contextPath = [string]$launch.context_path
    Write-Output "CALIBRATION_STARTING request=$requestId session=$sessionId duration_seconds=$DurationSeconds"

    $startedAt = Get-CalibrationUtcNow
    $deadline = $startedAt.AddSeconds($LiveReadinessTimeoutSeconds)
    if ((Get-CalibrationUtcNow) -ge $deadline) {
        throw "CALIBRATION_READINESS_TIMEOUT phase=pre_start request=$requestId session=$sessionId"
    }

    try {
        Invoke-CalibrationSessionEntrypoint `
            -Action start `
            -RequestId $requestId `
            -SessionId $sessionId `
            -DurationSeconds $DurationSeconds `
            -Distro $Distro `
            -AcceptanceContextPath $contextPath | Out-Null
    } catch {
        throw "SESSION_START_FAILED request=$requestId session=$sessionId cause=$($_.Exception.Message)"
    }

    $context = Read-CalibrationAcceptanceContext `
        -Path $contextPath `
        -RequestId $requestId `
        -SessionId $sessionId
    Assert-CalibrationAcceptanceContext `
        -Context $context `
        -RequestId $requestId `
        -SessionId $sessionId `
        -ExpectedDurationSeconds $DurationSeconds

    $firstCameraSequence = $null
    $readyCamera = $null
    $readyProgress = $null
    $readySessionStatus = $null
    $lastBlocker = 'awaiting_stack'

    while ($null -eq $readyCamera) {
        Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'before_stack' $lastBlocker
        try {
            $stack = Get-FormalDashboardJson -Path '/api/stack/readiness'
            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'after_stack' $lastBlocker
            if (-not (Test-CalibrationStackReady $stack)) {
                $lastBlocker = 'stack_not_ready'
                Wait-CalibrationPoll
                continue
            }

            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'before_camera' $lastBlocker
            $camera = Get-FormalDashboardJson -Path '/api/camera/status'
            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'after_camera' $lastBlocker
            if (-not (Test-CalibrationCameraReady $camera)) {
                $lastBlocker = 'camera_not_ready'
                Wait-CalibrationPoll
                continue
            }

            $cameraSequence = [long]$camera.sequence
            if ($null -eq $firstCameraSequence) {
                $firstCameraSequence = $cameraSequence
                $lastBlocker = 'camera_sequence_not_advanced'
                Wait-CalibrationPoll
                continue
            }
            if ($cameraSequence -le [long]$firstCameraSequence) {
                $lastBlocker = 'camera_sequence_not_advanced'
                Wait-CalibrationPoll
                continue
            }

            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'before_progress' $lastBlocker
            $progress = Get-FormalDashboardJson -Path '/api/calibration/progress'
            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'after_progress' $lastBlocker
            if (-not (Test-CalibrationProgressReady $progress)) {
                $lastBlocker = 'calibration_progress_invalid'
                Wait-CalibrationPoll
                continue
            }

            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'before_session_status' $lastBlocker
            $statusJson = Invoke-CalibrationSessionEntrypoint `
                -Action status `
                -RequestId $requestId `
                -SessionId $sessionId `
                -DurationSeconds $DurationSeconds `
                -Distro $Distro `
                -AcceptanceContextPath $contextPath
            $sessionStatus = $statusJson | ConvertFrom-Json
            Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'after_session_status' $lastBlocker
            if (
                [string]$sessionStatus.request_id -cne $requestId -or
                $null -eq $sessionStatus.request -or
                [string]$sessionStatus.request.session_id -cne $sessionId -or
                [string]$sessionStatus.state -cne 'RUNNING'
            ) {
                throw "SESSION_NOT_RUNNING request=$requestId session=$sessionId state=$($sessionStatus.state)"
            }
            if ([string]$sessionStatus.phase -cne 'RECORDING') {
                $lastBlocker = 'raw_recorder_not_recording'
                Wait-CalibrationPoll
                continue
            }

            $readyCamera = $camera
            $readyProgress = $progress
            $readySessionStatus = $sessionStatus
        } catch {
            if ($_.Exception.Message -like 'CALIBRATION_READINESS_TIMEOUT*') { throw }
            if ($_.Exception.Message -like 'SESSION_NOT_RUNNING*') { throw }
            $lastBlocker = 'dashboard_request_failed'
            Wait-CalibrationPoll
        }
    }

    $sessionStatus = $readySessionStatus

    if (-not $NoOpenDashboard) {
        try {
            Open-FormalDashboard -Url $script:DashboardMapUrl
        } catch {
            Write-Warning "Dashboard could not be opened automatically: $script:DashboardMapUrl"
        }
    }
    Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'after_browser' $lastBlocker

    $result = [ordered]@{
        schema_version = 1
        state = 'READY_FOR_CALIBRATION_CAPTURE'
        request_id = $requestId
        session_id = $sessionId
        active_slot = [string]$readyProgress.active_slot
        camera_state = [string]$readyCamera.state
        camera_sequence = [long]$readyCamera.sequence
        preview_fps = [double]$readyCamera.preview_fps
        distance_measurement_ready = $true
        acceptance_authority = $false
        dashboard_url = $script:DashboardMapUrl
    }
    Assert-CalibrationBeforeDeadline $deadline $requestId $sessionId 'before_ready_marker' $lastBlocker
    $result | ConvertTo-Json -Depth 4
    Write-Output (
        "READY_FOR_CALIBRATION_CAPTURE request=$requestId session=$sessionId " +
        "slot=$($result.active_slot) camera_state=$($result.camera_state) " +
        "camera_sequence=$($result.camera_sequence) preview_fps=$($result.preview_fps) " +
        'distance_measurement_ready=true acceptance_authority=false'
    )
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        Invoke-Go2CalibrationStartup @PSBoundParameters
    } catch {
        Write-Error $_.Exception.Message
        exit 1
    }
}
