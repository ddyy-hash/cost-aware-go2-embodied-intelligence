[CmdletBinding()]
param(
    [string]$Node = 'go2edu.tailb73f8d.ts.net',
    [string]$UserName = 'unitree',
    [string]$PrivateKeyFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$BulkTokenFile,
    [string]$SessionControlTokenFile,
    [string]$PreparedArchive,
    [string]$RobotFactorySerial = '',
    [string]$RobotTailnetIp = '192.0.2.11',
    [string]$AllowedPcTailnetIp,
    [ValidateRange(1, 65535)]
    [int]$SessionControlPort = 8788,
    [bool]$UseTailscaleProxy = $true
)

$ErrorActionPreference = 'Stop'

function Invoke-DirectRestMethod {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,
        [hashtable]$Headers,
        [ValidateRange(1, 300)]
        [int]$TimeoutSec = 20
    )
    $request = [System.Net.HttpWebRequest][System.Net.WebRequest]::Create($Uri)
    $request.Method = 'GET'
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

if (-not (Test-Path -LiteralPath $PrivateKeyFile)) {
    throw "Private key not found: $PrivateKeyFile"
}
$sshTransportOptions = @()
if ($UseTailscaleProxy) {
    Get-Command tailscale.exe -ErrorAction Stop | Out-Null
    # Tailscale userspace proxy keeps SSH functional when DERP cannot carry a
    # direct TCP/22 session. It still uses the robot's normal SSH host key.
    $sshTransportOptions = @('-o', 'ProxyCommand=tailscale.exe nc %h %p')
}

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$SessionControlTokenFile = if ($SessionControlTokenFile) {
    $SessionControlTokenFile
} else {
    Join-Path $workspaceRoot 'mapping-data\secrets\go2-session-control-token'
}
if (-not $AllowedPcTailnetIp) {
    Get-Command tailscale.exe -ErrorAction Stop | Out-Null
    $AllowedPcTailnetIp = (& tailscale.exe ip -4 | Select-Object -First 1).Trim()
}
function Test-TailnetIPv4([string]$Address) {
    $parsed = $null
    if (-not [System.Net.IPAddress]::TryParse($Address, [ref]$parsed)) { return $false }
    $bytes = $parsed.GetAddressBytes()
    return $bytes.Length -eq 4 -and $bytes[0] -eq 100 -and $bytes[1] -ge 64 -and $bytes[1] -le 127
}
if (-not (Test-TailnetIPv4 $RobotTailnetIp) -or -not (Test-TailnetIPv4 $AllowedPcTailnetIp)) {
    throw 'RobotTailnetIp and AllowedPcTailnetIp must be explicit Tailscale IPv4 addresses.'
}
if ($RobotFactorySerial -and $RobotFactorySerial -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{3,127}$') {
    throw 'RobotFactorySerial must contain 4-128 ASCII letters, digits, dot, underscore, or hyphen.'
}
if (-not (Test-Path -LiteralPath $SessionControlTokenFile -PathType Leaf)) {
    $tokenDirectory = Split-Path -Parent ([System.IO.Path]::GetFullPath($SessionControlTokenFile))
    New-Item -ItemType Directory -Force -Path $tokenDirectory | Out-Null
    $tokenBytes = New-Object byte[] 32
    $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $random.GetBytes($tokenBytes)
    } finally {
        $random.Dispose()
    }
    $newToken = [Convert]::ToBase64String($tokenBytes)
    [System.IO.File]::WriteAllText(
        [System.IO.Path]::GetFullPath($SessionControlTokenFile),
        $newToken + "`n",
        (New-Object System.Text.UTF8Encoding($false))
    )
}
$robotTools = @(
    'verify_go2_mapping_robot.sh',
    'install_go2_wifi_reconnect.sh',
    'install_go2_live_websocket_dependency.sh',
    'start_go2_mapping_capture.sh',
    'start_go2_pgtt_heightmap.sh',
    'run_go2_pgtt_readonly.sh',
    'install_go2_pgtt_heightmap.sh',
    'start_go2_realsense_sensor.sh',
    'install_go2_realsense_sensor.sh',
    'go2-mapping-lowstate-imu.service',
    'run_go2_calibration_preflight.sh',
    'write_go2_capture_runtime_manifest.sh',
    'stream_go2_high_fidelity_queue.sh',
    'upload_go2_segment_via_tunnel.sh',
    'build_go2_lowstate_imu_bridge.sh',
    'start_go2_lowstate_imu_bridge.sh',
    'build_go2_native_lowstate_imu_reader.sh',
    'start_go2_unitree_lowstate_imu_bridge.sh',
    'start_go2_lio_preview.sh',
    'go2-mapping-lio-preview.service',
    'start_go2_live_preview_stream.sh',
    'start_go2_robot_scan.sh',
    'install_go2_tailnet_session_control.sh',
    'go2-mapping-session-control.service',
    'build_go2_native_pointcloud_reader.sh',
    'start_go2_native_pointcloud_bridge.sh',
    'start_go2_unitree_pointcloud_bridge.sh',
    'build_go2_native_pointcloud_reader.sh',
    'run_go2_mapping_session.sh',
    'start_go2_nav_bridge.sh',
    'go2-mapping-nav-bridge.service',
    'go2-mapping-realsense-sensor.service',
    'go2-mapping-pointcloud-bridge.service',
    'go2-mapping-pgtt-heightmap.service',
    'start_go2_pgtt_terrain_router.sh',
    'go2-mapping-pgtt-terrain-router.service',
    'install_go2_pgtt_terrain_router.sh',
    'build_go2_motion_mode_helper.sh',
    'query_go2_motion_mode.py',
    'build_go2_vendor_sport_probe.sh',
    'query_go2_vendor_sport.py',
    'build_go2_vendor_sport_adapter.sh',
    'build_go2_lowstate_motor_probe.sh',
    'build_go2_vendor_lease_probe.sh',
    'build_go2_native_terrain_executor.sh',
    'run_go2_pgtt_hardware.sh',
    'install_go2_nav_bridge.sh',
    'diagnose_go2_lidar_network.sh',
    'verify_go2_nav2_runtime.sh',
    'verify_go2_mapping_release.sh',
    'install_go2_mapping_python_runtime.sh'
    ,'install_go2_sdk2_python.sh'
    ,'install_go2_pointcloud_bridge.sh'
)
$requiredPaths = @(
    (Join-Path $workspaceRoot 'mapping')
) + ($robotTools | ForEach-Object { Join-Path $workspaceRoot "tools\$_" })
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required mapping release path is missing: $path"
    }
}

$usePreparedArchive = -not [string]::IsNullOrWhiteSpace($PreparedArchive)
$stagingRoot = $null
$archiveFile = $null
$cleanupArchive = $false
if ($usePreparedArchive) {
    if (-not (Test-Path -LiteralPath $PreparedArchive -PathType Leaf)) {
        throw "Prepared robot archive not found: $PreparedArchive"
    }
    $archiveFile = (Resolve-Path -LiteralPath $PreparedArchive).Path
} else {
    $stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("go2-mapping-release-" + [Guid]::NewGuid().ToString('N'))
    $archiveFile = "$stagingRoot.tar.gz"
    $cleanupArchive = $true
}
try {
    if (-not $usePreparedArchive) {
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $stagingRoot 'tools') | Out-Null
    foreach ($toolName in $robotTools) {
        Copy-Item -LiteralPath (Join-Path $workspaceRoot "tools\$toolName") -Destination (Join-Path $stagingRoot 'tools')
    }
    # A robot release must not include PC dashboards, Node caches, or neural
    # dependencies. Ship only robot runtime code and shared runtime profiles.
    $mappingStage = Join-Path $stagingRoot 'mapping'
    New-Item -ItemType Directory -Path $mappingStage | Out-Null
    foreach ($relativePath in @('robot', 'config')) {
        Copy-Item -LiteralPath (Join-Path $workspaceRoot "mapping\$relativePath") -Destination $mappingStage -Recurse
    }
    Get-ChildItem -LiteralPath $stagingRoot -Directory -Filter '__pycache__' -Recurse | Remove-Item -Recurse -Force
    $robotRuntimeManifest = Join-Path $stagingRoot 'mapping\config\robot_runtime_manifest.json'
    if (-not (Test-Path -LiteralPath $robotRuntimeManifest -PathType Leaf) -or
        (Get-Item -LiteralPath $robotRuntimeManifest).Length -le 0) {
        throw 'Robot runtime manifest is empty or missing.'
    }
    $wheelhouseSource = Join-Path $workspaceRoot 'output\robot-release\wheelhouse-aarch64-cp38'
    if (-not (Test-Path -LiteralPath $wheelhouseSource -PathType Container)) { throw "Robot wheelhouse is missing: $wheelhouseSource" }
    Copy-Item -LiteralPath $wheelhouseSource -Destination (Join-Path $stagingRoot 'mapping\robot') -Recurse
    $sdkVendorSource = Join-Path $workspaceRoot 'output\robot-release\sdk-vendor'
    if (-not (Test-Path -LiteralPath $sdkVendorSource -PathType Container)) { throw "SDK2 Python vendor archive directory is missing: $sdkVendorSource" }
    New-Item -ItemType Directory -Force -Path (Join-Path $stagingRoot 'mapping\robot\vendor') | Out-Null
    Copy-Item -LiteralPath (Join-Path $sdkVendorSource 'unitree_sdk2_python-65691c8.tar.gz') -Destination (Join-Path $stagingRoot 'mapping\robot\vendor')
    Copy-Item -LiteralPath (Join-Path $sdkVendorSource 'cyclonedds-0.10.2.tar.gz') -Destination (Join-Path $stagingRoot 'mapping\robot\vendor')
    Copy-Item -LiteralPath (Join-Path $sdkVendorSource 'typing_extensions-4.7.1-py3-none-any.whl') -Destination (Join-Path $stagingRoot 'mapping\robot\vendor')
    $pgttPolicySourceRoot = Join-Path $workspaceRoot 'output\robot-release\pgtt-go2'
    $pgttPolicyLevels = @('03', '07', '10', '13', '17', '20')
    foreach ($level in $pgttPolicyLevels) {
        $pgttPolicySource = Join-Path $pgttPolicySourceRoot "policy_level${level}_run0.npz"
        if (-not (Test-Path -LiteralPath $pgttPolicySource -PathType Leaf)) { throw "PGTT NumPy policy is missing: $pgttPolicySource" }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $stagingRoot 'mapping\robot\pgtt-go2') | Out-Null
    foreach ($level in $pgttPolicyLevels) {
        Copy-Item -LiteralPath (Join-Path $pgttPolicySourceRoot "policy_level${level}_run0.npz") -Destination (Join-Path $stagingRoot 'mapping\robot\pgtt-go2')
    }

    $requiredRobotPythonEntrypoints = @(
        'mapping\robot\go2_mapping_lio\go2_mapping_lio\session_control.py',
        'mapping\robot\go2_mapping_lio\go2_mapping_lio\map_frame_cloud.py',
        'mapping\robot\go2_mapping_lio\go2_mapping_lio\map_frame_cloud_publisher.py',
        'mapping\robot\go2_mapping_lio\go2_mapping_lio\ros_topic_liveness.py',
        'mapping\robot\go2_mapping_nav_bridge\go2_mapping_nav_bridge\navigation_bridge.py',
        'mapping\robot\go2_mapping_nav_bridge\go2_mapping_nav_bridge\navigation_executor.py',
        'mapping\robot\go2_mapping_nav_bridge\go2_mapping_nav_bridge\nav2_params.py'
    )
    foreach ($relativePath in $requiredRobotPythonEntrypoints) {
        $stagedPath = Join-Path $stagingRoot $relativePath
        if (-not (Test-Path -LiteralPath $stagedPath -PathType Leaf) -or
            (Get-Item -LiteralPath $stagedPath).Length -le 0) {
            throw "Required robot Python entrypoint is empty or missing: $relativePath"
        }
    }

    $requiredNavigationPackageFiles = @(
        'mapping\robot\go2_mapping_nav_bridge\package.xml',
        'mapping\robot\go2_mapping_nav_bridge\setup.py',
        'mapping\robot\go2_mapping_nav_bridge\setup.cfg',
        'mapping\robot\go2_mapping_nav_bridge\resource\go2_mapping_nav_bridge',
        'mapping\robot\go2_mapping_nav_bridge\launch\go2_mapping_nav_bridge.launch.py',
        'mapping\robot\go2_mapping_nav_bridge\launch\go2_nav2_bringup.launch.py',
        'mapping\robot\go2_mapping_nav_bridge\config\nav2_params.yaml',
        'mapping\robot\go2_mapping_nav_bridge\go2_mapping_nav_bridge\nav2_map_preflight.py'
    )
    foreach ($relativePath in $requiredNavigationPackageFiles) {
        $stagedPath = Join-Path $stagingRoot $relativePath
        if (-not (Test-Path -LiteralPath $stagedPath -PathType Leaf) -or
            (Get-Item -LiteralPath $stagedPath).Length -le 0) {
            throw "Required navigation package file is empty or missing: $relativePath"
        }
    }

    $checksumPath = Join-Path $stagingRoot 'RELEASE.SHA256SUMS'
    $checksumLines = Get-ChildItem -LiteralPath $stagingRoot -File -Recurse | Where-Object {
        $_.FullName -ne $checksumPath
    } | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($stagingRoot.Length).TrimStart('\', '/') -replace '\\', '/'
        $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        "$digest  $relative"
    }
    # sha256sum runs on Linux. Write LF explicitly, because WriteAllLines emits
    # Windows CRLF and turns every verified relative path into a name ending in \r.
    $checksumText = (($checksumLines -join "`n") + "`n")
    [System.IO.File]::WriteAllText($checksumPath, $checksumText, [System.Text.Encoding]::ASCII)

    & tar -czf $archiveFile -C $stagingRoot RELEASE.SHA256SUMS tools mapping
    if ($LASTEXITCODE -ne 0) { throw 'tar archive creation failed' }
    }
    $releaseHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archiveFile).Hash.ToLowerInvariant()
    $remoteArchive = "/tmp/go2-mapping-$releaseHash.tar.gz"
    $remoteRelease = "/home/unitree/go2_mapping/releases/$releaseHash"
    $remoteCommand = "set -e; " +
        "rm -rf '$remoteRelease.incomplete'; mkdir -p '$remoteRelease.incomplete'; " +
        "tar -xzf '$remoteArchive' -C '$remoteRelease.incomplete'; find '$remoteRelease.incomplete/tools' -type f -name '*.sh' -exec chmod 0755 {} +; rm -f '$remoteArchive'; " +
        "cd '$remoteRelease.incomplete'; sha256sum -c RELEASE.SHA256SUMS; " +
        "find tools -type f -name '*.sh' -exec bash -n {} +; " +
        "test -s mapping/robot/go2_mapping_lio/go2_mapping_lio/session_control.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_lio/go2_mapping_lio/session_control.py; " +
        "test -s mapping/robot/go2_mapping_locomotion/terrain_control_service.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_locomotion/terrain_control_service.py; " +
        "test -s mapping/robot/go2_mapping_lio/go2_mapping_lio/map_frame_cloud.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_lio/go2_mapping_lio/map_frame_cloud.py; " +
        "test -s mapping/robot/go2_mapping_lio/go2_mapping_lio/map_frame_cloud_publisher.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_lio/go2_mapping_lio/map_frame_cloud_publisher.py; " +
        "test -s mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/navigation_bridge.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/navigation_bridge.py; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/navigation_executor.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/navigation_executor.py; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/nav2_params.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/nav2_params.py; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/package.xml; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/setup.py; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/setup.cfg; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/resource/go2_mapping_nav_bridge; " +
        "test -s mapping/robot/go2_mapping_nav_bridge/launch/go2_mapping_nav_bridge.launch.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/launch/go2_mapping_nav_bridge.launch.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/launch/go2_nav2_bringup.launch.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_nav_bridge/go2_mapping_nav_bridge/nav2_map_preflight.py; " +
        "test -s tools/start_go2_nav_bridge.sh; " +
        "bash -n tools/start_go2_nav_bridge.sh; " +
        "test -s tools/verify_go2_nav2_runtime.sh; " +
        "bash -n tools/verify_go2_nav2_runtime.sh; " +
        "test -s tools/verify_go2_mapping_release.sh; " +
        "bash -n tools/verify_go2_mapping_release.sh; " +
        "test -s mapping/robot/go2_mapping_lio/go2_mapping_lio/calibration_preflight.py; " +
        "PYTHONPYCACHEPREFIX='/tmp/go2-mapping-pycache-$releaseHash' python3 -m py_compile mapping/robot/go2_mapping_lio/go2_mapping_lio/calibration_preflight.py; " +
        "test -s mapping/robot/go2_mapping_native_lowstate/native_lowstate_imu_socket_server.cpp; " +
        "test -s mapping/robot/go2_mapping_lowstate_imu_bridge/package.xml; " +
        "test -s tools/go2-mapping-lowstate-imu.service; " +
        "rm -rf '/tmp/go2-mapping-pycache-$releaseHash'; " +
        "test -x /bin/bash; test -f tools/verify_go2_mapping_robot.sh; " +
        "bash tools/install_go2_mapping_python_runtime.sh mapping/robot/wheelhouse-aarch64-cp38 mapping/config/robot_wheelhouse_manifest.json; " +
        "bash tools/install_go2_sdk2_python.sh; " +
        "bash tools/install_go2_pointcloud_bridge.sh; " +
        "bash tools/build_go2_lowstate_imu_bridge.sh; " +
        "mkdir -p /home/unitree/.config/systemd/user; install -m 0644 tools/go2-mapping-lowstate-imu.service /home/unitree/.config/systemd/user/go2-mapping-lowstate-imu.service; systemctl --user daemon-reload; " +
        "GO2_MAPPING_ASSET_MANIFEST='mapping/config/robot_runtime_manifest.json' GO2_MAPPING_SKIP_LOCAL_BRIDGE=1 bash tools/verify_go2_mapping_release.sh; " +
        "mv '$remoteRelease.incomplete' '$remoteRelease'; " +
        "mkdir -p /home/unitree/go2_mapping; " +
        "previous_target=`$(readlink -f /home/unitree/go2_mapping/current 2>/dev/null || true); " +
        "if [ -n `"`$previous_target`" ] && [ `"`$previous_target`" != '$remoteRelease' ]; then ln -sfn `"`$previous_target`" /home/unitree/go2_mapping/previous; fi; " +
        "ln -sfn '$remoteRelease' /home/unitree/go2_mapping/current.next; " +
        "mv -Tf /home/unitree/go2_mapping/current.next /home/unitree/go2_mapping/current; " +
        "systemctl --user enable --now go2-mapping-lowstate-imu.service; " +
        "systemctl --user enable --now go2-mapping-realsense-sensor.service; " +
        "systemctl --user enable --now go2-mapping-pointcloud-bridge.service; " +
        "bash /home/unitree/go2_mapping/current/tools/install_go2_pgtt_heightmap.sh; " +
        "systemctl --user enable --now go2-mapping-pgtt-heightmap.service; " +
        "bash /home/unitree/go2_mapping/current/tools/install_go2_pgtt_terrain_router.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/build_go2_motion_mode_helper.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/build_go2_vendor_sport_probe.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/build_go2_vendor_sport_adapter.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/build_go2_lowstate_motor_probe.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/build_go2_vendor_lease_probe.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/build_go2_native_terrain_executor.sh; " +
        "bash /home/unitree/go2_mapping/current/tools/install_go2_nav_bridge.sh; " +
        "install -m 0644 tools/go2-mapping-lio-preview.service /home/unitree/.config/systemd/user/go2-mapping-lio-preview.service; systemctl --user daemon-reload; systemctl --user enable --now go2-mapping-lio-preview.service; " +
        "bash tools/verify_go2_mapping_release.sh"

    & scp @sshTransportOptions -i $PrivateKeyFile -o BatchMode=yes $archiveFile "${UserName}@${Node}:${remoteArchive}"
    if ($LASTEXITCODE -ne 0) { throw 'robot archive upload failed' }
    & ssh @sshTransportOptions -i $PrivateKeyFile -o BatchMode=yes "${UserName}@${Node}" $remoteCommand
    if ($LASTEXITCODE -ne 0) { throw 'robot release activation failed' }
    if ($BulkTokenFile) {
        if (-not (Test-Path -LiteralPath $BulkTokenFile -PathType Leaf)) {
            throw "Bulk token file not found: $BulkTokenFile"
        }
        & scp @sshTransportOptions -i $PrivateKeyFile -o BatchMode=yes $BulkTokenFile `
            "${UserName}@${Node}:/tmp/go2-mapping-bulk-token"
        if ($LASTEXITCODE -ne 0) { throw 'robot bulk token upload failed' }
        & ssh @sshTransportOptions -i $PrivateKeyFile -o BatchMode=yes "${UserName}@${Node}" `
            'set -e; mkdir -p /home/unitree/go2_mapping/runtime; install -m 0600 /tmp/go2-mapping-bulk-token /home/unitree/go2_mapping/runtime/bulk-auth-token; rm -f /tmp/go2-mapping-bulk-token'
        if ($LASTEXITCODE -ne 0) { throw 'robot bulk token activation failed' }
    }
    & scp @sshTransportOptions -i $PrivateKeyFile -o BatchMode=yes $SessionControlTokenFile `
        "${UserName}@${Node}:/tmp/go2-mapping-session-control-token"
    if ($LASTEXITCODE -ne 0) { throw 'robot session-control token upload failed' }
    $installControl = "set -e; trap 'rm -f /tmp/go2-mapping-session-control-token' EXIT; " +
        "chmod 0600 /tmp/go2-mapping-session-control-token; " +
        "bash /home/unitree/go2_mapping/current/tools/install_go2_tailnet_session_control.sh " +
        "--tailnet-ip '$RobotTailnetIp' --allow-peer '$AllowedPcTailnetIp' " +
        "--token-file /tmp/go2-mapping-session-control-token --port '$SessionControlPort'"
    if ($RobotFactorySerial) {
        $installControl += " --robot-serial '$RobotFactorySerial'"
    }
    & ssh @sshTransportOptions -i $PrivateKeyFile -o BatchMode=yes "${UserName}@${Node}" $installControl
    if ($LASTEXITCODE -ne 0) { throw 'robot Tailnet session-control installation failed' }
    $sessionControlToken = (Get-Content -LiteralPath $SessionControlTokenFile -Raw).Trim()
    $preflight = Invoke-DirectRestMethod -Uri "http://${RobotTailnetIp}:${SessionControlPort}/v1/preflight" `
        -Headers @{ Authorization = "Bearer $sessionControlToken" } -TimeoutSec 20
    if ($preflight.ready -ne $true) {
        throw "authenticated robot preflight failed after deployment: $($preflight.report)"
    }
    Write-Output "Activated robot mapping release: $releaseHash"
} finally {
    if ($stagingRoot) { Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue }
    if ($cleanupArchive -and $archiveFile) { Remove-Item -LiteralPath $archiveFile -Force -ErrorAction SilentlyContinue }
}
