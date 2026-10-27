[CmdletBinding()]
param(
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $workspaceRoot 'output\robot-release' }
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('go2-mapping-bundle-' + [Guid]::NewGuid().ToString('N'))
$archiveTemp = "$stagingRoot.tar.gz"
$robotTools = @(
    'verify_go2_mapping_robot.sh',
    'install_go2_live_websocket_dependency.sh',
    'start_go2_mapping_capture.sh',
    'run_go2_mapping_session.sh',
    'start_go2_pgtt_heightmap.sh',
    'run_go2_pgtt_readonly.sh',
    'install_go2_pgtt_heightmap.sh',
    'start_go2_unitree_pointcloud_bridge.sh',
    'build_go2_native_pointcloud_reader.sh',
    'start_go2_realsense_sensor.sh',
    'install_go2_realsense_sensor.sh',
    'go2-mapping-lowstate-imu.service',
    'build_go2_lowstate_imu_bridge.sh',
    'build_go2_native_lowstate_imu_reader.sh',
    'start_go2_lowstate_imu_bridge.sh',
    'start_go2_unitree_lowstate_imu_bridge.sh',
    'run_go2_calibration_preflight.sh',
    'write_go2_capture_runtime_manifest.sh',
    'stream_go2_high_fidelity_queue.sh',
    'upload_go2_segment_via_tunnel.sh',
    'start_go2_lio_preview.sh',
    'go2-mapping-lio-preview.service',
    'start_go2_live_preview_stream.sh',
    'start_go2_robot_scan.sh',
    'install_go2_tailnet_session_control.sh',
    'go2-mapping-session-control.service',
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
try {
    New-Item -ItemType Directory -Force -Path (Join-Path $stagingRoot 'tools') | Out-Null
    foreach ($tool in $robotTools) {
        $source = Join-Path $workspaceRoot "tools\$tool"
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing robot tool: $tool" }
        Copy-Item -LiteralPath $source -Destination (Join-Path $stagingRoot 'tools')
    }
    $mappingStage = Join-Path $stagingRoot 'mapping'
    New-Item -ItemType Directory -Force -Path $mappingStage | Out-Null
    Copy-Item -LiteralPath (Join-Path $workspaceRoot 'mapping\robot') -Destination $mappingStage -Recurse
    Copy-Item -LiteralPath (Join-Path $workspaceRoot 'mapping\config') -Destination $mappingStage -Recurse
    Get-ChildItem -LiteralPath $stagingRoot -Directory -Filter '__pycache__' -Recurse | Remove-Item -Recurse -Force
    $manifestPath = Join-Path $stagingRoot 'mapping\config\robot_runtime_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Robot runtime manifest is missing from the bundle.' }
    $wheelhouseManifestPath = Join-Path $stagingRoot 'mapping\config\robot_wheelhouse_manifest.json'
    if (-not (Test-Path -LiteralPath $wheelhouseManifestPath -PathType Leaf)) { throw 'Robot wheelhouse manifest is missing from the bundle.' }
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

    $checksumPath = Join-Path $stagingRoot 'RELEASE.SHA256SUMS'
    $checksumLines = Get-ChildItem -LiteralPath $stagingRoot -File -Recurse | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($stagingRoot.Length).TrimStart('\', '/') -replace '\\', '/'
        $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        "$digest  $relative"
    }
    [System.IO.File]::WriteAllText($checksumPath, (($checksumLines -join "`n") + "`n"), [System.Text.Encoding]::ASCII)

    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    & tar -czf $archiveTemp -C $stagingRoot RELEASE.SHA256SUMS tools mapping
    if ($LASTEXITCODE -ne 0) { throw 'robot bundle archive creation failed' }
    $bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archiveTemp).Hash.ToLowerInvariant()
    $archivePath = Join-Path $OutputDirectory "go2-mapping-robot-$bundleHash.tar.gz"
    Copy-Item -LiteralPath $archiveTemp -Destination $archivePath -Force
    Write-Output "GO2_MAPPING_ROBOT_BUNDLE_READY hash=$bundleHash path=$archivePath"
} finally {
    Remove-Item -LiteralPath $stagingRoot,$archiveTemp -Recurse -Force -ErrorAction SilentlyContinue
}
