#!/usr/bin/env bash
# Offline, read-only verification of an activated robot release.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
manifest="${GO2_MAPPING_ASSET_MANIFEST:-$release_root/mapping/config/robot_runtime_manifest.json}"
if [[ $manifest != /* ]]; then manifest="$release_root/$manifest"; fi
checksums="$release_root/RELEASE.SHA256SUMS"

[[ -s $manifest ]] || { echo "robot runtime manifest missing: $manifest" >&2; exit 3; }
[[ -s $checksums ]] || { echo "release checksum manifest missing: $checksums" >&2; exit 3; }
grep -Fq '"schema_version": 1' "$manifest" || { echo 'unsupported robot runtime manifest' >&2; exit 3; }
grep -Fq '"asset_type": "go2_robot_runtime"' "$manifest" || { echo 'wrong robot runtime asset type' >&2; exit 3; }
sha256sum -c "$checksums" >/dev/null

required_files=(
  mapping/robot/environment.yml
  mapping/robot/go2_mapping_capture/go2_mapping_capture/recording.py
  mapping/robot/go2_mapping_capture/go2_mapping_capture/retention.py
  mapping/robot/go2_mapping_lio/go2_mapping_lio/session_control.py
  mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py
  mapping/robot/go2_mapping_lio/go2_mapping_lio/calibration_preflight.py
  mapping/robot/go2_mapping_nav_bridge/package.xml
  mapping/robot/go2_mapping_nav_bridge/setup.py
  mapping/robot/go2_mapping_nav_bridge/config/nav2_params.yaml
  mapping/robot/go2_mapping_locomotion/__init__.py
  mapping/robot/go2_mapping_locomotion/request_gate.py
  mapping/robot/go2_mapping_locomotion/policy_runtime.py
  mapping/robot/go2_mapping_locomotion/sdk2_adapter.py
  mapping/robot/go2_mapping_locomotion/readonly_pgtt_runner.py
  mapping/robot/go2_mapping_locomotion/heightmap.py
  mapping/robot/go2_mapping_locomotion/heightmap_ros.py
  mapping/robot/go2_mapping_locomotion/readonly_pgtt_sdk_runner.py
  mapping/robot/go2_mapping_locomotion/terrain_control_service.py
  mapping/robot/go2_mapping_locomotion/vendor_sport_adapter.py
  mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_sport_probe.cpp
  mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_sport_adapter.cpp
  mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_lease_probe.cpp
  mapping/robot/go2_mapping_locomotion/cpp/go2_lowstate_motor_probe.cpp
  mapping/robot/go2_mapping_locomotion/cpp/go2_native_terrain_executor.cpp
)
[[ -s "$release_root/tools/start_go2_pgtt_heightmap.sh" ]] || {
  echo 'PGTT heightmap launcher is missing' >&2
  exit 3
}
for terrain_tool in \
  build_go2_vendor_sport_probe.sh build_go2_vendor_sport_adapter.sh \
  build_go2_vendor_lease_probe.sh build_go2_lowstate_motor_probe.sh \
  build_go2_native_terrain_executor.sh query_go2_vendor_sport.py; do
  [[ -s "$release_root/tools/$terrain_tool" ]] || {
    echo "terrain-control tool missing: $terrain_tool" >&2
    exit 3
  }
done
for relative in "${required_files[@]}"; do
  [[ -s "$release_root/$relative" ]] || { echo "required release file missing: $relative" >&2; exit 3; }
done
[[ -s "$release_root/tools/start_go2_realsense_sensor.sh" && -s "$release_root/tools/install_go2_realsense_sensor.sh" ]] || {
  echo 'RealSense sensor owner tools are missing' >&2
  exit 3
}
[[ -s "$release_root/mapping/robot/vendor/unitree_sdk2_python-65691c8.tar.gz" && -s "$release_root/mapping/robot/vendor/cyclonedds-0.10.2.tar.gz" && -s "$release_root/mapping/robot/vendor/typing_extensions-4.7.1-py3-none-any.whl" ]] || {
  echo 'SDK2 Python vendor archives are missing' >&2
  exit 3
}
[[ -s "$release_root/mapping/config/unitree_sdk2_python_manifest.json" && -s "$release_root/tools/install_go2_sdk2_python.sh" ]] || {
  echo 'SDK2 Python runtime manifest or installer is missing' >&2
  exit 3
}
[[ -s "$release_root/mapping/config/pgtt_go2_numpy_manifest.json" && -s "$release_root/mapping/robot/pgtt-go2/policy_level03_run0.npz" ]] || {
  echo 'PGTT NumPy policy or manifest is missing' >&2
  exit 3
}
[[ -s "$release_root/mapping/config/locomotion_backends.json" ]] || {
  echo 'locomotion backend registry is missing' >&2
  exit 3
}
grep -Fq '"asset_type": "go2_locomotion_backend_registry"' "$release_root/mapping/config/locomotion_backends.json" || {
  echo 'unsupported locomotion backend registry' >&2
  exit 3
}

for ros_package in rclpy livox_ros_driver2 lio_sam_ros2 realsense2_camera nav2_bringup nav2_msgs; do
  grep -Fq "\"$ros_package\"" "$manifest" || {
    echo "robot manifest is missing ROS package declaration: $ros_package" >&2
    exit 3
  }
done
for ros_package in nav2_collision_monitor nav2_velocity_smoother; do
  if ros2 pkg prefix "$ros_package" >/dev/null 2>&1; then
    echo "OPTIONAL_ROS_PACKAGE $ros_package PRESENT"
  else
    echo "OPTIONAL_ROS_PACKAGE $ros_package UNAVAILABLE"
  fi
done
for python_module in yaml zstandard mcap psutil; do
  grep -Fq "\"$python_module\"" "$manifest" || {
    echo "robot manifest is missing Python module declaration: $python_module" >&2
    exit 3
  }
done
command -v python3 >/dev/null 2>&1 || { echo 'python3 is required by the robot release' >&2; exit 3; }
python3 -c 'import yaml, zstandard, mcap, psutil' || {
  echo 'robot Python runtime is missing one or more declared modules' >&2
  exit 3
}
# Non-interactive SSH sessions do not inherit the ROS environment used by the
# authenticated preflight service. Load only the existing robot overlays; do
# not invent a second ROS installation or alter the operator shell.
for ros_setup in /opt/ros/foxy/setup.bash /unitree/module/graph_pid_ws/install/setup.bash; do
  if [[ -f "$ros_setup" ]]; then
    # shellcheck disable=SC1090
    # ROS Foxy setup files reference optional variables while nounset is on.
    set +u
    source "$ros_setup"
    set -u
  fi
done
nav2_overlay="${GO2_MAPPING_NAV2_WS:-/home/unitree/go2_mapping/runtime/nav2_ws}/install/local_setup.bash"
if [[ -f "$nav2_overlay" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$nav2_overlay"
  set -u
fi
command -v ros2 >/dev/null 2>&1 || { echo 'ROS 2 CLI is required by the robot release' >&2; exit 3; }
ros_packages=(rclpy livox_ros_driver2 lio_sam_ros2 realsense2_camera nav2_bringup nav2_msgs)
if [[ ${GO2_MAPPING_SKIP_LOCAL_BRIDGE:-0} != 1 ]]; then
  ros_packages+=(go2_mapping_nav_bridge)
fi
for ros_package in "${ros_packages[@]}"; do
  ros2 pkg prefix "$ros_package" >/dev/null 2>&1 || {
    echo "ROS package is not installed: $ros_package" >&2
    exit 3
  }
done

printf 'GO2_MAPPING_RELEASE_READY manifest=%s files=%d\n' "$manifest" "${#required_files[@]}"
