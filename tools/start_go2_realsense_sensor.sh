#!/usr/bin/env bash
# Sensor-only RealSense owner. It never starts a navigation or motion node.
set -Eeuo pipefail

if [[ -r /opt/ros/foxy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
  set -u
else
  echo 'RealSense owner requires /opt/ros/foxy/setup.bash' >&2
  exit 6
fi
if [[ -r /unitree/module/graph_pid_ws/install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
fi

command -v ros2 >/dev/null 2>&1 || { echo 'RealSense owner requires ros2' >&2; exit 6; }
ros2 pkg prefix realsense2_camera >/dev/null 2>&1 || {
  echo 'RealSense owner requires realsense2_camera' >&2
  exit 6
}

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
# Match the checked-in D435I contract and stay within the Jetson USB budget.
exec ros2 launch realsense2_camera rs_launch.py \
  camera_name:="${GO2_MAPPING_CAMERA_NAME:-camera}" \
  enable_color:="${GO2_MAPPING_CAMERA_ENABLE_COLOR:-true}" \
  enable_depth:="${GO2_MAPPING_CAMERA_ENABLE_DEPTH:-true}" \
  rgb_camera.profile:="${GO2_MAPPING_CAMERA_COLOR_PROFILE:-424x240x15}" \
  depth_module.profile:="${GO2_MAPPING_CAMERA_DEPTH_PROFILE:-424x240x15}" \
  align_depth.enable:="${GO2_MAPPING_CAMERA_ALIGN_DEPTH:-true}" \
  enable_infra1:="${GO2_MAPPING_CAMERA_ENABLE_INFRA1:-false}" \
  enable_infra2:="${GO2_MAPPING_CAMERA_ENABLE_INFRA2:-false}" \
  enable_fisheye1:="${GO2_MAPPING_CAMERA_ENABLE_FISHEYE1:-false}" \
  enable_fisheye2:="${GO2_MAPPING_CAMERA_ENABLE_FISHEYE2:-false}" \
  unite_imu_method:="${GO2_MAPPING_CAMERA_UNITE_IMU_METHOD:-0}" \
  output:=screen
