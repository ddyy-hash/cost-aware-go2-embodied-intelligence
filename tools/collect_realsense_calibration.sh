#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
source /home/unitree/go2_follow_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

output_dir="${1:-/home/unitree/go2_person_follow/calibration/images}"
count="${2:-128}"
launch_log=/tmp/go2_realsense_calibration_launch.log

mkdir -p "$output_dir"
rm -f "$output_dir"/calib_*.jpg

ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=false \
  enable_color:=true \
  enable_depth:=true \
  enable_infra1:=false \
  enable_infra2:=false \
  rgb_camera.profile:=640x480x15 \
  depth_module.profile:=640x480x15 \
  >"$launch_log" 2>&1 &
launch_pid=$!

cleanup() {
  kill "$launch_pid" 2>/dev/null || true
  wait "$launch_pid" 2>/dev/null || true
  pkill -f "/opt/ros/foxy/lib/realsense2_camera/realsense2_camera_node" 2>/dev/null || true
}
trap cleanup EXIT

sleep 8

python3 /home/unitree/go2_person_follow/tools/collect_realsense_calibration.py \
  --topic /camera/color/image_raw \
  --output-dir "$output_dir" \
  --count "$count" \
  --interval-s 0.10 \
  --timeout-s 160

find "$output_dir" -maxdepth 1 -name "calib_*.jpg" | wc -l
