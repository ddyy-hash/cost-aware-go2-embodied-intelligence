#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

launch_log=/tmp/go2_realsense_smoke_launch.log
color_sample=/tmp/go2_realsense_color_once.txt
depth_sample=/tmp/go2_realsense_depth_once.txt
topic_list=/tmp/go2_realsense_topics.txt

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

ros2 topic list >"$topic_list"

color_topic=""
for candidate in /camera/color/image_raw /camera/camera/color/image_raw; do
  if grep -qx "$candidate" "$topic_list"; then
    color_topic="$candidate"
    break
  fi
done

depth_topic=""
for candidate in /camera/depth/image_rect_raw /camera/camera/depth/image_rect_raw; do
  if grep -qx "$candidate" "$topic_list"; then
    depth_topic="$candidate"
    break
  fi
done

if [ -z "$color_topic" ] || [ -z "$depth_topic" ]; then
  cat "$topic_list"
  tail -80 "$launch_log"
  exit 1
fi

set +e
timeout 20s ros2 topic echo --no-arr "$color_topic" >"$color_sample"
color_code=$?
timeout 20s ros2 topic echo --no-arr "$depth_topic" >"$depth_sample"
depth_code=$?
set -e

if ! grep -q "height:" "$color_sample"; then
  echo "color_topic=$color_topic"
  tail -80 "$launch_log"
  exit "$color_code"
fi

if ! grep -q "height:" "$depth_sample"; then
  echo "depth_topic=$depth_topic"
  tail -80 "$launch_log"
  exit "$depth_code"
fi

echo "color_topic=$color_topic"
grep -m1 "height:" "$color_sample"
grep -m1 "width:" "$color_sample"
grep -m1 "encoding:" "$color_sample"
echo "depth_topic=$depth_topic"
grep -m1 "height:" "$depth_sample"
grep -m1 "width:" "$depth_sample"
grep -m1 "encoding:" "$depth_sample"
