#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
source /home/unitree/go2_follow_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

set +e
launch_log=/tmp/go2_follow_smoke_launch.log
timeout 25s ros2 launch go2_person_follow follow_with_realsense.launch.py \
  config:=/home/unitree/go2_follow_ws/install/go2_person_follow/share/go2_person_follow/config/follow_demo.yaml \
  dry_run:=true \
  >"$launch_log" 2>&1
code=$?
set -e

pkill -f "/opt/ros/foxy/lib/realsense2_camera/realsense2_camera_node" 2>/dev/null || true
pkill -f "/home/unitree/go2_follow_ws/install/go2_person_follow/lib/go2_person_follow/follow_node" 2>/dev/null || true

if ! grep -q "RealSense Node Is Up" "$launch_log"; then
  tail -80 "$launch_log"
  exit 1
fi

if ! grep -q "detector=auto_yolo:tensorrt_yolo" "$launch_log"; then
  tail -80 "$launch_log"
  exit 1
fi

grep -m1 "RealSense Node Is Up" "$launch_log"
grep -m1 "detector=auto_yolo:tensorrt_yolo" "$launch_log"

if [ "$code" -eq 124 ]; then
  exit 0
fi

exit "$code"
