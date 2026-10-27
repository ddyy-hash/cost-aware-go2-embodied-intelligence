#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
source /home/unitree/go2_follow_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

config_path="${1:-/home/unitree/go2_follow_ws/install/go2_person_follow/share/go2_person_follow/config/follow_demo.yaml}"

pkill -f '/home/unitree/go2_follow_ws/install/go2_person_follow/lib/go2_person_follow/follow_node' 2>/dev/null || true
pkill -f 'ros2 run go2_person_follow follow_node' 2>/dev/null || true
pkill -f '/home/unitree/go2_person_follow/bin/go2_wireless_bridge' 2>/dev/null || true
pkill -f '/home/unitree/go2_person_follow/bin/go2_sport_bridge' 2>/dev/null || true
pkill -f '/opt/ros/foxy/lib/realsense2_camera/realsense2_camera_node' 2>/dev/null || true

exec ros2 launch go2_person_follow follow_with_realsense.launch.py \
  config:="$config_path" \
  dry_run:=false
