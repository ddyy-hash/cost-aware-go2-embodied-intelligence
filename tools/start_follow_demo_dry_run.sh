#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
source /home/unitree/go2_follow_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

log_file="${1:-/tmp/go2_follow_demo_dry_run.log}"
config_path="${2:-/home/unitree/go2_follow_ws/install/go2_person_follow/share/go2_person_follow/config/follow_demo.yaml}"

pkill -f '/home/unitree/go2_follow_ws/install/go2_person_follow/lib/go2_person_follow/follow_node' 2>/dev/null || true
pkill -f 'ros2 run go2_person_follow follow_node' 2>/dev/null || true
pkill -f '/home/unitree/go2_person_follow/bin/go2_wireless_bridge' 2>/dev/null || true

rm -f "$log_file"

nohup ros2 run go2_person_follow follow_node \
  --ros-args \
  --params-file "$config_path" \
  -p dry_run:=true \
  >"$log_file" 2>&1 &

echo "$!"
