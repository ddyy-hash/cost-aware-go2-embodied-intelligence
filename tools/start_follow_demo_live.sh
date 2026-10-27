#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
source /home/unitree/go2_follow_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

log_file="${1:-/tmp/go2_follow_demo_live.log}"
base_config_path="${2:-/home/unitree/go2_follow_ws/install/go2_person_follow/share/go2_person_follow/config/follow_demo.yaml}"
config_path="/tmp/go2_follow_demo_live.yaml"

pkill -f '/home/unitree/go2_follow_ws/install/go2_person_follow/lib/go2_person_follow/follow_node' 2>/dev/null || true
pkill -f 'ros2 run go2_person_follow follow_node' 2>/dev/null || true
pkill -f '/home/unitree/go2_person_follow/bin/go2_wireless_bridge' 2>/dev/null || true
pkill -f '/home/unitree/go2_person_follow/bin/go2_sport_bridge' 2>/dev/null || true

rm -f "$log_file"
cp "$base_config_path" "$config_path"
sed -i 's/^\([[:space:]]*dry_run:[[:space:]]*\).*/\1false/' "$config_path"

nohup ros2 run go2_person_follow follow_node \
  --ros-args \
  --params-file "$config_path" \
  >"$log_file" 2>&1 &

echo "$!"
