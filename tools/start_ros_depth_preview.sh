#!/usr/bin/env bash
set -eo pipefail

pkill -f "ros_image_mjpeg.py --topic /camera/depth/image_rect_raw" 2>/dev/null || true

nohup bash -lc '
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 /home/unitree/go2_person_follow/tools/ros_image_mjpeg.py \
  --topic /camera/depth/image_rect_raw \
  --host 0.0.0.0 \
  --port 8083 \
  --fps 15 \
  --quality 80
' >/tmp/go2_ros_depth_mjpeg.log 2>&1 &

echo "depth_url=http://192.168.123.18:8083/"
