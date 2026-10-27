#!/usr/bin/env bash
set -eo pipefail

pkill -f "[r]os_image_mjpeg.py" 2>/dev/null || true
pkill -f "[r]ealsense2_camera_node" 2>/dev/null || true
sleep 2

if command -v usbreset >/dev/null 2>&1; then
  echo 123 | sudo -S usbreset 8086:0b3a >/tmp/usbreset_rs.log 2>&1 || true
  sleep 5
fi

nohup bash -lc '
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=false \
  enable_color:=true \
  enable_depth:=true \
  enable_infra1:=false \
  enable_infra2:=false \
  rgb_camera.profile:=640x480x15 \
  depth_module.profile:=640x480x15
' >/tmp/go2_realsense_preview.log 2>&1 &

nohup bash -lc '
sleep 8
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 /home/unitree/go2_person_follow/tools/ros_image_mjpeg.py \
  --topic /camera/color/image_raw \
  --host 0.0.0.0 \
  --port 8082 \
  --fps 15 \
  --quality 80
' >/tmp/go2_ros_image_mjpeg.log 2>&1 &

echo "preview_url=http://192.168.123.18:8082/"
