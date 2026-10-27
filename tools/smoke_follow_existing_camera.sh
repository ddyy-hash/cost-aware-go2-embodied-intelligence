#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
source /home/unitree/go2_follow_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

config_path="${1:-/home/unitree/go2_follow_ws/install/go2_person_follow/share/go2_person_follow/config/follow_demo.yaml}"
launch_log="${2:-/tmp/go2_follow_existing_camera_smoke.log}"

rm -f "$launch_log"

ros2 run go2_person_follow follow_node \
  --ros-args \
  --params-file "$config_path" \
  -p dry_run:=true \
  -p wireless_topic:=/test_wirelesscontroller \
  -p audio_enabled:=false \
  >"$launch_log" 2>&1 &

node_pid=$!

cleanup() {
  if kill -0 "$node_pid" 2>/dev/null; then
    kill "$node_pid" 2>/dev/null || true
    wait "$node_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 30); do
  if grep -q "Person follow node started" "$launch_log"; then
    break
  fi
  if ! kill -0 "$node_pid" 2>/dev/null; then
    tail -80 "$launch_log"
    exit 1
  fi
  sleep 0.5
done

python3 - <<'PY'
import time

import rclpy
from unitree_go.msg import WirelessController

rclpy.init()
node = rclpy.create_node("go2_follow_f1_smoke_publisher")
publisher = node.create_publisher(WirelessController, "/test_wirelesscontroller", 10)

deadline = time.time() + 8.0
while publisher.get_subscription_count() == 0 and time.time() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.05)

if publisher.get_subscription_count() == 0:
    raise RuntimeError("no subscriber matched /test_wirelesscontroller")

def publish_keys(keys, duration_s):
    end_s = time.time() + duration_s
    msg = WirelessController()
    msg.keys = keys
    while time.time() < end_s:
        publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.03)

for keys, duration_s in ((0, 0.25), (64, 0.20), (0, 0.25), (64, 0.20), (0, 0.25)):
    publish_keys(keys, duration_s)

node.destroy_node()
rclpy.shutdown()
PY

sleep 8

grep -m1 "Person follow node started" "$launch_log"
grep -m1 "detector=auto_yolo:tensorrt_yolo" "$launch_log"
grep -m1 "Follow mode enabled" "$launch_log"

if grep -q "dry_run move" "$launch_log"; then
  grep -m1 "dry_run move" "$launch_log"
elif grep -q "target_lost" "$launch_log"; then
  tail -80 "$launch_log"
  exit 1
else
  tail -80 "$launch_log"
  exit 1
fi
