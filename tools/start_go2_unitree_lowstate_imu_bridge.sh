#!/usr/bin/env bash
# Starts the isolated Unitree LowState reader and ROS Imu publisher.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
publisher_script="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/native_lowstate_imu_ros.py"
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
socket_path=${GO2_MAPPING_NATIVE_LOWSTATE_SOCKET:-$runtime_dir/native_lowstate_imu.sock}
output_topic=${GO2_MAPPING_LOWSTATE_IMU_TOPIC:-/dog_imu_raw}
frame_id=${GO2_MAPPING_LOWSTATE_IMU_FRAME:-go2_imu_raw}
interface=${GO2_MAPPING_UNITREE_INTERFACE:-eth0}
native_reader="$runtime_dir/bin/go2_native_lowstate_imu_socket_server"
publisher_log="$runtime_dir/native-lowstate-imu-ros.log"
reader_log="$runtime_dir/native-lowstate-imu-reader.log"

[[ -f $publisher_script ]] || { echo "missing ROS IMU publisher: $publisher_script" >&2; exit 2; }
mkdir -p "$runtime_dir"
if [[ ! -x $native_reader ]]; then
  GO2_MAPPING_NATIVE_BIN_DIR="$runtime_dir/bin" bash "$script_dir/build_go2_native_lowstate_imu_reader.sh"
fi

set +u
source /opt/ros/foxy/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

python3 "$publisher_script" \
  --socket "$socket_path" \
  --output-topic "$output_topic" \
  --fallback-frame-id "$frame_id" \
  >"$publisher_log" 2>&1 &
publisher_pid=$!

cleanup() {
  kill "$publisher_pid" 2>/dev/null || true
  wait "$publisher_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_socket() {
  local attempts=0
  while [[ ! -S $socket_path && $attempts -lt 50 ]]; do
    sleep 0.1
    attempts=$((attempts + 1))
  done
  [[ -S $socket_path ]]
}

wait_for_socket || { cat "$publisher_log" >&2; echo "ROS IMU socket did not become ready" >&2; exit 4; }
printf 'LOWSTATE_IMU_BRIDGE source=rt/lowstate output=%s socket=%s\n' "$output_topic" "$socket_path"
env LD_LIBRARY_PATH=/usr/local/lib "$native_reader" --interface "$interface" --socket "$socket_path" --frame-id "$frame_id" \
  >"$reader_log" 2>&1
