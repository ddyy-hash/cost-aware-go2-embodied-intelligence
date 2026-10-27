#!/usr/bin/env bash
# Starts the isolated Unitree DDS reader and ROS PointCloud2 publisher.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
publisher_script="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/native_pointcloud_ros.py"
livox_adapter_script="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/livox_custom_ros.py"
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
socket_path=${GO2_MAPPING_NATIVE_POINTCLOUD_SOCKET:-$runtime_dir/native_pointcloud.sock}
source_topic=${GO2_MAPPING_NATIVE_LIDAR_TOPIC:-rt/utlidar/cloud}
output_topic=${GO2_MAPPING_POINTCLOUD_TOPIC:-/livox/lidar}
lio_output_topic=${GO2_MAPPING_LIO_POINTCLOUD_TOPIC:-/go2/lio/lidar}
lio_livox_topic=${GO2_MAPPING_LIO_LIVOX_TOPIC:-/go2/lio/livox}
frame_id=${GO2_MAPPING_POINTCLOUD_FRAME:-go2_lidar_raw}
lio_stamp_guard_ms=${GO2_MAPPING_LIO_POINTCLOUD_GUARD_MS:-160}
interface=${GO2_MAPPING_UNITREE_INTERFACE:-eth0}
native_reader="$runtime_dir/bin/go2_native_pointcloud_socket_server"
publisher_log="$runtime_dir/native-pointcloud-ros.log"
livox_adapter_log="$runtime_dir/livox-custom-adapter.log"
reader_log="$runtime_dir/native-pointcloud-reader.log"

[[ -f $publisher_script ]] || { echo "missing ROS point-cloud publisher: $publisher_script" >&2; exit 2; }
[[ -f $livox_adapter_script ]] || { echo "missing Livox CustomMsg adapter: $livox_adapter_script" >&2; exit 2; }
mkdir -p "$runtime_dir"
if [[ ! -x $native_reader ]]; then
  GO2_MAPPING_NATIVE_BIN_DIR="$runtime_dir/bin" bash "$script_dir/build_go2_native_pointcloud_reader.sh"
fi

set +u
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

python3 "$publisher_script" \
  --socket "$socket_path" \
  --output-topic "$output_topic" \
  --lio-output-topic "$lio_output_topic" \
  --fallback-frame-id "$frame_id" \
  --lio-stamp-guard-ms "$lio_stamp_guard_ms" \
  >"$publisher_log" 2>&1 &
publisher_pid=$!

python3 "$livox_adapter_script" \
  --input-topic "$lio_output_topic" \
  --output-topic "$lio_livox_topic" \
  >"$livox_adapter_log" 2>&1 &
livox_adapter_pid=$!

cleanup() {
  kill "$publisher_pid" 2>/dev/null || true
  kill "$livox_adapter_pid" 2>/dev/null || true
  wait "$publisher_pid" 2>/dev/null || true
  wait "$livox_adapter_pid" 2>/dev/null || true
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

wait_for_socket || { cat "$publisher_log" >&2; echo "ROS point-cloud socket did not become ready" >&2; exit 4; }
printf 'POINTCLOUD_BRIDGE source=%s raw_output=%s lio_output=%s livox_output=%s lio_guard_ms=%s socket=%s\n' \
  "$source_topic" "$output_topic" "$lio_output_topic" "$lio_livox_topic" "$lio_stamp_guard_ms" "$socket_path"
env LD_LIBRARY_PATH=/usr/local/lib "$native_reader" --interface "$interface" --source-topic "$source_topic" --socket "$socket_path" \
  >"$reader_log" 2>&1
