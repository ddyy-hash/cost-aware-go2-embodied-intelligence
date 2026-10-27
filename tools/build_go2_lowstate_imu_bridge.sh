#!/usr/bin/env bash
# Builds the isolated read-only LowState reader used by the ROS socket bridge.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
native_reader="$release_root/tools/build_go2_native_lowstate_imu_reader.sh"
publisher="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/native_lowstate_imu_ros.py"
[[ -f $native_reader ]] || { echo "missing native LowState build entrypoint: $native_reader" >&2; exit 2; }
[[ -f $publisher ]] || { echo "missing ROS LowState publisher: $publisher" >&2; exit 2; }

PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/go2-mapping-pycache-lowstate}" \
  python3 -m py_compile "$publisher"
bash "$native_reader"
printf 'LOWSTATE_IMU_BRIDGE_READY reader=%s publisher=%s\n' \
  "${GO2_MAPPING_NATIVE_BIN_DIR:-/home/unitree/go2_mapping/runtime/bin}/go2_native_lowstate_imu_socket_server" "$publisher"
