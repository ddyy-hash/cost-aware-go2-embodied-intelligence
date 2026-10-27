#!/usr/bin/env bash
# Compile the Unitree-only DDS reader without sourcing ROS to avoid DDS ABI mixing.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
source_file="$release_root/mapping/robot/go2_mapping_native_pointcloud/native_pointcloud_socket_server.cpp"
output_dir=${GO2_MAPPING_NATIVE_BIN_DIR:-/home/unitree/go2_mapping/runtime/bin}
output_file="$output_dir/go2_native_pointcloud_socket_server"

[[ -f $source_file ]] || { echo "missing native point-cloud source: $source_file" >&2; exit 2; }
mkdir -p "$output_dir"

g++ -std=c++14 -O2 "$source_file" -o "$output_file" \
  -I/usr/local/include \
  -I/usr/local/include/ddscxx \
  -I/usr/local/include/iceoryx/v2.0.2 \
  -L/usr/local/lib \
  -lunitree_go2_sdk -lunitree_go2_idl_cpp -lunitree_ros2_idl_cpp \
  -lddsc -lddscxx -lrt -lpthread

printf 'NATIVE_POINTCLOUD_READER executable=%s\n' "$output_file"
