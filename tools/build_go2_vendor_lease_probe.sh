#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_file="$release_root/mapping/robot/go2_mapping_locomotion/cpp/go2_vendor_lease_probe.cpp"
output_dir=${GO2_MAPPING_NATIVE_BIN_DIR:-/home/unitree/go2_mapping/runtime/bin}
sdk_include=${GO2_MAPPING_GO2_SDK_INCLUDE:-/usr/local/include}
sdk_lib_dir=${GO2_MAPPING_GO2_SDK_LIB_DIR:-/usr/local/lib}
mkdir -p "$output_dir"
g++ -std=c++14 -O2 "$source_file" -o "$output_dir/go2_vendor_lease_probe" \
  -I"$sdk_include" -I"${GO2_MAPPING_GO2_DDS_INCLUDE:-/usr/local/include/ddscxx}" \
  -I"${GO2_MAPPING_GO2_ICEORYX_INCLUDE:-/usr/local/include/iceoryx/v2.0.2}" \
  -L"$sdk_lib_dir" -lunitree_go2_sdk -lunitree_go2_idl_cpp -lunitree_ros2_idl_cpp \
  -lddsc -lddscxx -lrt -lpthread
echo "GO2_VENDOR_LEASE_PROBE_READY executable=$output_dir/go2_vendor_lease_probe"
