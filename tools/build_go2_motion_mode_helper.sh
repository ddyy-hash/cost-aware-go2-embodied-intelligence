#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_file="$release_root/mapping/robot/go2_mapping_locomotion/cpp/go2_motion_mode_helper.cpp"
output_dir=${GO2_MAPPING_NATIVE_BIN_DIR:-/home/unitree/go2_mapping/runtime/bin}
sdk_include=${GO2_MAPPING_SDK_INCLUDE:-/usr/local/include}
sdk_lib=${GO2_MAPPING_UNITREE_SDK_LIB:-/usr/local/lib/libunitree_sdk2.a}
dds_lib_dir=${GO2_MAPPING_DDS_LIB_DIR:-/usr/local/lib}
motion_switcher_header="$sdk_include/unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"

# Go2 images commonly ship the older Go2-only SDK.  It has no MotionSwitcher
# implementation, and mixing it with the newer B2 headers causes ABI crashes.
# Treat that as an explicit, non-fatal capability miss so deployment can keep
# the read-only runtime healthy without pretending that mode takeover exists.
if [[ ! -f "$motion_switcher_header" ]]; then
    echo "GO2_MOTION_MODE_HELPER_SKIPPED reason=motion_switcher_header_missing path=$motion_switcher_header"
    exit 0
fi
if [[ ! -f "$sdk_lib" ]]; then
    echo "GO2_MOTION_MODE_HELPER_SKIPPED reason=unitree_sdk_library_missing path=$sdk_lib"
    exit 0
fi
if ! command -v nm >/dev/null 2>&1 || ! nm -C "$sdk_lib" 2>/dev/null | grep 'MotionSwitcherClient::' >/dev/null; then
    echo "GO2_MOTION_MODE_HELPER_SKIPPED reason=motion_switcher_symbol_missing path=$sdk_lib"
    exit 0
fi

mkdir -p "$output_dir"
g++ -std=c++14 -O2 "$source_file" -o "$output_dir/go2_motion_mode_helper" \
  -I"$sdk_include" -I"${GO2_MAPPING_DDS_INCLUDE:-/usr/local/include/ddscxx}" -I"${GO2_MAPPING_ICEORYX_INCLUDE:-/usr/local/include/iceoryx/v2.0.2}" \
  "$sdk_lib" -L"$dds_lib_dir" \
  -lddsc -lddscxx -lrt -lpthread
echo "GO2_MOTION_MODE_HELPER_READY executable=$output_dir/go2_motion_mode_helper"
