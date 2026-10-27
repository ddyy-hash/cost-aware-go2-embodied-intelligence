#!/usr/bin/env bash
# Compatibility entrypoint for the isolated Unitree LowState -> ROS Imu bridge.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/start_go2_unitree_lowstate_imu_bridge.sh" "$@"
