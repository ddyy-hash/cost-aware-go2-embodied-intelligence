#!/usr/bin/env bash
# Compatibility entrypoint for the canonical Unitree PointCloud2 bridge.
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/start_go2_unitree_pointcloud_bridge.sh" "$@"
