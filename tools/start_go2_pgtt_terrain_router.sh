#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
mkdir -p "$runtime_dir"
exec env PYTHONPATH="$release_root" python3 -m mapping.robot.go2_mapping_locomotion.terrain_router_monitor \
  --heightmap-file "$runtime_dir/pgtt-heightmap.bin" \
  --stair-probe-file "$runtime_dir/stair-probe-profile.bin" \
  --network-interface "${GO2_MAPPING_UNITREE_INTERFACE:-eth0}" \
  --duration "${GO2_MAPPING_TERRAIN_ROUTER_DURATION:-31536000}" \
  --period "${GO2_MAPPING_TERRAIN_ROUTER_PERIOD:-0.05}" \
  --status-file "$runtime_dir/terrain-router-status.json"
