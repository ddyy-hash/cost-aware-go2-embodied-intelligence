#!/usr/bin/env bash
# Opt-in Livox-to-PGTT heightmap publisher. It never publishes motion topics.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -r /opt/ros/foxy/setup.bash ]]; then
  set +u
  source /opt/ros/foxy/setup.bash
  set -u
else
  echo 'PGTT heightmap requires /opt/ros/foxy/setup.bash' >&2
  exit 6
fi
if [[ -r /unitree/module/graph_pid_ws/install/setup.bash ]]; then
  set +u
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
cd "$release_root"
exec python3 -m mapping.robot.go2_mapping_locomotion.heightmap_ros \
  --input-topic "${GO2_MAPPING_HEIGHTMAP_INPUT_TOPIC:-/livox/lidar}" \
  --output-topic "${GO2_MAPPING_HEIGHTMAP_OUTPUT_TOPIC:-/rt/elevation_heightmap}" \
  --frame-id "${GO2_MAPPING_HEIGHTMAP_FRAME_ID:-base_link}" \
  --sensor-frame-transform-id "${GO2_MAPPING_SENSOR_FRAME_TRANSFORM_ID:-AUTO_GO2_LIDAR_FRAME_v1}" \
  --output-file "${GO2_MAPPING_HEIGHTMAP_FILE:-/home/unitree/go2_mapping/runtime/pgtt-heightmap.bin}" \
  --stair-probe-file "${GO2_MAPPING_STAIR_PROBE_FILE:-/home/unitree/go2_mapping/runtime/stair-probe-profile.bin}"
