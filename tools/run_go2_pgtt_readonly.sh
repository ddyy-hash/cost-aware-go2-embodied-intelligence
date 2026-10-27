#!/usr/bin/env bash
# Run PGTT from SDK2 with a ROS-produced heightmap snapshot; never publishes.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# This process uses SDK2/CycloneDDS only. Do not source ROS 2/Fast
# DDS libraries here; the heightmap producer is a separate process.
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/unitree/cyclonedds_ws/install/cyclonedds}"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$release_root"
exec python3 -m mapping.robot.go2_mapping_locomotion.readonly_pgtt_sdk_runner "$@"
