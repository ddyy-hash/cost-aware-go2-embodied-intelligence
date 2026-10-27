#!/usr/bin/env bash
# Starts the ROS-agnostic Nav2 execution bridge for mission-derived goals.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
navigation_root=${GO2_MAPPING_NAVIGATION_ROOT:-$runtime_dir/navigation}
poll_seconds=${GO2_MAPPING_NAVIGATION_POLL_SECONDS:-1.0}
timeout_seconds=${GO2_MAPPING_NAVIGATION_TIMEOUT_SECONDS:-8.0}

if [[ -f /opt/ros/foxy/setup.bash ]]; then
  set +u
  source /opt/ros/foxy/setup.bash
  set -u
fi
if [[ -f /unitree/module/graph_pid_ws/install/setup.bash ]]; then
  set +u
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
fi
nav2_ws=${GO2_MAPPING_NAV2_WS:-$runtime_dir/nav2_ws}
if [[ -f "$nav2_ws/install/local_setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$nav2_ws/install/local_setup.bash"
  set -u
fi

cd "$release_root"
exec python3 -m mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_executor \
  --navigation-root "$navigation_root" \
  --poll-seconds "$poll_seconds" \
  --timeout-seconds "$timeout_seconds"
