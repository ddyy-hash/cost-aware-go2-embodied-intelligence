#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${GO2_MAPPING_CALIBRATION_PREFLIGHT_PYTHON:-/home/unitree/go2_mapping/runtime/calibration-preflight-venv/bin/python}

if [[ ! -x $python_bin ]]; then
  python_bin=$(command -v python3)
fi
if ! "$python_bin" -c 'import cv2; assert hasattr(cv2, "aruco")' >/dev/null 2>&1; then
  echo "calibration preflight requires a Python OpenCV build with ArUco support" >&2
  exit 3
fi

export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}
set +u
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

exec "$python_bin" "$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/calibration_preflight.py" "$@"
