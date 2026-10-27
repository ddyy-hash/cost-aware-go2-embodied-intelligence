#!/usr/bin/env bash
# Run a read-only, provisional LIO-SAM preview stack in its own process group.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
params=${GO2_MAPPING_LIO_PARAMS:-$release_root/mapping/config/lio_preview_mid360.yaml}
calibration_id=${GO2_MAPPING_CALIBRATION_ID:-PROVISIONAL_MID360_IMU_EXTRINSIC_UNVERIFIED_v1}
source_pose_topic=${GO2_MAPPING_LIO_SOURCE_POSE_TOPIC:-/lio_sam_ros2/mapping/odometry}
pose_topic=${GO2_MAPPING_LIO_POSE_TOPIC:-/go2/lio/pose}
quality_topic=${GO2_MAPPING_LIO_QUALITY_TOPIC:-/go2/lio/quality}
adapter="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/lio_pose_adapter.py"
registered_cloud_topic=${GO2_MAPPING_LIO_REGISTERED_CLOUD_TOPIC:-/lio_sam_ros2/mapping/cloud_registered}
global_map_topic=${GO2_MAPPING_LIO_GLOBAL_MAP_TOPIC:-/go2/mapping/map_cloud}
target_frame=${GO2_MAPPING_LIO_TARGET_FRAME:-odom}
map_cloud_publisher="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/map_frame_cloud_publisher.py"
pid_file="$runtime_dir/lio-preview.pid"
lock_file="$runtime_dir/lio-preview.lock"

[[ -f $params ]] || { echo "LIO preview parameters not found: $params" >&2; exit 2; }
[[ -f $adapter ]] || { echo "LIO pose adapter not found: $adapter" >&2; exit 2; }
[[ -f $map_cloud_publisher ]] || { echo "LIO map-frame cloud publisher not found: $map_cloud_publisher" >&2; exit 2; }
mkdir -p "$runtime_dir"

set +u
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

lio_prefix=$(ros2 pkg prefix lio_sam_ros2 2>/dev/null) || {
  echo 'LIO preview refused: lio_sam_ros2 is not installed on this robot' >&2; exit 3;
}
lio_binary_dir="$lio_prefix/lib/lio_sam_ros2"
[[ -d $lio_binary_dir ]] || {
  echo "LIO preview refused: package executable directory is missing: $lio_binary_dir" >&2; exit 3;
}

# Do not compete with an existing operator-owned LIO-SAM mapper.
# Fast DDS discovery on the robot can occasionally block `ros2 topic info`.
# This is only a conflict guard; make it bounded and retain the real
# pose-message liveness gate in the session runner.
existing_publishers=$(timeout 5s ros2 topic info "$source_pose_topic" 2>/dev/null \
  | awk -F': ' '/Publisher count/ {print $2}' || true)
if [[ ${existing_publishers:-0} -gt 0 ]]; then
  echo "LIO preview refused: $source_pose_topic already has a publisher" >&2
  exit 4
fi

exec 9>"$lock_file"
flock -n 9 || { echo 'LIO preview already has an active launcher lock' >&2; exit 5; }
printf '%s\n' "$$" >"$pid_file"
children=()

stop_children() {
  local child
  trap - INT TERM EXIT
  for child in "${children[@]:-}"; do
    # This launcher is commonly started with nohup, where child processes may
    # inherit an ignored SIGINT disposition. SIGTERM remains actionable and
    # lets the ROS launch wrappers release their child nodes.
    kill -TERM "$child" 2>/dev/null || true
  done
  for child in "${children[@]:-}"; do
    wait "$child" 2>/dev/null || true
  done
  [[ $(cat "$pid_file" 2>/dev/null || true) == "$$" ]] && rm -f "$pid_file"
}
trap stop_children INT TERM EXIT

start_lio_node() {
  local executable=$1
  local binary="$lio_binary_dir/$executable"
  [[ -x $binary ]] || { echo "LIO preview executable is not installed: $binary" >&2; exit 3; }
  "$binary" --ros-args --params-file "$params" -p use_sim_time:=false \
    >>"$runtime_dir/lio-preview.log" 2>&1 &
  children+=("$!")
}

: >"$runtime_dir/lio-preview.log"
printf 'Starting provisional LIO preview profile=%s calibration=%s\n' "$params" "$calibration_id" \
  | tee -a "$runtime_dir/lio-preview.log"
start_lio_node lio_sam_ros2_dogOdomForMapping
start_lio_node lio_sam_ros2_imuPreintegration
start_lio_node lio_sam_ros2_imageProjection
start_lio_node lio_sam_ros2_featureExtraction
start_lio_node lio_sam_ros2_mapOptmization
python3 "$adapter" --source-topic "$source_pose_topic" --pose-topic "$pose_topic" \
  --quality-topic "$quality_topic" --calibration-id "$calibration_id" \
  >>"$runtime_dir/lio-preview.log" 2>&1 &
children+=("$!")
python3 "$map_cloud_publisher" --input-topic "$registered_cloud_topic" --output-topic "$global_map_topic" \
  --map-frame "$target_frame" >>"$runtime_dir/lio-preview.log" 2>&1 &
children+=("$!")

while true; do
  for child in "${children[@]}"; do
    if ! kill -0 "$child" 2>/dev/null; then
      echo "LIO preview child exited unexpectedly: $child" >&2
      exit 6
    fi
  done
  sleep 1
done
