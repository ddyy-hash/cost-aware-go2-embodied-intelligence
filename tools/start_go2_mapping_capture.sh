#!/usr/bin/env bash
# Opt-in capture entrypoint. It never starts camera or motion-control nodes.
set -Eeuo pipefail

duration_s=${1:-60}
mode=${2:-full}
root=${GO2_MAPPING_DATA_ROOT:-/home/unitree/go2-mapping-data}
domain=${ROS_DOMAIN_ID:-42}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
liveness_probe="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
capture_limit_gib=${GO2_MAPPING_CAPTURE_MAX_GIB:-40}
recovery_headroom_gib=${GO2_MAPPING_RECOVERY_HEADROOM_GIB:-4}
minimum_free_gib=${GO2_MAPPING_MIN_FREE_GIB:-$((capture_limit_gib + recovery_headroom_gib + 1))}
progressive_upload=${GO2_MAPPING_PROGRESSIVE_UPLOAD:-0}
rotate_max_bytes=${GO2_MAPPING_ROTATE_MAX_BYTES:-8388608}
uplink_budget_mbps=${GO2_MAPPING_UPLINK_BUDGET_MBPS:-30}
publisher_discovery_attempts=${GO2_MAPPING_PUBLISHER_DISCOVERY_ATTEMPTS:-3}
high_fidelity_rate_mbps=""
capture_session_id=${GO2_MAPPING_CAPTURE_SESSION_ID:-}
stop_file=${GO2_MAPPING_SESSION_STOP_FILE:-}
coordinate_scope=${GO2_MAPPING_COORDINATE_SCOPE:-GLOBAL_MAP}

if [[ ! $duration_s =~ ^[1-9][0-9]*$ ]]; then
  echo "duration must be a positive integer" >&2
  exit 2
fi
if [[ $mode != full && $mode != baseline ]]; then
  echo "mode must be full or baseline" >&2
  exit 2
fi
if [[ $coordinate_scope != GLOBAL_MAP && $coordinate_scope != SESSION_ODOM_LOCAL ]]; then
  echo "GO2_MAPPING_COORDINATE_SCOPE must be GLOBAL_MAP or SESSION_ODOM_LOCAL" >&2
  exit 2
fi
if [[ $coordinate_scope == SESSION_ODOM_LOCAL && $mode != full ]]; then
  echo "SESSION_ODOM_LOCAL requires full capture mode" >&2
  exit 2
fi
if [[ $progressive_upload != 0 && $progressive_upload != 1 ]]; then
  echo "GO2_MAPPING_PROGRESSIVE_UPLOAD must be 0 or 1" >&2
  exit 2
fi
if [[ $coordinate_scope == SESSION_ODOM_LOCAL && $progressive_upload != 1 ]]; then
  echo "SESSION_ODOM_LOCAL requires progressive high-fidelity upload" >&2
  exit 2
fi
if [[ ! $publisher_discovery_attempts =~ ^[1-9][0-9]*$ ]]; then
  echo "GO2_MAPPING_PUBLISHER_DISCOVERY_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if [[ -n $capture_session_id && ! $capture_session_id =~ ^segment-[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "capture session id must match segment-YYYYMMDDTHHMMSSZ" >&2
  exit 2
fi
if [[ ! $rotate_max_bytes =~ ^[1-9][0-9]*$ ]] || (( rotate_max_bytes < 8 * 1024 * 1024 )); then
  echo "GO2_MAPPING_ROTATE_MAX_BYTES must be an integer of at least 8 MiB" >&2
  exit 2
fi
for value in "$capture_limit_gib" "$recovery_headroom_gib" "$minimum_free_gib"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "capture capacity values must be positive integers" >&2; exit 2; }
done

# Reserve enough room for the bounded raw segment, recovery headroom, and one
# GiB of filesystem metadata/checkpoint overhead before creating any ROS node.
if [[ $mode == full ]]; then
  if ! mkdir -p "$root"; then
    echo "full capture refused: cannot create data root $root" >&2
    exit 5
  fi
  available_gib=$(df -BG "$root" 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
  if [[ ! ${available_gib:-} =~ ^[0-9]+$ ]] || (( available_gib < minimum_free_gib )); then
    echo "full capture refused: ${available_gib:-0}GiB available; need at least ${minimum_free_gib}GiB before a ${capture_limit_gib}GiB recording" >&2
    exit 5
  fi
fi

set +u
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID="$domain"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

[[ -f $liveness_probe ]] || { echo "missing ROS topic liveness probe: $liveness_probe" >&2; exit 3; }

has_publisher() {
  local topic=$1
  local attempt
  for ((attempt = 1; attempt <= publisher_discovery_attempts; attempt++)); do
    if timeout 5s ros2 topic info "$topic" 2>/dev/null | awk -F': ' '/Publisher count/ {exit !($2 > 0)}'; then
      return 0
    fi
    if (( attempt < publisher_discovery_attempts )); then
      sleep 1
    fi
  done
  return 1
}

# Discovery alone is insufficient: a stale publisher can remain in the DDS
# graph while the physical sensor is disconnected.  Full capture is allowed
# only after a payload has actually crossed the ROS boundary.
has_live_sample() {
  local topic=$1
  local message_type=$2
  local wait_s=${3:-6}
  local required_frame_id=${4:-}
  local extra=()
  # Foxy's `ros2 topic hz` cannot select a compatible reliability policy. Use
  # a real subscriber with RELIABLE QoS so this test observes sensor payloads.
  [[ -n $required_frame_id ]] && extra=(--require-frame-id "$required_frame_id")
  python3 "$liveness_probe" --topic "$topic" --message-type "$message_type" --timeout "$wait_s" "${extra[@]}"
}

has_advancing_header_sample() {
  local topic=$1
  local message_type=$2
  local wait_s=${3:-6}
  # Raw LiDAR keeps its sensor clock. Require progression between two source
  # timestamps but never compare that clock's epoch directly to host time.
  python3 "$liveness_probe" --topic "$topic" --message-type "$message_type" --timeout "$wait_s" \
    --minimum-header-progress-ms 1
}

signal_recorder_for_close() {
  local recorder_pid=$1
  local recorder_child
  recorder_child=$(pgrep -P "$recorder_pid" | head -n 1 || true)
  if [[ $recorder_child =~ ^[1-9][0-9]*$ ]]; then
    kill -INT "$recorder_child" 2>/dev/null || true
  else
    kill -INT "$recorder_pid" 2>/dev/null || true
  fi
}

monitor_capture_size() {
  local recorder_pid=$1
  local record_dir=$2
  local maximum_bytes=$3
  local capacity_marker=$4
  local session_stop_file=$5
  local intentional_stop_marker=$6
  while kill -0 "$recorder_pid" 2>/dev/null; do
    if [[ -n $session_stop_file && -f $session_stop_file ]]; then
      echo "graceful stop requested by session control" >&2
      : >"$intentional_stop_marker"
      signal_recorder_for_close "$recorder_pid"
      return
    fi
    local used_bytes
    used_bytes=$(du -sb "$record_dir" 2>/dev/null | awk '{print $1}')
    used_bytes=${used_bytes:-0}
    if [[ $used_bytes =~ ^[0-9]+$ ]] && (( used_bytes >= maximum_bytes )); then
      printf 'capture exceeds the configured %s GiB recording limit\n' "$capture_limit_gib" >&2
      : >"$capacity_marker"
      # timeout owns ros2 bag as its direct child. Close that child normally.
      signal_recorder_for_close "$recorder_pid"
      return
    fi
    sleep 2
  done
}

wait_for_recorder_ready() {
  local recorder_pid=$1
  local record_dir=$2
  local attempt
  for ((attempt = 0; attempt < 150; attempt++)); do
    kill -0 "$recorder_pid" 2>/dev/null || return 1
    if [[ -d $record_dir ]] && find "$record_dir" -maxdepth 1 -type f \
      \( -name '*.db3' -o -name '*.db3.active' -o -name '*.db3.zstd' \) \
      -print -quit | grep -q .; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

camera_topics=(
  /camera/color/image_raw
  /camera/depth/image_rect_raw
  /camera/color/camera_info
  /camera/depth/camera_info
  /tf_static
)
for topic in "${camera_topics[@]}"; do
  has_publisher "$topic" || { echo "missing required camera publisher: $topic" >&2; exit 3; }
done

# The existing RealSense owner may or may not publish an aligned depth image.
# Keep it when present for replayable RGB-D assistance, but never reconfigure
# or restart the owner merely to create this optional topic.
aligned_depth_topic=${GO2_MAPPING_ALIGNED_DEPTH_TOPIC:-/camera/aligned_depth_to_color/image_raw}
if has_publisher "$aligned_depth_topic"; then
  camera_topics+=("$aligned_depth_topic")
else
  echo "optional aligned depth topic is unavailable: $aligned_depth_topic" >&2
fi

# `/tf_static` is latched and checked above; dynamic `/tf` must also be kept
# for a replayable camera-to-base-to-lidar chain. It is recorded even when a
# particular baseline has no current publisher, so the bag remains truthful.
topics=("${camera_topics[@]}" /tf)
label=rgbd-baseline
if [[ $mode == full ]]; then
  lidar_topic=${GO2_MAPPING_CAPTURE_LIDAR_TOPIC:-/go2/lio/lidar}
  has_publisher "$lidar_topic" || { echo "full capture refused: $lidar_topic has no publisher" >&2; exit 4; }
  has_publisher /dog_imu_raw || { echo "full capture refused: /dog_imu_raw has no publisher" >&2; exit 4; }
  has_live_sample "$lidar_topic" sensor_msgs/msg/PointCloud2 || { echo "full capture refused: $lidar_topic has no live sample" >&2; exit 4; }
  has_advancing_header_sample "$lidar_topic" sensor_msgs/msg/PointCloud2 || { echo "full capture refused: $lidar_topic source header is not advancing" >&2; exit 4; }
  has_live_sample /dog_imu_raw sensor_msgs/msg/Imu || { echo "full capture refused: /dog_imu_raw has no live sample" >&2; exit 4; }
  topics+=("$lidar_topic" /dog_imu_raw /dog_odom)
  if [[ $progressive_upload == 1 ]]; then
    lio_pose_topic=${GO2_MAPPING_LIO_POSE_TOPIC:-/go2/lio/pose}
    lio_registered_cloud_topic=${GO2_MAPPING_LIO_REGISTERED_CLOUD_TOPIC:-/lio_sam_ros2/mapping/cloud_registered}
    lio_mapping_odometry_topic=${GO2_MAPPING_LIO_MAPPING_ODOMETRY_TOPIC:-/lio_sam_ros2/mapping/odometry}
    lio_global_map_topic=${GO2_MAPPING_LIO_GLOBAL_MAP_TOPIC:-/go2/mapping/map_cloud}
    has_publisher "$lio_pose_topic" || { echo "progressive capture refused: $lio_pose_topic has no publisher" >&2; exit 4; }
    has_live_sample "$lio_pose_topic" geometry_msgs/msg/PoseWithCovarianceStamped || {
      echo "progressive capture refused: $lio_pose_topic has no live sample" >&2
      exit 4
    }
    has_publisher "$lio_registered_cloud_topic" || { echo "progressive capture refused: $lio_registered_cloud_topic has no publisher" >&2; exit 4; }
    has_live_sample "$lio_registered_cloud_topic" sensor_msgs/msg/PointCloud2 || {
      echo "progressive capture refused: $lio_registered_cloud_topic has no live sample" >&2
      exit 4
    }
    has_publisher "$lio_mapping_odometry_topic" || { echo "progressive capture refused: $lio_mapping_odometry_topic has no publisher" >&2; exit 4; }
    has_live_sample "$lio_mapping_odometry_topic" nav_msgs/msg/Odometry || {
      echo "progressive capture refused: $lio_mapping_odometry_topic has no live sample" >&2
      exit 4
    }
    if [[ $coordinate_scope == SESSION_ODOM_LOCAL ]]; then
      has_live_sample "$lio_registered_cloud_topic" sensor_msgs/msg/PointCloud2 8 odom || {
        echo "local session capture refused: $lio_registered_cloud_topic has no live PointCloud2 with header.frame_id=odom" >&2
        exit 4
      }
      topics+=("$lio_pose_topic" "$lio_registered_cloud_topic" "$lio_mapping_odometry_topic")
    else
      # ``cloud_registered`` and mapping odometry are compact ``odom``-frame
      # products. A room-scale replay requires an explicit global ``map``
      # product; raw LiDAR and IMU remain the immutable primary evidence.
      has_publisher "$lio_global_map_topic" || {
        echo "progressive capture refused: $lio_global_map_topic has no publisher" >&2
        exit 4
      }
      has_live_sample "$lio_global_map_topic" sensor_msgs/msg/PointCloud2 8 map || {
        echo "progressive capture refused: $lio_global_map_topic has no live PointCloud2 with header.frame_id=map" >&2
        exit 4
      }
      topics+=("$lio_pose_topic" "$lio_registered_cloud_topic" "$lio_mapping_odometry_topic" "$lio_global_map_topic")
    fi
  fi
  label=lidar-imu-rgbd
fi

if [[ -z $capture_session_id ]]; then
  capture_session_id="segment-$(date -u +%Y%m%dT%H%M%SZ)"
fi
segment="$root/$label/$capture_session_id"
if [[ -e $segment || -e "$segment.sha256" ]]; then
  echo "capture session already exists: $segment" >&2
  exit 2
fi
mkdir -p "$(dirname "$segment")"
max_capture_bytes=$((capture_limit_gib * 1024 * 1024 * 1024))
capacity_marker="$segment.capacity-stop"
intentional_stop_marker="$segment.intentional-stop"
rm -f "$capacity_marker"
rm -f "$intentional_stop_marker"
progressive_watcher_pid=""

publish_session_phase() {
  local phase=$1
  local phase_file=${GO2_MAPPING_SESSION_PHASE_FILE:-}
  [[ -n $phase_file ]] || return 0
  [[ $phase == RECORDING || $phase == FINALIZING ]] || {
    echo "invalid mapping session phase: $phase" >&2
    return 2
  }
  mkdir -p "$(dirname "$phase_file")"
  local temporary="${phase_file}.$$.partial"
  printf '%s\n' "$phase" >"$temporary"
  mv -f -- "$temporary" "$phase_file"
}
upload_pending=0

if [[ $progressive_upload == 1 ]]; then
  [[ $mode == full ]] || { echo "progressive high-fidelity upload requires full mode" >&2; exit 2; }
  if ! ros2 bag record --help 2>&1 | grep -q -- '--max-bag-size'; then
    echo "progressive capture refused: this ROS 2 bag recorder lacks --max-bag-size" >&2
    exit 6
  fi
  high_fidelity_rate_mbps=$(awk -v total="$uplink_budget_mbps" '
    BEGIN {
      if (total <= 0) exit 1;
      # Start at the measured ceiling; the live preview demand heartbeat
      # subtracts its exact requirement before any bulk transfer.
      printf "%.6f", total;
    }') || {
    echo "dual queue refused: use a positive uplink budget" >&2
    exit 2
  }
  export GO2_MAPPING_HIGH_FIDELITY_RATE_MBPS="$high_fidelity_rate_mbps"
  mkdir -p "${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}"
  bash "$script_dir/stream_go2_high_fidelity_queue.sh" "$segment" "$root" \
    >"${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}/high-fidelity-upload.log" 2>&1 &
  progressive_watcher_pid=$!
fi

# Foxy image lacks rosbag2 MCAP plugin. Record sqlite3+zstd explicitly rather
# than silently claiming an MCAP segment. A multi-gigabyte sqlite3 bag can need
# well over 20 seconds to checkpoint after SIGINT, so leave a bounded grace
# period instead of force-killing a potentially corrupt segment.
bag_args=(ros2 bag record --storage sqlite3 --compression-mode file --compression-format zstd --output "$segment")
if [[ $progressive_upload == 1 ]]; then
  bag_args+=(--max-bag-size "$rotate_max_bytes")
fi
bag_args+=("${topics[@]}")
capture_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
timeout --signal=INT --kill-after=180s "$duration_s" "${bag_args[@]}" &
recorder_pid=$!
recorder_ready=1
monitor_pid=""
if ! wait_for_recorder_ready "$recorder_pid" "$segment"; then
  recorder_ready=0
  echo "rosbag recorder did not publish a writable bag before the readiness deadline" >&2
  signal_recorder_for_close "$recorder_pid"
else
  publish_session_phase RECORDING
  monitor_capture_size "$recorder_pid" "$segment" "$max_capture_bytes" "$capacity_marker" "$stop_file" "$intentional_stop_marker" &
  monitor_pid=$!
fi
wait "$recorder_pid"
status=$?
if [[ $monitor_pid =~ ^[1-9][0-9]*$ ]]; then
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
fi
if [[ $recorder_ready == 0 ]]; then
  status=6
fi
intentional_stop_requested=0
if [[ -n $stop_file && -f $stop_file && -f $intentional_stop_marker ]]; then
  intentional_stop_requested=1
fi
rm -f "$intentional_stop_marker"
capacity_stopped=0
if [[ -f $capacity_marker ]]; then
  rm -f "$capacity_marker"
  capacity_stopped=1
  # A capacity guard is an intentional graceful close, not an incomplete bag.
  if [[ $status -eq 124 || $status -eq 130 || $status -eq 137 ]]; then
    status=0
  fi
fi
set -e
if [[ $status -eq 124 || $status -eq 130 ]]; then
  status=0
fi
# Foxy's rosbag recorder returns 2 after the monitor has sent SIGINT in
# response to this session's authenticated Tailnet stop request. This is a
# completed, writable capture only after recorder readiness and the monitor's
# private marker both prove that we initiated that close. Other exit code 2
# outcomes remain failures.
if [[ $intentional_stop_requested -eq 1 && $status -eq 2 ]]; then
  status=0
fi
capture_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
publish_session_phase FINALIZING
topic_contracts=()
for topic in "${topics[@]}"; do
  case "$topic" in
    /camera/color/image_raw|/camera/depth/image_rect_raw|/camera/aligned_depth_to_color/image_raw)
      topic_contracts+=("$topic=sensor_msgs/msg/Image")
      ;;
    /camera/color/camera_info|/camera/depth/camera_info)
      topic_contracts+=("$topic=sensor_msgs/msg/CameraInfo")
      ;;
    /tf|/tf_static)
      topic_contracts+=("$topic=tf2_msgs/msg/TFMessage")
      ;;
    /livox/lidar|/go2/lio/lidar)
      topic_contracts+=("$topic=sensor_msgs/msg/PointCloud2")
      ;;
    /dog_imu_raw)
      topic_contracts+=("$topic=sensor_msgs/msg/Imu")
      ;;
    /dog_odom)
      topic_contracts+=("$topic=nav_msgs/msg/Odometry")
      ;;
    /lio_sam_ros2/mapping/cloud_registered)
      topic_contracts+=("$topic=sensor_msgs/msg/PointCloud2")
      ;;
    /lio_sam_ros2/mapping/odometry)
      topic_contracts+=("$topic=nav_msgs/msg/Odometry")
      ;;
    /go2/mapping/map_cloud)
      topic_contracts+=("$topic=sensor_msgs/msg/PointCloud2")
      ;;
    *)
      if [[ $topic == "${GO2_MAPPING_LIO_POSE_TOPIC:-/go2/lio/pose}" ]]; then
        topic_contracts+=("$topic=geometry_msgs/msg/PoseWithCovarianceStamped")
      else
        echo "capture runtime manifest has no message type contract for $topic" >&2
        exit 6
      fi
      ;;
  esac
done
runtime_manifest="$segment/capture.runtime.json"
if [[ $status -eq 0 ]]; then
  GO2_MAPPING_COORDINATE_SCOPE="$coordinate_scope" bash "$script_dir/write_go2_capture_runtime_manifest.sh" "$segment" "$mode" \
    "$capture_started_at" "$capture_finished_at" "${topic_contracts[@]}"
  [[ -s $runtime_manifest ]] || { echo "capture runtime manifest was not published: $runtime_manifest" >&2; exit 6; }
fi
find "$segment" -type f -print0 | sort -z | xargs -0 sha256sum >"$segment.sha256"
if [[ -n $progressive_watcher_pid ]]; then
  kill "$progressive_watcher_pid" 2>/dev/null || true
  wait "$progressive_watcher_pid" 2>/dev/null || true
  if [[ $status -eq 0 ]]; then
    if ! bash "$script_dir/stream_go2_high_fidelity_queue.sh" --finalize "$segment" "$root"; then
      upload_pending=1
      echo "high-fidelity upload pending; raw artifacts remain local: $segment" >&2
    fi
  else
    echo "high-fidelity queue left the active bag part local because capture failed" >&2
  fi
fi
printf 'CAPTURE segment=%s mode=%s status=%d capacity_stopped=%d upload_pending=%s manifest=%s.sha256\n' \
  "$segment" "$label" "$status" "$capacity_stopped" "$upload_pending" "$segment"
exit "$status"
