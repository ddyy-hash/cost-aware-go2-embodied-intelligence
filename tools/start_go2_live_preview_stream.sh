#!/usr/bin/env bash
# Starts a read-only keyframe publisher only after a pose-qualified LIO stream exists.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
producer="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/live_keyframe_producer.py"
liveness_probe="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
sequence_state=${GO2_MAPPING_LIVE_SEQUENCE_STATE:-$runtime_dir/live-preview.sequence}
calibration_id=${GO2_MAPPING_CALIBRATION_ID:-}
map_id=${GO2_MAPPING_MAP_ID:-go2-live}
building_id=${GO2_MAPPING_BUILDING_ID:-$map_id}
session_id=${GO2_MAPPING_CAPTURE_SESSION_ID:-standalone-live}
pose_topic=${GO2_MAPPING_LIO_POSE_TOPIC:-/go2/lio/pose}
lidar_topic=${GO2_MAPPING_LIVE_LIDAR_TOPIC:-/lio_sam_ros2/deskew/cloud_deskewed}
pose_timeout_s=${GO2_MAPPING_LIO_POSE_TIMEOUT_S:-12}
lidar_timeout_s=${GO2_MAPPING_LIVE_LIDAR_TIMEOUT_S:-12}
pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}
endpoint=${GO2_MAPPING_LIVE_ENDPOINT:-ws://${pc_tailnet_ip}:8765/ws/live/keyframe}
health_url=${GO2_MAPPING_LIVE_HEALTH_URL:-http://${pc_tailnet_ip}:8765/api/live/readiness}
camera_assist=${GO2_MAPPING_CAMERA_ASSIST:-1}
texture_keyframes=${GO2_MAPPING_TEXTURE_KEYFRAMES:-1}
texture_endpoint=${GO2_MAPPING_TEXTURE_ENDPOINT:-ws://${pc_tailnet_ip}:8765/ws/live/texture}
preview_demand_state=${GO2_MAPPING_PREVIEW_DEMAND_STATE:-$runtime_dir/preview-demand.json}
bootstrap_total_mbps=${GO2_MAPPING_BOOTSTRAP_BANDWIDTH_MBPS:-4.0}
sensor_frame_transform_id=${GO2_MAPPING_SENSOR_FRAME_TRANSFORM_ID:-PROVISIONAL_GO2_UTLIDAR_PITCH180_v2}

# Corporate/proxy environments must not intercept the robot-to-PC Tailnet path.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${pc_tailnet_ip}"
export no_proxy="${no_proxy:+$no_proxy,}${pc_tailnet_ip}"

[[ -n $calibration_id ]] || { echo 'live preview refused: GO2_MAPPING_CALIBRATION_ID is required' >&2; exit 2; }
[[ -f $producer ]] || { echo "missing keyframe producer: $producer" >&2; exit 2; }
[[ -f $liveness_probe ]] || { echo "missing ROS topic liveness probe: $liveness_probe" >&2; exit 2; }
[[ $pose_timeout_s =~ ^[1-9][0-9]*$ ]] || { echo 'live preview refused: pose timeout must be a positive integer' >&2; exit 2; }
[[ $lidar_timeout_s =~ ^[1-9][0-9]*$ ]] || { echo 'live preview refused: LiDAR timeout must be a positive integer' >&2; exit 2; }
python3 -c 'import websocket' >/dev/null 2>&1 || {
  echo 'live preview refused: websocket-client is not installed; run install_go2_live_websocket_dependency.sh <wheel>' >&2
  exit 2
}
set +u
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

topic_has_publisher() {
  timeout 4s ros2 topic info "$1" 2>/dev/null | awk -F': ' '/Publisher count/ {exit !($2 > 0)}'
}

python3 "$liveness_probe" --topic "$pose_topic" \
  --message-type geometry_msgs/msg/PoseWithCovarianceStamped --timeout "$pose_timeout_s" || {
  echo "live preview refused: no live LIO pose on $pose_topic" >&2; exit 3;
}
python3 "$liveness_probe" --topic "$lidar_topic" \
  --message-type sensor_msgs/msg/PointCloud2 --timeout "$lidar_timeout_s" || {
  echo "live preview refused: no live deskewed LiDAR cloud on $lidar_topic" >&2; exit 3;
}
curl --fail --silent --show-error "$health_url" >/dev/null || {
  echo "live preview refused: PC mapper endpoint is unavailable at $endpoint" >&2; exit 4;
}
mkdir -p "$runtime_dir"
args=(--endpoint "$endpoint" --map-id "$map_id" --building-id "$building_id" --session-id "$session_id"
  --calibration-id "$calibration_id" --pose-topic "$pose_topic" --sequence-state "$sequence_state"
  --lidar-topic "$lidar_topic"
  --sensor-frame-transform-id "$sensor_frame_transform_id"
  --pose-cloud-max-skew-ms "${GO2_MAPPING_LIVE_POSE_CLOUD_MAX_SKEW_MS:-100}"
  --surface-cell-m "${GO2_MAPPING_DETAIL_SURFACE_CELL_M:-0.005}"
  --detail-payload-limit-bytes "${GO2_MAPPING_DETAIL_PAYLOAD_LIMIT_BYTES:-4194304}"
  --detail-target-hz "${GO2_MAPPING_DETAIL_TARGET_HZ:-5}"
  --detail-minimum-hz "${GO2_MAPPING_DETAIL_MINIMUM_HZ:-1}"
  --preview-demand-state "$preview_demand_state"
  --bootstrap-total-mbps "$bootstrap_total_mbps"
  # A wide ACK window tolerates long DERP RTT and ordered reconnect replay.
  --delivery-window "${GO2_MAPPING_LIVE_DELIVERY_WINDOW:-20}"
  --preview-queue-size "${GO2_MAPPING_LIVE_PREVIEW_QUEUE_SIZE:-64}"
  --lidar-min-range-m "${GO2_MAPPING_LIVE_LIDAR_MIN_RANGE_M:-0.45}"
  --lidar-max-range-m "${GO2_MAPPING_LIVE_LIDAR_MAX_RANGE_M:-12}")
if [[ ${GO2_MAPPING_LIVE_POINTS_IN_MAP_FRAME:-0} == 1 ]]; then
  args+=(--points-in-map-frame)
fi
if [[ $camera_assist != 0 && $camera_assist != 1 ]]; then
  echo 'live preview refused: GO2_MAPPING_CAMERA_ASSIST must be 0 or 1' >&2
  exit 2
fi
if [[ $texture_keyframes != 0 && $texture_keyframes != 1 ]]; then
  echo 'live preview refused: GO2_MAPPING_TEXTURE_KEYFRAMES must be 0 or 1' >&2
  exit 2
fi
if [[ $texture_keyframes == 1 && $camera_assist != 1 ]]; then
  echo 'live preview refused: texture keyframes require the existing camera subscriber' >&2
  exit 2
fi
if [[ $camera_assist == 1 ]]; then
  camera_depth_topic=${GO2_MAPPING_CAMERA_DEPTH_TOPIC:-/camera/aligned_depth_to_color/image_raw}
  camera_depth_aligned=${GO2_MAPPING_CAMERA_DEPTH_ALIGNED:-1}
  if [[ $camera_depth_aligned != 0 && $camera_depth_aligned != 1 ]]; then
    echo 'live preview refused: GO2_MAPPING_CAMERA_DEPTH_ALIGNED must be 0 or 1' >&2
    exit 2
  fi
  if ! topic_has_publisher "$camera_depth_topic"; then
    camera_depth_topic=${GO2_MAPPING_CAMERA_RAW_DEPTH_TOPIC:-/camera/depth/image_rect_raw}
    camera_depth_aligned=0
    echo "camera assist using unaligned depth fallback: $camera_depth_topic" >&2
  fi
  args+=(--camera-enabled
    --camera-color-topic "${GO2_MAPPING_CAMERA_COLOR_TOPIC:-/camera/color/image_raw}"
    --camera-depth-topic "$camera_depth_topic"
    --camera-info-topic "${GO2_MAPPING_CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
    --camera-depth-info-topic "${GO2_MAPPING_CAMERA_DEPTH_INFO_TOPIC:-/camera/depth/camera_info}"
    --camera-max-hz "${GO2_MAPPING_CAMERA_MAX_HZ:-15}"
    --camera-max-width "${GO2_MAPPING_CAMERA_MAX_WIDTH:-640}"
    --camera-max-height "${GO2_MAPPING_CAMERA_MAX_HEIGHT:-480}"
    --camera-depth-max-width "${GO2_MAPPING_CAMERA_DEPTH_MAX_WIDTH:-320}"
    --camera-depth-max-height "${GO2_MAPPING_CAMERA_DEPTH_MAX_HEIGHT:-240}"
    --camera-jpeg-quality "${GO2_MAPPING_CAMERA_JPEG_QUALITY:-82}"
    --camera-max-arrival-skew-ms "${GO2_MAPPING_CAMERA_MAX_ARRIVAL_SKEW_MS:-120}"
    --camera-depth-max-age-ms "${GO2_MAPPING_CAMERA_DEPTH_MAX_AGE_MS:-1000}"
    --camera-depth-scale-m "${GO2_MAPPING_CAMERA_DEPTH_SCALE_M:-0.001}")
  if [[ $texture_keyframes == 1 ]]; then
    args+=(--texture-enabled
      --texture-endpoint "$texture_endpoint"
      --texture-max-hz "${GO2_MAPPING_TEXTURE_MAX_HZ:-2}"
      --texture-queue-size "${GO2_MAPPING_TEXTURE_QUEUE_SIZE:-8}"
      --texture-delivery-window "${GO2_MAPPING_TEXTURE_DELIVERY_WINDOW:-2}"
      --texture-jpeg-quality "${GO2_MAPPING_TEXTURE_JPEG_QUALITY:-90}"
      --texture-pose-max-skew-ms "${GO2_MAPPING_TEXTURE_POSE_MAX_SKEW_MS:-150}")
  fi
  if [[ $camera_depth_aligned == 1 ]]; then
    args+=(--camera-depth-aligned)
  fi
fi
if [[ -n ${GO2_MAPPING_LIVE_TOKEN:-} ]]; then
  args+=(--token "$GO2_MAPPING_LIVE_TOKEN")
fi
exec python3 "$producer" "${args[@]}"
