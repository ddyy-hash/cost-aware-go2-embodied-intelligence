#!/usr/bin/env bash
# Read-only ROS Foxy/Nav2 runtime probe. It never launches nodes or sends goals.
set -Eeuo pipefail

require_nav2=0
probe_timeout=${GO2_NAV2_PROBE_TIMEOUT_SECONDS:-8}
for argument in "$@"; do
  case "$argument" in
    --require-nav2) require_nav2=1 ;;
    --help|-h)
      printf '%s\n' 'Usage: verify_go2_nav2_runtime.sh [--require-nav2]'
      exit 0
      ;;
    *)
      echo "unknown argument: $argument" >&2
      exit 2
      ;;
  esac
done

if ! [[ $probe_timeout =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v value="$probe_timeout" 'BEGIN { exit !(value > 0) }'; then
  echo 'GO2_NAV2_PROBE_TIMEOUT_SECONDS must be a positive number' >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
liveness_probe="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
report_dir=${GO2_MAPPING_REPORT_DIR:-"$release_root/reports"}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
report_file="$report_dir/nav2-runtime-$timestamp.txt"
mkdir -p -- "$report_dir"
exec > >(tee "$report_file") 2>&1

pass_count=0
warn_count=0
fail_count=0

check_pass() { printf 'CHECK %s PASS %s\n' "$1" "$2"; pass_count=$((pass_count + 1)); }
check_warn() { printf 'CHECK %s WARN %s\n' "$1" "$2"; warn_count=$((warn_count + 1)); }
check_fail() { printf 'CHECK %s FAIL %s\n' "$1" "$2"; fail_count=$((fail_count + 1)); }

check_liveness() {
  local label=$1
  local topic=$2
  local message_type=$3
  local frame_id=${4:-}
  local progress_ms=${5:-}
  local require_payload=${6:-0}
  local output
  local args=(--topic "$topic" --message-type "$message_type" --timeout "$probe_timeout")
  [[ -n $frame_id ]] && args+=(--require-frame-id "$frame_id")
  [[ -n $progress_ms ]] && args+=(--minimum-header-progress-ms "$progress_ms")
  [[ $require_payload == 1 ]] && args+=(--require-nonempty-data)
  if output=$(timeout "${probe_timeout}s" python3 "$liveness_probe" "${args[@]}" 2>&1); then
    check_pass "$label" "$output"
  else
    check_fail "$label" "$output"
  fi
}

check_action_server() {
  local action_name=$1
  local output
  output=$(timeout "${probe_timeout}s" ros2 action info "$action_name" 2>&1 || true)
  if grep -Eq 'Action servers:[[:space:]]*[1-9][0-9]*' <<<"$output"; then
    check_pass "action_server_${action_name#/}" "$output"
  else
    check_fail "action_server_${action_name#/}" "$output"
  fi
}

check_lifecycle_active() {
  local node_name=$1
  local output
  output=$(timeout "${probe_timeout}s" ros2 lifecycle get "$node_name" 2>&1 || true)
  if grep -Eiq '(^|[[:space:]])active([[:space:]]|$)' <<<"$output"; then
    check_pass "lifecycle_${node_name#/}" "$output"
  else
    check_fail "lifecycle_${node_name#/}" "$output"
  fi
}

check_topic_connection() {
  local label=$1
  local topic=$2
  local required=$3
  local output
  output=$(timeout "${probe_timeout}s" ros2 topic info "$topic" -v 2>&1 || true)
  if grep -Eq "$required" <<<"$output"; then
    check_pass "$label" "$output"
  else
    check_fail "$label" "$output"
  fi
}

check_tf_chain() {
  local parent=$1
  local child=$2
  local output
  output=$(timeout "${probe_timeout}s" ros2 run tf2_ros tf2_echo "$parent" "$child" 2>&1 || true)
  if grep -q 'Translation:' <<<"$output" && grep -q 'Rotation:' <<<"$output"; then
    check_pass "tf_${parent}_${child}" "$output"
  else
    check_fail "tf_${parent}_${child}" "$output"
  fi
}

printf 'GO2_NAV2_RUNTIME_PROBE version=1 timestamp=%s require_nav2=%s release=%s\n' \
  "$timestamp" "$require_nav2" "$release_root"

if [[ -r /opt/ros/foxy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
  set -u
  check_pass ros_foxy_setup /opt/ros/foxy/setup.bash
else
  check_fail ros_foxy_setup missing
fi
if [[ -r /unitree/module/graph_pid_ws/install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
  check_pass unitree_graph_setup /unitree/module/graph_pid_ws/install/setup.bash
else
  check_warn unitree_graph_setup missing
fi
nav2_overlay="${GO2_MAPPING_NAV2_WS:-/home/unitree/go2_mapping/runtime/nav2_ws}/install/local_setup.bash"
if [[ -r "$nav2_overlay" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$nav2_overlay"
  set -u
  check_pass nav2_overlay_setup "$nav2_overlay"
else
  check_warn nav2_overlay_setup missing
fi

if ! command -v ros2 >/dev/null 2>&1; then
  check_fail ros2_command missing
  printf 'SUMMARY state=UNVERIFIED pass=%d warn=%d fail=%d report=%s\n' "$pass_count" "$warn_count" "$fail_count" "$report_file"
  exit 2
fi
check_pass ros2_command "$(command -v ros2)"

required_packages=(
  nav2_bringup nav2_msgs nav2_map_server nav2_controller nav2_planner
  nav2_bt_navigator nav2_waypoint_follower nav2_lifecycle_manager
  robot_state_publisher tf2_ros
)
for package_name in "${required_packages[@]}"; do
  package_prefix=$(ros2 pkg prefix "$package_name" 2>/dev/null || true)
  if [[ -n $package_prefix ]]; then
    check_pass "package_$package_name" "$package_prefix"
  elif [[ $require_nav2 == 1 ]]; then
    check_fail "package_$package_name" missing
  else
    check_warn "package_$package_name" missing
  fi
done
optional_packages=(nav2_behaviors nav2_recoveries nav2_velocity_smoother nav2_collision_monitor)
for package_name in "${optional_packages[@]}"; do
  package_prefix=$(ros2 pkg prefix "$package_name" 2>/dev/null || true)
  if [[ -n $package_prefix ]]; then
    check_pass "optional_package_$package_name" "$package_prefix"
  else
    check_warn "optional_package_$package_name" unavailable_on_foxy_or_overlay
  fi
done

topic_list=$(timeout 8s ros2 topic list 2>/dev/null || true)
if [[ -z $topic_list ]]; then
  if [[ $require_nav2 == 1 ]]; then check_fail ros_topic_graph unavailable; else check_warn ros_topic_graph unavailable; fi
else
  grep -Fxq '/tf' <<<"$topic_list" && check_pass tf_topic /tf || {
    if [[ $require_nav2 == 1 ]]; then check_fail tf_topic missing; else check_warn tf_topic missing; fi
  }
  costmap_topics=$(grep -E '(^|/)costmap($|/)' <<<"$topic_list" || true)
  if [[ -n $costmap_topics ]]; then
    check_pass costmap_topics "$costmap_topics"
  elif [[ $require_nav2 == 1 ]]; then
    check_fail costmap_topics missing
  else
    check_warn costmap_topics missing
  fi
fi

action_list=$(timeout 8s ros2 action list 2>/dev/null || true)
for action_name in /navigate_to_pose /follow_waypoints; do
  if grep -Fxq "$action_name" <<<"$action_list"; then
    check_pass "action_${action_name#/}" available
  elif [[ $require_nav2 == 1 ]]; then
    check_fail "action_${action_name#/}" missing
  else
    check_warn "action_${action_name#/}" missing
  fi
done

node_list=$(timeout 8s ros2 node list 2>/dev/null || true)
for node_name in /controller_server /planner_server /bt_navigator /map_server; do
  if grep -Fxq "$node_name" <<<"$node_list"; then
    check_pass "node_${node_name#/}" running
  elif [[ $require_nav2 == 1 ]]; then
    check_fail "node_${node_name#/}" missing
  else
    check_warn "node_${node_name#/}" missing
  fi
done

if [[ $require_nav2 == 1 ]]; then
  if [[ ! -s $liveness_probe ]]; then
    check_fail liveness_probe missing
  else
    nav2_lidar_topic=${GO2_NAV2_LIDAR_TOPIC:-/livox/lidar}
    nav2_imu_topic=${GO2_NAV2_IMU_TOPIC:-/dog_imu_raw}
    nav2_color_topic=${GO2_NAV2_COLOR_TOPIC:-/camera/color/image_raw}
    nav2_depth_topic=${GO2_NAV2_DEPTH_TOPIC:-/camera/depth/image_rect_raw}
    nav2_info_topic=${GO2_NAV2_CAMERA_INFO_TOPIC:-/camera/color/camera_info}
    nav2_odom_topic=${GO2_NAV2_ODOM_TOPIC:-/odom}

    check_liveness sensor_lidar "$nav2_lidar_topic" sensor_msgs/msg/PointCloud2 '' 1.0
    check_liveness sensor_imu "$nav2_imu_topic" sensor_msgs/msg/Imu '' 1.0
    check_liveness sensor_color "$nav2_color_topic" sensor_msgs/msg/Image '' '' 1
    check_liveness sensor_depth "$nav2_depth_topic" sensor_msgs/msg/Image '' '' 1
    check_liveness sensor_camera_info "$nav2_info_topic" sensor_msgs/msg/CameraInfo
    check_liveness odometry "$nav2_odom_topic" nav_msgs/msg/Odometry odom 1.0
    check_liveness tf_messages /tf tf2_msgs/msg/TFMessage
    check_liveness tf_static_messages /tf_static tf2_msgs/msg/TFMessage

    check_tf_chain map odom
    check_tf_chain odom base_link
    check_action_server /navigate_to_pose
    check_action_server /follow_waypoints

    for lifecycle_node in /map_server /controller_server /planner_server /bt_navigator /behavior_server /waypoint_follower; do
      check_lifecycle_active "$lifecycle_node"
    done

    check_topic_connection cmd_vel_smoothed_has_subscriber /cmd_vel_smoothed 'Subscription count:[[:space:]]*[1-9][0-9]*'
    check_topic_connection cmd_vel_has_publisher /cmd_vel 'Publisher count:[[:space:]]*[1-9][0-9]*'
    check_topic_connection cmd_vel_has_subscriber /cmd_vel 'Subscription count:[[:space:]]*[1-9][0-9]*'
  fi
fi

state=READY
if [[ $fail_count -gt 0 ]]; then state=BLOCKED; elif [[ $warn_count -gt 0 ]]; then state=UNVERIFIED; fi
printf 'SUMMARY state=%s pass=%d warn=%d fail=%d report=%s\n' "$state" "$pass_count" "$warn_count" "$fail_count" "$report_file"
if [[ $require_nav2 == 1 && $fail_count -gt 0 ]]; then exit 2; fi
exit 0
