#!/usr/bin/env bash
# Runs one durable robot capture and uploads only after its segment is closed.
set -Eeuo pipefail

duration_s=${1:-300}
mode=${2:-baseline}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
data_root=${GO2_MAPPING_DATA_ROOT:-/home/unitree/go2-mapping-data}
pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}
high_fidelity_url=${GO2_MAPPING_HIGH_FIDELITY_URL:-http://192.0.2.10:8000}
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
live_enabled=${GO2_MAPPING_LIVE_ENABLED:-0}
lio_pid_file="$runtime_dir/lio-preview.pid"
live_pid_file="$runtime_dir/live-preview.pid"
lio_started_by_session=0
live_started_by_session=0
lio_session_pid=""
live_session_pid=""
requested_progressive_upload=${GO2_MAPPING_PROGRESSIVE_UPLOAD:-}
rotate_max_bytes=${GO2_MAPPING_ROTATE_MAX_BYTES:-8388608}
uplink_budget_mbps=${GO2_MAPPING_UPLINK_BUDGET_MBPS:-30}

if [[ ! $duration_s =~ ^[1-9][0-9]*$ ]]; then
  echo "duration must be a positive integer" >&2
  exit 2
fi
if [[ $mode != full && $mode != baseline ]]; then
  echo "mode must be full or baseline" >&2
  exit 2
fi
if [[ -z $requested_progressive_upload ]]; then
  progressive_upload=0
  [[ $mode == full ]] && progressive_upload=1
else
  progressive_upload=$requested_progressive_upload
fi
if [[ $progressive_upload != 0 && $progressive_upload != 1 ]]; then
  echo "GO2_MAPPING_PROGRESSIVE_UPLOAD must be 0 or 1" >&2
  exit 2
fi
export GO2_MAPPING_PROGRESSIVE_UPLOAD="$progressive_upload"
export GO2_MAPPING_ROTATE_MAX_BYTES="$rotate_max_bytes"
export GO2_MAPPING_UPLINK_BUDGET_MBPS="$uplink_budget_mbps"
export GO2_MAPPING_HIGH_FIDELITY_URL="$high_fidelity_url"
export GO2_MAPPING_DELETE_RECONSTRUCTED=${GO2_MAPPING_DELETE_RECONSTRUCTED:-1}
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${pc_tailnet_ip}"
export no_proxy="${no_proxy:+$no_proxy,}${pc_tailnet_ip}"

wait_for_process() {
  local pattern=$1
  local log_file=$2
  local attempts=0
  while ! pgrep -f "$pattern" >/dev/null 2>&1; do
    if [[ $attempts -ge 120 ]]; then
      echo "mapping session refused: bridge process did not become ready: $pattern" >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    sleep 0.5
    attempts=$((attempts + 1))
  done
}

pid_is_alive() {
  local pid_file=$1
  [[ -s $pid_file ]] || return 1
  local pid
  pid=$(<"$pid_file")
  [[ $pid =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null
}

pid_matches_command() {
  local pid_file=$1
  local expected_command=$2
  pid_is_alive "$pid_file" || return 1
  local pid command
  pid=$(<"$pid_file")
  [[ -r /proc/$pid/cmdline ]] || return 1
  command=$(tr '\0' ' ' <"/proc/$pid/cmdline")
  [[ $command == *"$expected_command"* ]]
}

stop_session_owned_process() {
  local pid=$1
  local pid_file=$2
  local expected_command=$3
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 0
  if [[ -r /proc/$pid/cmdline ]]; then
    local command
    command=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    [[ $command == *"$expected_command"* ]] || {
      echo "mapping session cleanup refused unexpected PID $pid: $command" >&2
      return 1
    }
    kill -TERM "$pid" 2>/dev/null || true
    local attempt
    for ((attempt = 0; attempt < 100; attempt++)); do
      [[ -e /proc/$pid ]] || break
      sleep 0.1
    done
    if [[ -r /proc/$pid/cmdline ]]; then
      command=$(tr '\0' ' ' <"/proc/$pid/cmdline")
      if [[ $command == *"$expected_command"* ]]; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  fi
  if [[ -s $pid_file && $(<"$pid_file") == "$pid" ]]; then
    rm -f "$pid_file"
  fi
}

cleanup_session_owned_processes() {
  if [[ $live_started_by_session == 1 ]]; then
    stop_session_owned_process "$live_session_pid" "$live_pid_file" "live_keyframe_producer.py" || true
  fi
  if [[ $lio_started_by_session == 1 ]]; then
    stop_session_owned_process "$lio_session_pid" "$lio_pid_file" "start_go2_lio_preview.sh" || true
  fi
}

trap cleanup_session_owned_processes EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

wait_for_pid_file() {
  local pid_file=$1
  local log_file=$2
  local expected_command=$3
  local attempts=0
  while ! pid_matches_command "$pid_file" "$expected_command"; do
    if [[ $attempts -ge 80 ]]; then
      echo "mapping session refused: service did not become ready: $pid_file" >&2
      tail -n 100 "$log_file" >&2 || true
      return 1
    fi
    sleep 0.5
    attempts=$((attempts + 1))
  done
}

wait_for_lio_pose() {
  local liveness="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
  set +u
  source /opt/ros/foxy/setup.bash
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp} \
    python3 "$liveness" --topic "${GO2_MAPPING_LIO_POSE_TOPIC:-/go2/lio/pose}" \
      --message-type geometry_msgs/msg/PoseWithCovarianceStamped --timeout 45
}

wait_for_livox_lio_input() {
  local liveness="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
  set +u
  source /opt/ros/foxy/setup.bash
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp} \
    python3 "$liveness" --topic "${GO2_MAPPING_LIO_LIVOX_TOPIC:-/go2/lio/livox}" \
      --message-type livox_ros_driver2/msg/CustomMsg --timeout 30
}

if ! curl --fail --silent --show-error "$high_fidelity_url/health" >/dev/null; then
  echo "high-fidelity receiver unavailable; recording continues locally and upload remains pending" >&2
fi

if [[ $mode == full ]]; then
  mkdir -p "$runtime_dir"
  if ! pgrep -f 'go2_native_pointcloud_socket_server.*--source-topic rt/utlidar/cloud' >/dev/null 2>&1; then
    nohup bash "$script_dir/start_go2_unitree_pointcloud_bridge.sh" \
      >"$runtime_dir/pointcloud-bridge.session.log" 2>&1 < /dev/null &
  fi
  wait_for_process 'go2_native_pointcloud_socket_server.*--source-topic rt/utlidar/cloud' \
    "$runtime_dir/pointcloud-bridge.session.log"
  if ! pgrep -f 'go2_native_lowstate_imu_socket_server' >/dev/null 2>&1; then
    nohup bash "$script_dir/start_go2_unitree_lowstate_imu_bridge.sh" \
      >"$runtime_dir/lowstate-imu-bridge.session.log" 2>&1 < /dev/null &
  fi
  wait_for_process 'go2_native_lowstate_imu_socket_server' "$runtime_dir/lowstate-imu-bridge.session.log"

  if [[ -n ${GO2_MAPPING_MISSION_ID:-} || -n ${GO2_MAPPING_TASK_ID:-} || -n ${GO2_MAPPING_TASK_LEASE_ID:-} ]]; then
    if ! pgrep -f 'navigation_executor.py' >/dev/null 2>&1; then
      nohup bash "$script_dir/start_go2_nav_bridge.sh" \
        >"$runtime_dir/nav-bridge.session.log" 2>&1 < /dev/null &
    fi
    wait_for_process 'navigation_executor.py' "$runtime_dir/nav-bridge.session.log"
  fi

  if [[ $live_enabled == 1 || $progressive_upload == 1 ]]; then
    export GO2_MAPPING_CALIBRATION_ID=${GO2_MAPPING_CALIBRATION_ID:-PROVISIONAL_MID360_IMU_EXTRINSIC_UNVERIFIED_v1}
    wait_for_livox_lio_input
    if ! pid_matches_command "$lio_pid_file" "start_go2_lio_preview.sh"; then
      rm -f "$lio_pid_file"
      nohup bash "$script_dir/start_go2_lio_preview.sh" \
        >"$runtime_dir/lio-preview.session.log" 2>&1 < /dev/null &
      lio_session_pid=$!
      lio_started_by_session=1
    fi
    wait_for_pid_file "$lio_pid_file" "$runtime_dir/lio-preview.session.log" "start_go2_lio_preview.sh"
    if [[ $lio_started_by_session == 1 && $(<"$lio_pid_file") != "$lio_session_pid" ]]; then
      echo "mapping session refused: LIO PID ownership changed during startup" >&2
      exit 5
    fi
    wait_for_lio_pose

    if [[ $live_enabled == 1 ]]; then
      live_identity_matches=0
      if pid_matches_command "$live_pid_file" "live_keyframe_producer.py"; then
        if [[ -z ${GO2_MAPPING_CAPTURE_SESSION_ID:-} ]] || \
          pid_matches_command "$live_pid_file" "--session-id $GO2_MAPPING_CAPTURE_SESSION_ID"; then
          live_identity_matches=1
        else
          stale_live_pid=$(<"$live_pid_file")
          stop_session_owned_process "$stale_live_pid" "$live_pid_file" \
            "live_keyframe_producer.py"
        fi
      fi
      if [[ $live_identity_matches == 0 ]]; then
        rm -f "$live_pid_file"
        nohup bash "$script_dir/start_go2_live_preview_stream.sh" \
          >"$runtime_dir/live-preview.session.log" 2>&1 < /dev/null &
        live_session_pid=$!
        live_started_by_session=1
        printf '%s\n' "$live_session_pid" >"$live_pid_file"
      fi
      wait_for_pid_file "$live_pid_file" "$runtime_dir/live-preview.session.log" "live_keyframe_producer.py"
      if [[ $live_started_by_session == 1 && $(<"$live_pid_file") != "$live_session_pid" ]]; then
        echo "mapping session refused: live preview PID ownership changed during startup" >&2
        exit 5
      fi
    fi
  fi
fi

set +e
capture_output=$(bash "$script_dir/start_go2_mapping_capture.sh" "$duration_s" "$mode" 2>&1)
capture_status=$?
set -e
printf '%s\n' "$capture_output"
if [[ $capture_status -ne 0 ]]; then
  echo "mapping session capture failed; local robot data was not deleted" >&2
  exit "$capture_status"
fi

segment=$(sed -n 's/^CAPTURE segment=\([^ ]*\).*/\1/p' <<<"$capture_output" | tail -n 1)
[[ -n $segment && -d $segment ]] || { echo "capture completed without a discoverable segment" >&2; exit 4; }
[[ $segment == "$data_root"/* ]] || { echo "segment outside configured data root: $segment" >&2; exit 4; }
[[ ! -e "$segment/DEPRECATED.json" ]] || {
  echo "mapping session refused to publish deprecated segment: $segment" >&2
  exit 4
}

if [[ $progressive_upload == 1 ]]; then
  printf 'MAPPING_SESSION_PROGRESSIVE_COMPLETE segment=%s mode=%s preview_policy=realtime_priority high_fidelity=remaining_capacity\n' \
    "$segment" "$mode"
  exit 0
fi

mapfile -d '' artifacts < <(find "$segment" -type f -print0 | sort -z)
manifest="$segment.sha256"
[[ -f $manifest ]] && artifacts+=("$manifest")
[[ ${#artifacts[@]} -gt 0 ]] || { echo "closed segment has no artifacts" >&2; exit 4; }

for artifact in "${artifacts[@]}"; do
  relative=${artifact#"$data_root"/}
  bash "$script_dir/upload_go2_segment_via_tunnel.sh" "$artifact" "$relative"
done

printf 'MAPPING_SESSION_COMPLETE segment=%s artifacts=%d mode=%s\n' "$segment" "${#artifacts[@]}" "$mode"
