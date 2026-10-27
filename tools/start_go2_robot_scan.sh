#!/usr/bin/env bash
# One-command Go2 scan entrypoint. It never launches or restarts motion or camera nodes.
set -Eeuo pipefail

action=${1:-start}
duration_s=${2:-${GO2_MAPPING_SCAN_DURATION_S:-3600}}
mode=${3:-${GO2_MAPPING_SCAN_MODE:-full}}
requested_session_id=${GO2_MAPPING_CAPTURE_SESSION_ID:-}
coordinate_scope=${GO2_MAPPING_COORDINATE_SCOPE:-GLOBAL_MAP}
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
pid_file="$runtime_dir/robot-scan.pid"
log_file="$runtime_dir/robot-scan.log"
session_file="$runtime_dir/robot-scan.session"
lock_file="$runtime_dir/robot-scan.lock"
runner="$release_root/tools/run_go2_mapping_session.sh"
pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}
session_endpoint=${GO2_MAPPING_SESSION_ENDPOINT:-http://${pc_tailnet_ip}:8765/api/robot/session}
live_endpoint=${GO2_MAPPING_LIVE_ENDPOINT:-ws://${pc_tailnet_ip}:8765/ws/live/keyframe}
live_health_url=${GO2_MAPPING_LIVE_HEALTH_URL:-http://${pc_tailnet_ip}:8765/api/live/readiness}
heartbeat_interval_s=2
# Keep direct Tailnet control and business requests outside ambient proxies.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${pc_tailnet_ip}"
export no_proxy="${no_proxy:+$no_proxy,}${pc_tailnet_ip}"
mkdir -p "$runtime_dir"

pid_is_alive() {
  [[ -s $pid_file ]] || return 1
  local pid
  pid=$(<"$pid_file")
  [[ $pid =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null
}

show_status() {
  if pid_is_alive; then
    printf 'GO2_ROBOT_SCAN_RUNNING pid=%s session=%s\n' "$(<"$pid_file")" "$(<"$session_file")"
    printf 'log=%s\n' "$log_file"
  else
    printf 'GO2_ROBOT_SCAN_STOPPED\n'
    if [[ -f $log_file ]]; then
      tail -n 20 "$log_file"
    fi
  fi
}

post_session_state() {
  local state=$1
  local session_id=$2
  local payload
  payload=$(printf '{"session_id":"%s","mode":"%s","state":"%s","capture_limit_gib":%s,"preview_policy":"realtime_preview_priority_high_fidelity_remaining_capacity"}' \
    "$session_id" "$mode" "$state" "${GO2_MAPPING_CAPTURE_MAX_GIB:-10}")
  if ! curl --max-time 4 --fail --silent --show-error -X POST \
    -H 'Content-Type: application/json' --data-binary "$payload" "$session_endpoint" >/dev/null; then
    printf 'session control state update failed: %s\n' "$state" >&2
    return 0
  fi
}

if [[ $action == supervise ]]; then
  session_id=${4:-}
  [[ $session_id =~ ^segment-[0-9]{8}T[0-9]{6}Z$ ]] || { echo 'supervisor session id is invalid' >&2; exit 2; }
  runner_pid=""
  heartbeat_pid=""
  result_dir="$runtime_dir/session-control/results"
  result_file="$result_dir/$session_id.json"
  phase_file="$runtime_dir/session-control/phases/$session_id.state"
  mkdir -p "$(dirname "$phase_file")"
  printf 'STARTING\n' >"$phase_file"
  result_written=0
  write_session_result() {
    local exit_code=$1
    local state=FAILED
    [[ $exit_code == 0 ]] && state=COMPLETED
    python3 - "$result_file" "$session_id" "$state" "$exit_code" <<'PY'
import json
import os
from pathlib import Path
import sys
import uuid

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "session_id": sys.argv[2],
    "state": sys.argv[3],
    "exit_code": int(sys.argv[4]),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as output:
        descriptor = -1
        output.write((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    if descriptor >= 0:
        os.close(descriptor)
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
PY
    result_written=1
  }
  runner_process_group_isolated() {
    local candidate=$1
    [[ $candidate =~ ^[1-9][0-9]*$ && -r /proc/$candidate/cmdline ]] || return 1
    local command
    command=$(tr '\0' ' ' <"/proc/$candidate/cmdline")
    [[ $command == *"bash $runner $duration_s $mode"* ]] || return 1
    local process_group
    process_group=$(ps -o pgid= -p "$candidate" 2>/dev/null | tr -d '[:space:]')
    [[ $process_group == "$candidate" ]]
  }
  cleanup_supervisor() {
    local supervisor_status=$?
    set +e
    if runner_process_group_isolated "$runner_pid"; then
      kill -TERM -- "-$runner_pid" 2>/dev/null || true
    fi
    [[ $heartbeat_pid =~ ^[1-9][0-9]*$ ]] && kill -TERM "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    if [[ $result_written == 0 ]]; then
      write_session_result "$supervisor_status" || true
    fi
    post_session_state STOPPED "$session_id" || true
    rm -f -- "$phase_file"
    return "$supervisor_status"
  }
  trap cleanup_supervisor EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  (
    while true; do
      heartbeat_state=STARTING
      if [[ -s $phase_file ]]; then
        read -r heartbeat_state <"$phase_file" || heartbeat_state=STARTING
      fi
      [[ $heartbeat_state == STARTING || $heartbeat_state == RECORDING || $heartbeat_state == FINALIZING ]] || heartbeat_state=STARTING
      post_session_state "$heartbeat_state" "$session_id" || true
      sleep "$heartbeat_interval_s"
    done
  ) &
  heartbeat_pid=$!
  GO2_MAPPING_SESSION_PHASE_FILE="$phase_file" setsid bash "$runner" "$duration_s" "$mode" &
  runner_pid=$!
  if ! runner_process_group_isolated "$runner_pid"; then
    echo "mapping session supervisor observed a non-isolated runner process group; continuing to preserve the runner exit result" >&2
  fi
  set +e
  wait "$runner_pid"
  status=$?
  set -e
  runner_pid=""
  write_session_result "$status"
  exit "$status"
fi

if [[ $action == status ]]; then
  show_status
  exit 0
fi
if [[ $action != start ]]; then
  echo 'usage: start_go2_robot_scan.sh [start|status] [duration_seconds] [full|baseline]' >&2
  exit 2
fi
[[ $duration_s =~ ^[1-9][0-9]*$ ]] || { echo 'scan duration must be a positive integer' >&2; exit 2; }
[[ $mode == full || $mode == baseline ]] || { echo 'scan mode must be full or baseline' >&2; exit 2; }
[[ $coordinate_scope == GLOBAL_MAP || $coordinate_scope == SESSION_ODOM_LOCAL ]] || {
  echo 'GO2_MAPPING_COORDINATE_SCOPE must be GLOBAL_MAP or SESSION_ODOM_LOCAL' >&2; exit 2;
}
[[ $coordinate_scope != SESSION_ODOM_LOCAL || $mode == full ]] || {
  echo 'SESSION_ODOM_LOCAL requires full scan mode' >&2; exit 2;
}
[[ -f $runner ]] || { echo "mapping session runner is missing: $runner" >&2; exit 3; }

exec 9>"$lock_file"
flock -n 9 || { show_status; exit 0; }
if pid_is_alive; then
  show_status
  exit 0
fi
rm -f "$pid_file"

high_fidelity_url=${GO2_MAPPING_HIGH_FIDELITY_URL:-http://192.0.2.10:8000}
bulk_token=${GO2_MAPPING_BULK_AUTH_TOKEN:-}
bulk_token_file=${GO2_MAPPING_BULK_TOKEN_FILE:-/home/unitree/go2_mapping/runtime/bulk-auth-token}
if [[ -z $bulk_token && -r $bulk_token_file ]]; then
  bulk_token=$(<"$bulk_token_file")
fi
health_auth=()
[[ -n $bulk_token ]] && health_auth=(-H "Authorization: Bearer $bulk_token")
high_fidelity_available=1
if ! curl --connect-timeout 3 --max-time 6 --fail --silent --show-error "${health_auth[@]}" "$high_fidelity_url/health" >/dev/null; then
  high_fidelity_available=0
  echo "PC high-fidelity receiver is unavailable; raw recording will remain buffered locally: $high_fidelity_url" >&2
fi
live_backend_available=1
if ! curl --connect-timeout 3 --max-time 6 --fail --silent --show-error "$live_health_url" >/dev/null; then
  live_backend_available=0
  echo 'PC live reconstruction backend is unavailable; live preview is disabled for this capture' >&2
fi

if [[ -n $requested_session_id ]]; then
  [[ $requested_session_id =~ ^segment-[0-9]{8}T[0-9]{6}Z$ ]] || {
    echo 'GO2_MAPPING_CAPTURE_SESSION_ID must match segment-YYYYMMDDTHHMMSSZ' >&2
    exit 2
  }
  session_id="$requested_session_id"
else
  session_id="segment-$(date -u +%Y%m%dT%H%M%SZ)"
fi
printf '%s\n' "$session_id" >"$session_file"
: >"$log_file"
live_enabled=0
progressive_upload=0
if [[ $mode == full ]]; then
  live_enabled=$live_backend_available
  progressive_upload=1
fi
export GO2_MAPPING_LIVE_ENABLED="$live_enabled"
export GO2_MAPPING_PROGRESSIVE_UPLOAD="$progressive_upload"
export GO2_MAPPING_ROTATE_MAX_BYTES=${GO2_MAPPING_ROTATE_MAX_BYTES:-8388608}
export GO2_MAPPING_UPLINK_BUDGET_MBPS=${GO2_MAPPING_UPLINK_BUDGET_MBPS:-30}
export GO2_MAPPING_CAPTURE_MAX_GIB=${GO2_MAPPING_CAPTURE_MAX_GIB:-10}
export GO2_MAPPING_CAPTURE_SESSION_ID="$session_id"
export GO2_MAPPING_COORDINATE_SCOPE="$coordinate_scope"
export GO2_MAPPING_BUILDING_ID=${GO2_MAPPING_BUILDING_ID:-${GO2_MAPPING_MAP_ID:-go2-live}}
calibration_id=${GO2_MAPPING_CALIBRATION_ID:-PROVISIONAL_AUTOCAL_PENDING_v1}
calibration_sha256=${GO2_MAPPING_CALIBRATION_SHA256:-}
calibration_manifest="/home/unitree/go2_mapping/config/calibration_manifest.json"
if [[ $calibration_id != PROVISIONAL_AUTOCAL_PENDING_v1 ]]; then
  if [[ ! -f $calibration_manifest ]]; then
    echo "ERROR: Calibration manifest not found: $calibration_manifest" >&2
    exit 3
  fi
  if [[ -n $calibration_sha256 ]]; then
    actual_sha256=$(sha256sum "$calibration_manifest" | awk '{print $1}')
    if [[ $actual_sha256 != "$calibration_sha256" ]]; then
      echo "ERROR: Calibration manifest SHA256 mismatch" >&2
      echo "  Expected: $calibration_sha256" >&2
      echo "  Actual:   $actual_sha256" >&2
      exit 3
    fi
  fi
fi
export GO2_MAPPING_CALIBRATION_ID="$calibration_id"
export GO2_MAPPING_CALIBRATION_MANIFEST="$calibration_manifest"
export GO2_MAPPING_CAMERA_ASSIST=1
export GO2_MAPPING_CAMERA_MAX_HZ=${GO2_MAPPING_CAMERA_MAX_HZ:-5}
export GO2_MAPPING_DETAIL_SURFACE_CELL_M=${GO2_MAPPING_DETAIL_SURFACE_CELL_M:-0.005}
export GO2_MAPPING_DETAIL_PAYLOAD_LIMIT_BYTES=${GO2_MAPPING_DETAIL_PAYLOAD_LIMIT_BYTES:-4194304}
export GO2_MAPPING_DETAIL_TARGET_HZ=${GO2_MAPPING_DETAIL_TARGET_HZ:-5}
export GO2_MAPPING_DETAIL_MINIMUM_HZ=${GO2_MAPPING_DETAIL_MINIMUM_HZ:-1}
export GO2_MAPPING_LIVE_DELIVERY_WINDOW=${GO2_MAPPING_LIVE_DELIVERY_WINDOW:-20}
export GO2_MAPPING_LIVE_PREVIEW_QUEUE_SIZE=${GO2_MAPPING_LIVE_PREVIEW_QUEUE_SIZE:-64}
export GO2_MAPPING_LIVE_LIDAR_TOPIC=${GO2_MAPPING_LIVE_LIDAR_TOPIC:-/lio_sam_ros2/deskew/cloud_deskewed}
export GO2_MAPPING_LIVE_POINTS_IN_MAP_FRAME=${GO2_MAPPING_LIVE_POINTS_IN_MAP_FRAME:-0}
export GO2_MAPPING_LIVE_POSE_CLOUD_MAX_SKEW_MS=${GO2_MAPPING_LIVE_POSE_CLOUD_MAX_SKEW_MS:-100}
export GO2_MAPPING_LIO_REGISTERED_CLOUD_TOPIC=${GO2_MAPPING_LIO_REGISTERED_CLOUD_TOPIC:-/lio_sam_ros2/mapping/cloud_registered}
export GO2_MAPPING_LIO_MAPPING_ODOMETRY_TOPIC=${GO2_MAPPING_LIO_MAPPING_ODOMETRY_TOPIC:-/lio_sam_ros2/mapping/odometry}
export GO2_MAPPING_LIO_GLOBAL_MAP_TOPIC=${GO2_MAPPING_LIO_GLOBAL_MAP_TOPIC:-/go2/mapping/map_cloud}
export GO2_MAPPING_CAMERA_MAX_WIDTH=${GO2_MAPPING_CAMERA_MAX_WIDTH:-640}
export GO2_MAPPING_CAMERA_MAX_HEIGHT=${GO2_MAPPING_CAMERA_MAX_HEIGHT:-480}
export GO2_MAPPING_CAMERA_DEPTH_MAX_WIDTH=${GO2_MAPPING_CAMERA_DEPTH_MAX_WIDTH:-320}
export GO2_MAPPING_CAMERA_DEPTH_MAX_HEIGHT=${GO2_MAPPING_CAMERA_DEPTH_MAX_HEIGHT:-240}
export GO2_MAPPING_CAMERA_DEPTH_ALIGNED=0
export GO2_MAPPING_CAMERA_DEPTH_TOPIC=/camera/depth/image_rect_raw
export GO2_MAPPING_CAMERA_RAW_DEPTH_TOPIC=/camera/depth/image_rect_raw
export GO2_MAPPING_LIVE_ENDPOINT="$live_endpoint"
export GO2_MAPPING_LIVE_HEALTH_URL="$live_health_url"
export GO2_MAPPING_HIGH_FIDELITY_URL="$high_fidelity_url"
export GO2_MAPPING_BULK_AUTH_TOKEN="$bulk_token"
export GO2_MAPPING_UPLOAD_CHUNK_KIB=${GO2_MAPPING_UPLOAD_CHUNK_KIB:-8192}
export GO2_MAPPING_UPLOAD_WORKERS=${GO2_MAPPING_UPLOAD_WORKERS:-4}
export GO2_MAPPING_DELETE_RECONSTRUCTED=${GO2_MAPPING_DELETE_RECONSTRUCTED:-1}
nohup bash "$release_root/tools/start_go2_robot_scan.sh" supervise "$duration_s" "$mode" "$session_id" \
  9>&- >>"$log_file" 2>&1 < /dev/null &
scan_pid=$!
printf '%s\n' "$scan_pid" >"$pid_file"
sleep 2
if ! kill -0 "$scan_pid" 2>/dev/null; then
  tail -n 100 "$log_file" >&2 || true
  rm -f "$pid_file"
  exit 5
fi

printf 'GO2_ROBOT_SCAN_STARTED pid=%s session=%s duration_s=%s mode=%s\n' \
  "$scan_pid" "$session_id" "$duration_s" "$mode"
printf 'preview_policy=realtime_priority high_fidelity=remaining_capacity capture_limit_gib=%s\n' \
  "${GO2_MAPPING_CAPTURE_MAX_GIB:-10}"
printf 'live_available=%s high_fidelity_available=%s local_buffering=%s\n' \
  "$live_backend_available" "$high_fidelity_available" "$((1 - high_fidelity_available))"
printf 'status: bash %s status\n' "$release_root/tools/start_go2_robot_scan.sh"
