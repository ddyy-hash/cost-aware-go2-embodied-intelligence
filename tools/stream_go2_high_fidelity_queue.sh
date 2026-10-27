#!/usr/bin/env bash
# Streams only immutable ROS bag parts through the high-fidelity queue.
set -Eeuo pipefail

finalize=0
if [[ ${1:-} == --finalize ]]; then
  finalize=1
  shift
fi
segment=${1:?usage: stream_go2_high_fidelity_queue.sh [--finalize] SEGMENT ROOT}
root=${2:?usage: stream_go2_high_fidelity_queue.sh [--finalize] SEGMENT ROOT}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
uploader=${GO2_MAPPING_UPLOADER:-$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/progressive_artifact_uploader.py}
high_fidelity_url=${GO2_MAPPING_HIGH_FIDELITY_URL:-http://192.0.2.10:8000}
runtime_dir=${GO2_MAPPING_RUNTIME_DIR:-/home/unitree/go2_mapping/runtime}
state_dir="$runtime_dir/high-fidelity-upload-state"
session_id=${GO2_MAPPING_PROGRESSIVE_SESSION_ID:-$(basename "$segment")}
rate_mbps=${GO2_MAPPING_HIGH_FIDELITY_RATE_MBPS:?GO2_MAPPING_HIGH_FIDELITY_RATE_MBPS is required}
total_ceiling_mbps=${GO2_MAPPING_UPLINK_BUDGET_MBPS:?GO2_MAPPING_UPLINK_BUDGET_MBPS is required}
route_id=${GO2_MAPPING_ROUTE_ID:-tailnet:unknown}
route_identity_tool="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/route_identity.py"
pc_tailnet_ip=${GO2_MAPPING_PC_TAILNET_IP:-192.0.2.10}
bandwidth_state=${GO2_MAPPING_BANDWIDTH_STATE:-$runtime_dir/bandwidth-allocation.json}
preview_demand_state=${GO2_MAPPING_PREVIEW_DEMAND_STATE:-$runtime_dir/preview-demand.json}
chunk_kib=${GO2_MAPPING_UPLOAD_CHUNK_KIB:-8192}
parallel_workers=${GO2_MAPPING_UPLOAD_WORKERS:-4}
probe_interval_seconds=${GO2_MAPPING_CAPACITY_PROBE_INTERVAL_SECONDS:-30}
bulk_auth_token=${GO2_MAPPING_BULK_AUTH_TOKEN:-}
delete_reconstructed=${GO2_MAPPING_DELETE_RECONSTRUCTED:-0}
poll_seconds=${GO2_MAPPING_PROGRESSIVE_POLL_SECONDS:-1}
retry_attempts=${GO2_MAPPING_UPLOAD_RETRY_ATTEMPTS:-8}
retry_initial_seconds=${GO2_MAPPING_UPLOAD_RETRY_INITIAL_SECONDS:-2}
retry_max_seconds=${GO2_MAPPING_UPLOAD_RETRY_MAX_SECONDS:-20}
priority_part_min_index=${GO2_MAPPING_PRIORITY_PART_MIN_INDEX:-}

[[ $segment == "$root"/* ]] || { echo "high-fidelity queue: segment is outside root: $segment" >&2; exit 2; }
deprecated_marker="$segment/DEPRECATED.json"
if [[ -e $deprecated_marker ]]; then
  echo "high-fidelity queue: refusing deprecated segment: $segment" >&2
  exit 3
fi
[[ -f $uploader ]] || { echo "high-fidelity queue: missing uploader: $uploader" >&2; exit 2; }
[[ -f $route_identity_tool ]] || { echo "high-fidelity queue: missing route identity tool: $route_identity_tool" >&2; exit 2; }
[[ $rate_mbps =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "high-fidelity queue: rate must be positive" >&2; exit 2; }
[[ $chunk_kib =~ ^[1-9][0-9]*$ ]] || { echo "high-fidelity queue: chunk size must be a positive integer" >&2; exit 2; }
[[ $parallel_workers =~ ^[1-8]$ ]] || { echo "high-fidelity queue: workers must be between 1 and 8" >&2; exit 2; }
[[ $probe_interval_seconds =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || { echo "high-fidelity queue: probe interval must be positive" >&2; exit 2; }
[[ $delete_reconstructed == 0 || $delete_reconstructed == 1 ]] || { echo "high-fidelity queue: delete policy must be 0 or 1" >&2; exit 2; }
[[ $poll_seconds =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "high-fidelity queue: poll interval must be positive" >&2; exit 2; }
[[ -z $priority_part_min_index || $priority_part_min_index =~ ^[0-9]+$ ]] || {
  echo "high-fidelity queue: priority part minimum index must be a non-negative integer" >&2
  exit 2
}
for value in "$retry_attempts" "$retry_initial_seconds" "$retry_max_seconds"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "high-fidelity queue: retry values must be positive integers" >&2; exit 2; }
done
(( retry_max_seconds >= retry_initial_seconds )) || {
  echo "high-fidelity queue: retry maximum must not be below the initial delay" >&2; exit 2;
}

mkdir -p "$state_dir"
state_file="$state_dir/${session_id}.uploaded"
touch "$state_file"
active_uploader_pid=""
active_uploader_output=""

stop_active_uploader() {
  if [[ $active_uploader_pid =~ ^[1-9][0-9]*$ ]] && kill -0 "$active_uploader_pid" 2>/dev/null; then
    kill -TERM "$active_uploader_pid" 2>/dev/null || true
    local attempt
    for ((attempt = 0; attempt < 50; attempt++)); do
      [[ -e /proc/$active_uploader_pid ]] || break
      sleep 0.1
    done
    kill -KILL "$active_uploader_pid" 2>/dev/null || true
    wait "$active_uploader_pid" 2>/dev/null || true
  fi
  [[ -z $active_uploader_output ]] || rm -f -- "$active_uploader_output"
  active_uploader_pid=""
  active_uploader_output=""
}

trap stop_active_uploader EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

current_route_identity() {
  local wifi_identity=unknown detected=""
  if command -v nmcli >/dev/null 2>&1; then
    wifi_identity=$(nmcli -t -f NAME,TYPE connection show --active 2>/dev/null \
      | awk -F: '$2 == "802-11-wireless" {print $1; exit}')
    wifi_identity=${wifi_identity:-unknown}
  fi
  if command -v tailscale >/dev/null 2>&1; then
    detected=$(tailscale status --json 2>/dev/null \
      | python3 "$route_identity_tool" --peer-ip "$pc_tailnet_ip" --wifi-identity "$wifi_identity" 2>/dev/null) || true
  fi
  printf '%s\n' "${detected:-$route_id}"
}

uploaded() {
  local relative=$1
  grep -Fqx -- "$relative" "$state_file"
}

closed_bag_part_index() {
  local name=${1##*/}
  if [[ $name =~ _([0-9]+)[.]db3[.]zstd$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

upload_file() {
  local file=$1
  local relative=${file#"$root"/}
  local attempt=1
  local delay_seconds=$retry_initial_seconds
  local output digest size_bytes
  [[ $relative != "$file" && $relative != /* && $relative != *..* ]] || {
    echo "high-fidelity queue: unsafe artifact path: $file" >&2
    return 2
  }
  if uploaded "$relative"; then
    # Retained closed bags are immutable. Avoid retransmitting them after a
    # restart when their reconstruction-release deletion path is disabled.
    # Deletion-enabled runs still revalidate bags with the receiver.
    if [[ $file != *.db3.zstd || $delete_reconstructed == 0 ]]; then
      return 0
    fi
  fi
  while (( attempt <= retry_attempts )); do
    local measured_route_id
    measured_route_id=$(current_route_identity)
    auth_args=()
    [[ -n $bulk_auth_token ]] && auth_args=(--auth-token "$bulk_auth_token")
    active_uploader_output=$(mktemp "$state_dir/.uploader-output.XXXXXX")
    python3 "$uploader" "$file" "$relative" \
      --endpoint "$high_fidelity_url" --session-id "$session_id" --chunk-kib "$chunk_kib" --rate-mbps "$rate_mbps" \
      --bandwidth-state "$bandwidth_state" --total-ceiling-mbps "$total_ceiling_mbps" \
      --preview-demand-state "$preview_demand_state" \
      --route-id "$measured_route_id" --parallel-workers "$parallel_workers" \
      --capacity-probe-interval "$probe_interval_seconds" "${auth_args[@]}" >"$active_uploader_output" &
    active_uploader_pid=$!
    local uploader_status=0
    wait "$active_uploader_pid" || uploader_status=$?
    output=$(<"$active_uploader_output")
    rm -f -- "$active_uploader_output"
    active_uploader_pid=""
    active_uploader_output=""
    if [[ $uploader_status -eq 0 ]]; then
      uploaded "$relative" || printf '%s\n' "$relative" >>"$state_file"
      printf 'HIGH_FIDELITY_ARTIFACT_VERIFIED path=%s\n' "$relative"
      if [[ $delete_reconstructed == 1 && $file == *.db3.zstd ]]; then
        if [[ $output != *'"release_authorized": true'* || $output != *'"state": "PC_RECONSTRUCTION_INGESTED"'* ]]; then
          printf 'RECONSTRUCTION_RELEASE_PENDING path=%s\n' "$relative"
          return 0
        fi
        [[ -f $file && $file == "$segment"/* && $file == "$root"/* ]] || {
          echo "high-fidelity queue: refusing to delete an unsafe artifact path: $file" >&2
          return 2
        }
        digest=$(sha256sum -- "$file" | awk '{print $1}')
        size_bytes=$(stat -c '%s' -- "$file")
        rm -f -- "$file"
        [[ ! -e $file ]] || { echo "high-fidelity queue: release deletion failed: $relative" >&2; return 1; }
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$digest" "$size_bytes" "$relative" \
          >>"$state_dir/${session_id}.released.tsv"
        printf 'RECONSTRUCTION_RELEASE_AUTHORIZED path=%s sha256=%s bytes=%s\n' "$relative" "$digest" "$size_bytes"
      fi
      return 0
    fi
    if (( attempt == retry_attempts )); then
      echo "high-fidelity queue: upload failed after $attempt attempts: $relative" >&2
      return 1
    fi
    echo "high-fidelity queue: upload attempt $attempt failed for $relative; retrying in ${delay_seconds}s" >&2
    sleep "$delay_seconds"
    delay_seconds=$((delay_seconds * 2))
    (( delay_seconds > retry_max_seconds )) && delay_seconds=$retry_max_seconds
    attempt=$((attempt + 1))
  done
}

upload_closed_bags() {
  local entries=()
  local entry file part_index failures=0
  [[ -d $segment ]] || return 0
  [[ ! -e $deprecated_marker ]] || {
    echo "high-fidelity queue: refusing deprecated segment: $segment" >&2
    return 3
  }
  # File-level Zstd is produced only after rosbag has closed a rotated SQLite
  # part. Never infer closure from the part number or read a writable .db3.
  mapfile -d '' entries < <(find "$segment" -type f -name '*.db3.zstd' -printf '%T@:%p\0' | sort -z -n)
  for entry in "${entries[@]}"; do
    file=${entry#*:}
    if (( ! finalize )) && [[ -n $priority_part_min_index ]]; then
      part_index=$(closed_bag_part_index "$file") || {
        echo "high-fidelity queue: cannot derive bag part index: $file" >&2
        failures=1
        continue
      }
      if (( 10#$part_index < 10#$priority_part_min_index )); then
        continue
      fi
    fi
    if ! upload_file "$file"; then
      failures=1
      # A receiver restart or temporary tailnet outage must not kill the live
      # queue. Raw artifacts remain immutable on the robot and are retried on
      # the next poll. Finalization remains strict so the caller records that
      # upload is still pending.
      if (( ! finalize )); then
        echo "high-fidelity queue: artifact remains pending; keeping watcher alive: ${file#"$root"/}" >&2
      fi
    fi
  done
  return "$failures"
}

upload_final_metadata() {
  [[ ! -e $deprecated_marker ]] || {
    echo "high-fidelity queue: refusing deprecated segment: $segment" >&2
    return 3
  }
  local metadata="$segment/metadata.yaml"
  local manifest="$segment.sha256"
  local runtime="$segment/capture.runtime.json"
  [[ -f $metadata ]] && upload_file "$metadata"
  [[ -f $runtime ]] && upload_file "$runtime"
  [[ -f $manifest ]] && upload_file "$manifest"
  return 0
}

if (( finalize )); then
  [[ -d $segment ]] || { echo "high-fidelity queue: segment does not exist: $segment" >&2; exit 2; }
  upload_closed_bags || exit 1
  upload_final_metadata
  exit 0
fi

while true; do
  upload_closed_bags || true
  sleep "$poll_seconds"
done
