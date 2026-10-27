#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"

mode=${1:-run}
if [[ "$mode" == stop-owned || "$mode" == verify-owned ]]; then
  owned_pid=${2:-}
  if [[ ! "$owned_pid" =~ ^[1-9][0-9]*$ || ! -r "/proc/${owned_pid}/cmdline" ]]; then
    echo "owned reconstruction worker Linux PID is not running: ${owned_pid:-missing}" >&2
    exit 1
  fi
  owned_cwd=$(readlink -f -- "/proc/${owned_pid}/cwd")
  owned_command=$(tr '\0' ' ' <"/proc/${owned_pid}/cmdline")
  owned_pgid=$(ps -o pgid= -p "$owned_pid" | tr -d '[:space:]')
  if [[ "$owned_cwd" != "$root" || "$owned_pgid" != "$owned_pid" || "$owned_command" != *"tools/run_go2_automatic_reconstruction.py"* ]]; then
    echo "reconstruction worker ownership does not match this worktree and process group" >&2
    exit 1
  fi
  if [[ "$mode" == verify-owned ]]; then
    expected_session_id=${3:-}
    if [[ ! "$expected_session_id" =~ ^segment-[0-9]{8}T[0-9]{6}Z$ || "$owned_command" != *"--session-id ${expected_session_id}"* ]]; then
      echo "reconstruction worker ownership does not match the expected session" >&2
      exit 1
    fi
    exit 0
  fi
  kill -TERM -- "-${owned_pid}"
  exit 0
fi
if [[ "$mode" != run ]]; then
  echo "unsupported reconstruction launcher mode: $mode" >&2
  exit 2
fi

session_id=${2:?session ID is required}
inbox_root=${3:?inbox root is required}
output_root=${4:?output root is required}
status_path=${5:?status path is required}
sensor_frame_transform_id=${6:?sensor frame transform ID is required}
source_mode=${7:-RAW_LIDAR_WITH_RECORDED_LIO}
pid_file=${GO2_MAPPING_RECONSTRUCTION_PID_FILE:?GO2_MAPPING_RECONSTRUCTION_PID_FILE is required}

if [[ ! "$session_id" =~ ^segment-[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "session ID must match segment-YYYYMMDDTHHMMSSZ" >&2
  exit 2
fi
for path in "$inbox_root" "$output_root" "$status_path" "$pid_file"; do
  if [[ "$path" != /* ]]; then
    echo "reconstruction paths must be absolute: $path" >&2
    exit 2
  fi
done

mkdir -p -- "$inbox_root" "$output_root" "$(dirname -- "$status_path")" "$(dirname -- "$pid_file")"
pid_file_temporary="${pid_file}.tmp.$$"
printf '%s\n' "$$" >"$pid_file_temporary"
mv -f -- "$pid_file_temporary" "$pid_file"

# The worker performs incremental observed-only fusion, then its final mode is
# high_fidelity_batch_reconstruction after the verified manifest closes.
exec "${MAMBA_ROOT_PREFIX}/bin/micromamba" run -n go2-mapping-pc \
  python tools/run_go2_automatic_reconstruction.py \
  --session-id "$session_id" \
  --inbox-root "$inbox_root" \
  --output-root "$output_root" \
  --status-path "$status_path" \
  --sensor-frame-transform-id "$sensor_frame_transform_id" \
  --source-mode "$source_mode" \
  --poll-interval "${GO2_MAPPING_RECONSTRUCTION_POLL_INTERVAL:-1}"
