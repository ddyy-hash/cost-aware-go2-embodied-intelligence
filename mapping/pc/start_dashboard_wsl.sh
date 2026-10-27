#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"

mode=${1:-run}
if [[ "$mode" == stop-owned || "$mode" == verify-owned ]]; then
  owned_pid=${2:-}
  dashboard_port=${3:-8767}
  if [[ ! "$owned_pid" =~ ^[1-9][0-9]*$ || ! -r "/proc/${owned_pid}/cmdline" ]]; then
    echo "owned dashboard Linux PID is not running: ${owned_pid:-missing}" >&2
    exit 1
  fi
  owned_cwd=$(readlink -f -- "/proc/${owned_pid}/cwd")
  owned_command=$(tr '\0' ' ' <"/proc/${owned_pid}/cmdline")
  owned_pgid=$(ps -o pgid= -p "$owned_pid" | tr -d '[:space:]')
  if [[ "$owned_cwd" != "$root" || "$owned_pgid" != "$owned_pid" ]]; then
    echo "dashboard Linux ownership does not match this worktree and process group" >&2
    exit 1
  fi
  if [[ "$owned_command" != *"micromamba run -n go2-mapping-pc python -m uvicorn mapping.pc.go2_mapping_dashboard.server:app --host 0.0.0.0 --port ${dashboard_port}"* ]]; then
    echo "dashboard Linux command does not match the owned port-8765 service" >&2
    exit 1
  fi
  if [[ "$mode" == verify-owned ]]; then
    exit 0
  fi
  kill -TERM -- "-${owned_pid}"
  for _ in {1..300}; do
    if ! ss -H -ltn "sport = :${dashboard_port}" | grep -q .; then
      exit 0
    fi
    sleep 0.1
  done
  echo "owned dashboard did not release WSL port ${dashboard_port}" >&2
  exit 1
fi
if [[ "$mode" != run ]]; then
  echo "unsupported dashboard launcher mode: $mode" >&2
  exit 2
fi

pid_file=${GO2_MAPPING_DASHBOARD_PID_FILE:?GO2_MAPPING_DASHBOARD_PID_FILE is required}
dashboard_port=${GO2_MAPPING_DASHBOARD_PORT:-8767}
mkdir -p -- "$(dirname -- "$pid_file")"
pid_file_temporary="${pid_file}.tmp.$$"
printf '%s\n' "$$" >"$pid_file_temporary"
mv -f -- "$pid_file_temporary" "$pid_file"
# The caller owns this foreground process and records its Linux process group so
# a later restart can terminate only this worktree's dashboard.
exec "${MAMBA_ROOT_PREFIX}/bin/micromamba" run -n go2-mapping-pc \
  python -m uvicorn mapping.pc.go2_mapping_dashboard.server:app --host 0.0.0.0 --port "${dashboard_port}"
