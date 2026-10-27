#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
log_file="${HOME}/go2-mapping-bootstrap.log"
pid_file="${HOME}/go2-mapping-bootstrap.pid"

if [[ -r $pid_file ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
  printf 'Bootstrap already running with PID %s. Log: %s\n' "$(<"$pid_file")" "$log_file"
  exit 0
fi

nohup bash "$script_dir/bootstrap_wsl.sh" >"$log_file" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
printf 'Started bootstrap PID %s. Log: %s\n' "$pid" "$log_file"
