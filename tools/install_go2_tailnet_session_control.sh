#!/usr/bin/env bash
# Maintenance-only installation of the authenticated robot Tailnet control service.
set -Eeuo pipefail

tailnet_ip=""
allowed_peer=""
token_source=""
robot_serial=""
port=8788

while (($#)); do
  case "$1" in
    --tailnet-ip) tailnet_ip=${2:-}; shift 2 ;;
    --allow-peer) allowed_peer=${2:-}; shift 2 ;;
    --token-file) token_source=${2:-}; shift 2 ;;
    --robot-serial) robot_serial=${2:-}; shift 2 ;;
    --port) port=${2:-}; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n $tailnet_ip && -n $allowed_peer && -n $token_source ]] || {
  echo 'usage: install_go2_tailnet_session_control.sh --tailnet-ip IP --allow-peer IP --token-file FILE [--robot-serial SERIAL] [--port PORT]' >&2
  exit 2
}
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && ((port <= 65535)) || { echo 'invalid control port' >&2; exit 2; }
[[ -r $token_source ]] || { echo "session-control token is unreadable: $token_source" >&2; exit 2; }

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
module="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/session_control.py"
unit_source="$release_root/tools/go2-mapping-session-control.service"
[[ -f $module && -f $unit_source ]] || { echo 'session-control release files are incomplete' >&2; exit 3; }
service_user=$(id -un)
[[ $service_user == unitree ]] || {
  echo "session-control installer must run as unitree, not $service_user" >&2
  exit 5
}

python3 - "$tailnet_ip" "$allowed_peer" <<'PY'
import ipaddress
import sys
network = ipaddress.ip_network("100.64.0.0/10")
for value in sys.argv[1:]:
    address = ipaddress.ip_address(value)
    if not isinstance(address, ipaddress.IPv4Address) or address not in network or str(address) != value:
        raise SystemExit("session-control addresses must be canonical Tailnet IPv4 values")
PY

actual_tailnet_ip=$(tailscale ip -4 | sed -n '1p')
[[ $actual_tailnet_ip == "$tailnet_ip" ]] || {
  echo "explicit robot Tailnet IP does not match tailscale ip -4: $tailnet_ip != $actual_tailnet_ip" >&2
  exit 4
}

runtime_dir=/home/unitree/go2_mapping/runtime
service_runtime="$runtime_dir/session-control"
token_target="$runtime_dir/session-control-token"
environment_file="$runtime_dir/session-control.env"
unit_dir=/home/unitree/.config/systemd/user
unit_target="$unit_dir/go2-mapping-session-control.service"
mkdir -p -- "$service_runtime" "$unit_dir"
chmod 0700 "$runtime_dir" "$service_runtime"

existing_robot_serial=""
if [[ -z $robot_serial && -r $environment_file ]]; then
  existing_robot_serial=$(sed -n 's/^GO2_MAPPING_ROBOT_SERIAL=//p' "$environment_file" | tail -n 1)
  robot_serial=$existing_robot_serial
fi
if [[ -n $robot_serial && ! $robot_serial =~ ^[A-Za-z0-9][A-Za-z0-9._-]{3,127}$ ]]; then
  echo 'robot factory serial must contain 4-128 ASCII letters, digits, dot, underscore, or hyphen' >&2
  exit 2
fi

token=$(tr -d '\r\n' <"$token_source")
(( ${#token} >= 32 && ${#token} <= 512 )) || { echo 'session-control token must contain 32-512 characters' >&2; exit 2; }
token_tmp="$token_target.$$.partial"
env_tmp="$environment_file.$$.partial"
unit_tmp="$unit_target.$$.partial"
trap 'rm -f -- "$token_tmp" "$env_tmp" "$unit_tmp"' EXIT
printf '%s\n' "$token" >"$token_tmp"
chmod 0600 "$token_tmp"
mv -f -- "$token_tmp" "$token_target"
{
  printf 'GO2_SESSION_CONTROL_BIND=%s\nGO2_SESSION_CONTROL_ALLOW_PEER=%s\nGO2_SESSION_CONTROL_PORT=%s\n' \
    "$tailnet_ip" "$allowed_peer" "$port"
  if [[ -n $robot_serial ]]; then
    printf 'GO2_MAPPING_ROBOT_SERIAL=%s\n' "$robot_serial"
  fi
} >"$env_tmp"
chmod 0600 "$env_tmp"
mv -f -- "$env_tmp" "$environment_file"
cp -- "$unit_source" "$unit_tmp"
chmod 0644 "$unit_tmp"
mv -f -- "$unit_tmp" "$unit_target"

linger=$(loginctl show-user "$service_user" -p Linger --value)
if [[ $linger != yes ]]; then
  loginctl enable-linger "$service_user"
  linger=$(loginctl show-user "$service_user" -p Linger --value)
fi
[[ $linger == yes ]] || {
  echo "persistent user manager was not enabled for $service_user" >&2
  exit 5
}
systemctl --user daemon-reload
systemctl --user enable --now go2-mapping-session-control.service
systemctl --user restart go2-mapping-session-control.service
systemctl --user is-active --quiet go2-mapping-session-control.service
http_status=000
for attempt in $(seq 1 100); do
  http_status=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' \
    "http://${tailnet_ip}:${port}/v1/preflight" || true)
  [[ $http_status == 401 ]] && break
  sleep 0.1
done
[[ $http_status == 401 ]] || {
  echo "session-control local authentication boundary returned HTTP $http_status instead of 401" >&2
  exit 5
}
printf 'GO2_TAILNET_SESSION_CONTROL_READY bind=%s allow_peer=%s port=%s\n' \
  "$tailnet_ip" "$allowed_peer" "$port"
