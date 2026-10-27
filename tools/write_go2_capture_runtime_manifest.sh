#!/usr/bin/env bash
# Write immutable capture runtime provenance without touching ROS ownership.
set -Eeuo pipefail

segment=${1:?usage: write_go2_capture_runtime_manifest.sh SEGMENT MODE STARTED_AT FINISHED_AT [TOPIC=TYPE ...]}
mode=${2:?capture mode is required}
coordinate_scope=${GO2_MAPPING_COORDINATE_SCOPE:-GLOBAL_MAP}
started_at=${3:?capture start time is required}
finished_at=${4:?capture finish time is required}
shift 4

[[ -d $segment ]] || { echo "runtime manifest: capture segment does not exist: $segment" >&2; exit 2; }
[[ $mode == full || $mode == baseline || $mode == calibration ]] || {
  echo "runtime manifest: invalid capture mode: $mode" >&2; exit 2;
}
[[ $coordinate_scope == GLOBAL_MAP || $coordinate_scope == SESSION_ODOM_LOCAL ]] || {
  echo "runtime manifest: invalid coordinate scope: $coordinate_scope" >&2; exit 2;
}
[[ $# -gt 0 ]] || { echo "runtime manifest: at least one TOPIC=TYPE is required" >&2; exit 2; }

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
release_sha256=$(basename "$(readlink -f /home/unitree/go2_mapping/current 2>/dev/null || printf '%s' "$release_root")")
if [[ ! $release_sha256 =~ ^[0-9a-fA-F]{64}$ ]]; then
  release_sha256=$(sha256sum "$release_root/RELEASE.SHA256SUMS" 2>/dev/null | awk '{print $1}')
fi
[[ ${release_sha256:-} =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "runtime manifest: active release identity is unavailable" >&2; exit 3;
}

profile="$release_root/mapping/config/go2edu_realsense_d435i_camera_profile.json"
[[ -f $profile ]] || { echo "runtime manifest: camera profile is missing: $profile" >&2; exit 3; }
enumeration=""
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  enumeration=$(rs-enumerate-devices -s 2>/dev/null || true)
fi

robot_serial=${GO2_MAPPING_ROBOT_SERIAL:-}
robot_identity_source=operator
robot_factory_verified=true
if [[ -z $robot_serial ]]; then
  machine_id=$(tr -d '[:space:]' </etc/machine-id 2>/dev/null || true)
  [[ -n $machine_id ]] || { echo "runtime manifest: robot identity unavailable" >&2; exit 3; }
  robot_serial="go2-host-$(printf '%s' "$machine_id" | sha256sum | cut -c1-16)"
  robot_identity_source=machine_id_sha256
  robot_factory_verified=false
fi

depth_scale_m=${GO2_MAPPING_DEPTH_SCALE_M:-}
if [[ -z $depth_scale_m ]] && command -v ros2 >/dev/null 2>&1; then
  while IFS= read -r camera_node; do
    [[ -n $camera_node ]] || continue
    candidate=$(timeout 3s ros2 param get "$camera_node" depth_module.depth_units 2>/dev/null \
      | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^0[.][0-9]+$/) {print $i; exit}}' || true)
    if [[ -n ${candidate:-} ]]; then
      depth_scale_m=$candidate
      break
    fi
  done < <(timeout 3s ros2 node list 2>/dev/null | grep -E '(^|/)realsense2_camera$|(^|/)camera$' || true)
fi
if [[ -n $depth_scale_m ]] && [[ ! $depth_scale_m =~ ^0\.[0-9]+$ ]]; then
  echo "runtime manifest: GO2_MAPPING_DEPTH_SCALE_M must be a positive decimal in meters" >&2
  exit 3
fi

temporary="$segment/.capture.runtime.json.tmp"
output="$segment/capture.runtime.json"
python3 - "$profile" "$temporary" "$output" "$robot_serial" "$robot_identity_source" \
  "$robot_factory_verified" "$depth_scale_m" "$release_sha256" "$mode" "$started_at" "$finished_at" \
  "${ROS_DOMAIN_ID:-42}" "$enumeration" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys

profile_path, temporary_path, output_path = map(Path, sys.argv[1:4])
coordinate_scope = os.environ.get("GO2_MAPPING_COORDINATE_SCOPE", "GLOBAL_MAP")
robot_serial, robot_source = sys.argv[4:6]
robot_verified = sys.argv[6].lower() == "true"
depth_scale_m = sys.argv[7] or None
release_sha256, mode, started_at, finished_at = sys.argv[8:12]
ros_domain_id = int(sys.argv[12])
enumeration = sys.argv[13]
topic_arguments = sys.argv[14:]
profile = json.loads(profile_path.read_text(encoding="utf-8"))
camera = profile["device"]
camera_serial = str(camera["serial"])
if not enumeration or camera_serial not in enumeration:
    raise SystemExit(f"runtime manifest: configured RealSense {camera_serial} was not verified by rs-enumerate-devices")
topics = {}
for argument in topic_arguments:
    if "=" not in argument:
        raise SystemExit(f"runtime manifest: invalid TOPIC=TYPE argument: {argument}")
    topic, type_name = argument.split("=", 1)
    if not topic.startswith("/") or not type_name or topic in topics:
        raise SystemExit(f"runtime manifest: invalid/duplicate topic contract: {argument}")
    topics[topic] = type_name
payload = {
    "schema_version": 1,
    "robot": {
        "model": "go2_edu",
        "serial": robot_serial,
        "identity_source": robot_source,
        "factory_serial_verified": robot_verified,
    },
    "camera": {
        "model": str(camera["model"]),
        "serial": camera_serial,
        "firmware": str(camera["firmware"]),
        "identity_source": "rs-enumerate-devices+versioned_profile",
        **({"depth_scale_m": float(depth_scale_m)} if depth_scale_m is not None else {}),
    },
    "release_sha256": release_sha256.lower(),
    "capture_mode": mode,
    "coordinate_scope": coordinate_scope,
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
    "ros_domain_id": ros_domain_id,
    "topics": dict(sorted(topics.items())),
}
temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary_path.open("rb") as stream:
    os.fsync(stream.fileno())
os.replace(temporary_path, output_path)
PY
printf 'CAPTURE_RUNTIME_MANIFEST path=%s robot_serial=%s factory_serial_verified=%s\n' \
  "$output" "$robot_serial" "$robot_factory_verified"
