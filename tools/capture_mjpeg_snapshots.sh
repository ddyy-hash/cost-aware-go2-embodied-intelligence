#!/usr/bin/env bash
set -eo pipefail

output_dir="${1:-/home/unitree/go2_person_follow/calibration/live_person_images}"
snapshot_url="${2:-http://127.0.0.1:8082/snapshot.jpg}"
count="${3:-128}"

rm -rf "$output_dir"
mkdir -p "$output_dir"

for i in $(seq 0 "$((count - 1))"); do
  output_path="$output_dir/$(printf 'calib_%05d.jpg' "$i")"
  curl -sf "$snapshot_url" -o "$output_path"
  sleep 0.1
done

find "$output_dir" -maxdepth 1 -type f -name 'calib_*.jpg' | wc -l
