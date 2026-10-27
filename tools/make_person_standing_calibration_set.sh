#!/usr/bin/env bash
set -eo pipefail

output_dir="${1:-/home/unitree/go2_person_follow/calibration/person_standing_mixed_images}"
person_dir="${2:-/home/unitree/go2_person_follow/calibration/person_standing_images}"
base_dir="${3:-/home/unitree/go2_person_follow/calibration/images}"
test_dir="${4:-/home/unitree/go2_person_follow/test_images}"

rm -rf "$output_dir"
mkdir -p "$output_dir"

i=0
for image in "$person_dir"/calib_*.jpg; do
  if [ "$i" -ge 104 ]; then
    break
  fi
  cp "$image" "$output_dir/$(printf 'person_standing_%05d.jpg' "$i")"
  i=$((i + 1))
done

j=0
for image in "$base_dir"/calib_*.jpg; do
  if [ "$j" -ge 16 ]; then
    break
  fi
  cp "$image" "$output_dir/$(printf 'base_%05d.jpg' "$j")"
  j=$((j + 1))
done

for k in $(seq 0 3); do
  cp "$test_dir/bus.jpg" "$output_dir/$(printf 'person_bus_%05d.jpg' "$k")"
  cp "$test_dir/zidane.jpg" "$output_dir/$(printf 'person_zidane_%05d.jpg' "$k")"
done

find "$output_dir" -maxdepth 1 -type f | wc -l
