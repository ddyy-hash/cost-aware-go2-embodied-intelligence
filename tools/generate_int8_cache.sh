#!/usr/bin/env bash
set -eo pipefail

onnx="${1:-/home/unitree/go2_person_follow/models/yolov8n.onnx}"
calib_dir="${2:-/home/unitree/go2_person_follow/calibration/standing_images}"
cache="${3:-/home/unitree/go2_person_follow/models/yolov8n_int8_standing_calibration.cache}"
probe_engine="${4:-/home/unitree/go2_person_follow/models/yolov8n_int8_standing_probe.engine}"
log_file="${5:-/tmp/yolov8n_int8_standing_cache_build.log}"

rm -f "$cache" "$probe_engine" "$log_file"

python3 /home/unitree/go2_person_follow/tools/build_calibrated_yolo_engine.py \
  --onnx "$onnx" \
  --engine "$probe_engine" \
  --cache "$cache" \
  --calib-dir "$calib_dir" \
  --input-size 640 \
  --batch-size 1 \
  --max-images 128 \
  --workspace-mib 1024 \
  --fp16 \
  >"$log_file" 2>&1 &

builder_pid=$!
cache_seen=0
last_size=0

for _ in $(seq 1 900); do
  if ! kill -0 "$builder_pid" 2>/dev/null; then
    wait "$builder_pid"
    break
  fi

  if [ -s "$cache" ]; then
    current_size="$(wc -c <"$cache")"
    if [ "$cache_seen" -eq 1 ] && [ "$current_size" -eq "$last_size" ]; then
      kill "$builder_pid" 2>/dev/null || true
      wait "$builder_pid" 2>/dev/null || true
      break
    fi
    cache_seen=1
    last_size="$current_size"
  fi

  sleep 1
done

if ! [ -s "$cache" ]; then
  tail -80 "$log_file"
  exit 1
fi

ls -lh "$cache"
tail -20 "$log_file"
