#!/usr/bin/env bash
set -eo pipefail

root_dir="${1:-/home/unitree/go2_person_follow}"
models_dir="$root_dir/models"
tools_dir="$root_dir/tools"
calib_dir="$root_dir/calibration"

onnx_path="$models_dir/yolov8n.onnx"
live_dir="$calib_dir/live_person_images"
mixed_dir="$calib_dir/live_person_mixed_images"
base_dir="$calib_dir/images"
test_dir="$root_dir/test_images"
cache_path="$models_dir/yolov8n_int8_live_person_calibration.cache"
probe_engine_path="$models_dir/yolov8n_int8_live_person_probe.engine"
engine_path="$models_dir/yolov8n_int8_live_person.engine"
cache_log="/tmp/yolov8n_int8_live_person_cache_build.log"
trtexec_log="/tmp/yolov8n_int8_live_person_trtexec.log"

echo "Building live person mixed calibration set"
"$tools_dir/make_person_standing_calibration_set.sh" \
  "$mixed_dir" \
  "$live_dir" \
  "$base_dir" \
  "$test_dir"

echo "Generating TensorRT INT8 calibration cache"
"$tools_dir/generate_int8_cache.sh" \
  "$onnx_path" \
  "$mixed_dir" \
  "$cache_path" \
  "$probe_engine_path" \
  "$cache_log"

echo "Building TensorRT INT8 engine"
rm -f "$engine_path" "$trtexec_log"
/usr/src/tensorrt/bin/trtexec \
  --onnx="$onnx_path" \
  --saveEngine="$engine_path" \
  --int8 \
  --calib="$cache_path" \
  --workspace=1024 \
  --duration=3 \
  --warmUp=200 \
  2>&1 | tee "$trtexec_log"

test -s "$engine_path"
echo "Engine ready: $engine_path"
