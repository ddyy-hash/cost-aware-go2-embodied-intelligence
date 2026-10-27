#!/usr/bin/env bash
# Resumable download and local reconstruction of the official MobileVLA-R1 checkpoint.
set -Eeuo pipefail

destination=${1:-models/mobilevla-r1-download}
repo_url=${MOBILEVLA_HF_REPO_URL:-https://huggingface.co/AIGeeksGroup/MobileVLA-R1/resolve/main}
workers=${MOBILEVLA_DOWNLOAD_WORKERS:-4}
[[ $workers =~ ^[1-9][0-9]*$ ]] || { echo 'MOBILEVLA_DOWNLOAD_WORKERS must be a positive integer' >&2; exit 2; }
mkdir -p -- "$destination/parts"

download_part() {
  local letter=$1
  local part="weight.zip.part-a$letter"
  local target="$destination/parts/$part"
  local url="$repo_url/$part"
  printf 'MOBILEVLA_DOWNLOAD part=%s target=%s\n' "$part" "$target"
  curl --fail --location --retry 8 --retry-delay 5 --connect-timeout 30 \
    --silent --show-error --progress-bar --continue-at - --output "$target" "$url"
}

letters=({a..z})
for ((offset=0; offset<${#letters[@]}; offset+=workers)); do
  pids=()
  for ((index=offset; index<offset+workers && index<${#letters[@]}; index++)); do
    download_part "${letters[$index]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || { echo "MobileVLA download batch failed near part index $offset" >&2; exit 1; }
done

archive="$destination/weight.zip"
cat "$destination"/parts/weight.zip.part-a? > "$archive"
extract_root="$destination/extracted"
rm -rf -- "$extract_root"
mkdir -p -- "$extract_root"
if command -v unzip >/dev/null 2>&1; then
  unzip -t "$archive" >/dev/null
  unzip -q "$archive" -d "$extract_root"
else
  python3 - "$archive" "$extract_root" <<'PY'
import sys
import zipfile

archive, destination = sys.argv[1:]
with zipfile.ZipFile(archive) as bundle:
    bad = bundle.testzip()
    if bad is not None:
        raise SystemExit("MobileVLA ZIP CRC check failed: " + bad)
    bundle.extractall(destination)
PY
fi
printf 'MOBILEVLA_CHECKPOINT_ARCHIVE_READY archive=%s extracted=%s\n' "$archive" "$extract_root"
