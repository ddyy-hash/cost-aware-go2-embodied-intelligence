#!/usr/bin/env bash
# Install only pre-staged wheels; this path never contacts an index.
set -Eeuo pipefail

wheelhouse=${1:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../mapping/robot/wheelhouse-aarch64-cp38" && pwd)"}
manifest=${2:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../mapping/config" && pwd)/robot_wheelhouse_manifest.json"}
[[ -d $wheelhouse ]] || { echo "robot wheelhouse missing: $wheelhouse" >&2; exit 3; }
[[ -s $manifest ]] || { echo "robot wheelhouse manifest missing: $manifest" >&2; exit 3; }

for wheel in "$wheelhouse"/*.whl; do
  [[ -f $wheel ]] || { echo "robot wheelhouse contains no wheels: $wheelhouse" >&2; exit 3; }
done
python3 - "$manifest" "$wheelhouse" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1 or manifest.get("target") != "cp38-manylinux2014_aarch64":
    raise SystemExit("unsupported robot wheelhouse manifest")
root = Path(sys.argv[2])
expected = {item["filename"]: item for item in manifest.get("wheels", [])}
actual = {path.name: path for path in root.glob("*.whl")}
if set(expected) != set(actual):
    raise SystemExit("robot wheelhouse file list does not match manifest")
for name, item in expected.items():
    path = actual[name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
        raise SystemExit("robot wheelhouse digest or size mismatch: " + name)
PY
python3 -m pip install --user --no-index --find-links "$wheelhouse" \
  "$wheelhouse"/*.whl
python3 -c 'import yaml, zstandard, mcap, psutil, websocket'
printf 'GO2_MAPPING_PYTHON_RUNTIME_READY wheelhouse=%s manifest=%s\n' "$wheelhouse" "$manifest"
