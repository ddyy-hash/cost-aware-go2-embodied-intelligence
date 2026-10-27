from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(package_prefix, files, ros_distro):
    records = []
    for path in sorted((Path(item) for item in files), key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return {"schema_version": 1, "package_prefix": str(package_prefix), "ros_distro": ros_distro, "files": records}


def verify_manifest(expected, current):
    problems = []
    for field in ("package_prefix", "ros_distro"):
        if expected.get(field) != current.get(field):
            problems.append("{} mismatch".format(field))
    expected_files = {item["path"]: item for item in expected.get("files", [])}
    current_files = {item["path"]: item for item in current.get("files", [])}
    for path in sorted(set(expected_files) | set(current_files)):
        if path not in expected_files:
            problems.append("unexpected file {}".format(path))
        elif path not in current_files:
            problems.append("missing file {}".format(path))
        elif expected_files[path]["sha256"] != current_files[path]["sha256"]:
            problems.append("hash mismatch {}".format(path))
    return problems


def write_manifest(path, manifest):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(target))
