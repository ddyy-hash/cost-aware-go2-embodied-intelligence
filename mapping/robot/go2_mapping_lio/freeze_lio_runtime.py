#!/usr/bin/env python3
"""Capture or verify the installed LIO-SAM file fingerprint; never starts LIO."""
from __future__ import print_function

import argparse
import os
import subprocess
from pathlib import Path

from go2_mapping_lio.runtime_manifest import build_manifest, verify_manifest, write_manifest


def installed_files(prefix):
    root = Path(prefix)
    files = []
    for directory in (root / "lib", root / "share"):
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    dependencies = []
    for executable in [path for path in files if "/lib/" in str(path)]:
        try:
            output = subprocess.check_output(["ldd", str(executable)], stderr=subprocess.STDOUT).decode("utf-8", "replace")
        except subprocess.CalledProcessError:
            continue
        for line in output.splitlines():
            if "=>" in line:
                candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
                if candidate.startswith("/") and Path(candidate).is_file():
                    dependencies.append(Path(candidate))
    return sorted(set(files + dependencies), key=lambda item: str(item))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    prefix = subprocess.check_output(["ros2", "pkg", "prefix", "lio_sam_ros2"]).decode("utf-8").strip()
    current = build_manifest(prefix, installed_files(prefix), os.environ.get("ROS_DISTRO", "unknown"))
    if args.verify:
        import json
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors = verify_manifest(expected, current)
        if errors:
            raise SystemExit("LIO runtime manifest mismatch: " + "; ".join(errors))
    else:
        write_manifest(args.manifest, current)


if __name__ == "__main__":
    main()
