"""Run the non-mutating Go2 SportClient capability probe."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("/home/unitree/go2_mapping/runtime/bin/go2_vendor_sport_probe"))
    parser.add_argument("--network", default="eth0")
    args = parser.parse_args()
    if not args.binary.is_file():
        print("GO2_VENDOR_SPORT_PROBE_BLOCKED reason=binary_missing", flush=True)
        return 2
    result = subprocess.run([str(args.binary), args.network], text=True, capture_output=True, timeout=8, check=False)
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output, flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
