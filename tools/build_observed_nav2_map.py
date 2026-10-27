#!/usr/bin/env python3
"""Build a versioned Nav2 candidate from a coarse map and observed XYZ points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_dashboard.nav2_map import _write_atomic
from mapping.pc.go2_mapping_dashboard.observed_nav2_map import build_observed_nav2_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-map", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True, help=".npy or .npz containing an Nx3 float array")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=1.0)
    args = parser.parse_args()
    coarse = json.loads(args.coarse_map.read_text(encoding="utf-8"))
    loaded = np.load(args.points)
    points = loaded["points"] if isinstance(loaded, np.lib.npyio.NpzFile) and "points" in loaded else loaded
    result = build_observed_nav2_map(coarse, points, resolution=args.resolution, padding=args.padding)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pgm = f"P5\n{result['width']} {result['height']}\n255\n".encode("ascii") + b"".join(bytes(row) for row in result["grid"])
    metadata = result["metadata"]
    yaml = (f"image: map.pgm\nresolution: {metadata['resolution']:.9g}\norigin: [{metadata['origin'][0]:.9g}, {metadata['origin'][1]:.9g}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n").encode("ascii")
    _write_atomic(output / "map.pgm", pgm)
    _write_atomic(output / "map.yaml", yaml)
    _write_atomic(output / "map.provenance.json", (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"output_dir": str(output), **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
