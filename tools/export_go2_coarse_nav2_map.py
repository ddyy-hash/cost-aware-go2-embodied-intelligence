#!/usr/bin/env python3
"""Export a v2 USER_PRIOR coarse map as a Nav2 PGM/YAML map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_dashboard.nav2_map import write_nav2_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, default=ROOT / "mapping-data" / "pc-stack" / "coarse-map.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=1.0)
    parser.add_argument("--wall-thickness", type=float, default=0.12)
    parser.add_argument("--max-cells", type=int, default=4_000_000)
    args = parser.parse_args()
    payload = json.loads(args.map_path.read_text(encoding="utf-8"))
    result = write_nav2_map(
        payload,
        args.output_dir,
        resolution=args.resolution,
        padding=args.padding,
        wall_thickness=args.wall_thickness,
        max_cells=args.max_cells,
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
