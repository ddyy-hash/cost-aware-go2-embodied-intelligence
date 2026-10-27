"""Publish the verified final LiDAR surface for an older shared-static Dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_dashboard.static_surface_compatibility import publish_verified_surface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=ROOT / "mapping-data" / "pc-stack" / "automatic-reconstruction-current.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output" / "reconstruction" / "derived",
    )
    parser.add_argument(
        "--static-root",
        type=Path,
        default=ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            publish_verified_surface(
                status_path=args.status_path,
                output_root=args.output_root,
                static_root=args.static_root,
            ),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
