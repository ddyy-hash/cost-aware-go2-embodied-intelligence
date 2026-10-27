#!/usr/bin/env python3
"""Publish one already-verified provisional RGB-D sidecar under the official Dashboard static route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_dashboard.provisional_surface_preview import (
    publish_current_provisional_surface_preview,
)


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--preview-manifest-path", required=True, type=Path)
    parser.add_argument(
        "--static-root",
        type=Path,
        default=ROOT / "mapping" / "pc" / "go2_mapping_dashboard" / "static",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = publish_current_provisional_surface_preview(
            status_path=args.status_path,
            output_root=args.output_root,
            preview_manifest_path=args.preview_manifest_path,
            static_root=args.static_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "state": "PROVISIONAL_SURFACE_PREVIEW_PUBLICATION_ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "delivery_authorized": False,
                    "measurement_authority": False,
                    "final_store_eligible": False,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
