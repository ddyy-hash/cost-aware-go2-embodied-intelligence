#!/usr/bin/env python3
"""Atomically publish one verified local ``odom`` observed-surface pointer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_dashboard.derived_surface import (  # noqa: E402
    publish_session_odom_local_preview,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an explicitly local, read-only observed LIO odom surface."
    )
    parser.add_argument("--surface-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--status-path", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pointer = publish_session_odom_local_preview(
            args.surface_json,
            output_root=args.output_root,
            status_path=args.status_path,
        )
    except (OSError, ValueError) as exc:
        print(f"publish_go2_session_odom_local_surface: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(pointer, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
