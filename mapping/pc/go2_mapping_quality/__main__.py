"""Create a machine-readable quality decision from calibration evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .acceptance import DEFAULT_CONFIG_PATH, QualityLevel, assess_quality, meets_requirement


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess Go2 mapping calibration and session evidence")
    parser.add_argument("--calibration", type=Path, help="Measured calibration manifest JSON")
    parser.add_argument("--session-evidence", type=Path, help="Raw capture and validation evidence JSON")
    parser.add_argument("--quality-gates", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--require", choices=[item.value for item in QualityLevel], default=QualityLevel.PREVIEW_UNCALIBRATED.value)
    parser.add_argument("--output", type=Path, help="Write the decision JSON atomically to this path")
    args = parser.parse_args()

    assessment = assess_quality(args.calibration, args.session_evidence, args.quality_gates)
    report = assessment.snapshot()
    report["required_level"] = args.require
    report["passed_required_level"] = meets_requirement(assessment, args.require)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        sys.stdout.write(rendered)
    return 0 if report["passed_required_level"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
