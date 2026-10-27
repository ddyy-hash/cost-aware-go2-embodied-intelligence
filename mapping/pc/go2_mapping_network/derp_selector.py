"""Select a private DERP only from measurements made at both endpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Optional


_LATENCY = re.compile(r"^\s*-\s+([a-zA-Z0-9_-]+):\s+([0-9]+(?:\.[0-9]+)?)ms(?:\s|$)")


@dataclass(frozen=True)
class DerpDecision:
    preferred_region_id: int
    preferred_region_code: Optional[str]
    private_score_ms: Optional[float]
    best_official_score_ms: Optional[float]
    reason: str


def parse_netcheck_latencies(report: str) -> dict[str, float]:
    latencies: dict[str, float] = {}
    for line in report.splitlines():
        match = _LATENCY.match(line)
        if match:
            latencies[match.group(1)] = float(match.group(2))
    return latencies


def select_derp_region(
    *,
    pc_latencies: Mapping[str, float],
    robot_latencies: Mapping[str, float],
    private_regions: Mapping[str, int],
    peer_is_direct: bool,
    improvement_ratio: float = 0.10,
) -> DerpDecision:
    if not 0.0 <= improvement_ratio < 1.0:
        raise ValueError("improvement_ratio must be in [0, 1)")
    if peer_is_direct:
        return DerpDecision(0, None, None, None, "p2p_direct_preferred")

    private_scores = {
        code: float(pc_latencies[code]) + float(robot_latencies[code])
        for code in private_regions
        if code in pc_latencies and code in robot_latencies
    }
    official_scores = {
        code: float(pc_latencies[code]) + float(robot_latencies[code])
        for code in pc_latencies.keys() & robot_latencies.keys()
        if code not in private_regions
    }
    best_official = min(official_scores.values()) if official_scores else None
    if not private_scores:
        return DerpDecision(0, None, None, best_official, "private_region_not_reachable_from_both_ends")

    private_code = min(private_scores, key=private_scores.__getitem__)
    private_score = private_scores[private_code]
    if best_official is None or private_score <= best_official * (1.0 - improvement_ratio):
        return DerpDecision(
            int(private_regions[private_code]),
            private_code,
            private_score,
            best_official,
            "private_region_materially_faster",
        )
    return DerpDecision(0, None, private_score, best_official, "private_gain_below_threshold")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-netcheck", type=Path, required=True)
    parser.add_argument("--robot-netcheck", type=Path, required=True)
    parser.add_argument("--private-region", action="append", required=True, metavar="CODE=ID")
    parser.add_argument("--peer-direct", action="store_true")
    parser.add_argument("--minimum-improvement-percent", type=float, default=10.0)
    args = parser.parse_args()
    private_regions = {}
    for item in args.private_region:
        code, separator, region_id = item.partition("=")
        if not separator or not code or not region_id.isdigit():
            parser.error("--private-region must be CODE=ID")
        private_regions[code] = int(region_id)
    decision = select_derp_region(
        pc_latencies=parse_netcheck_latencies(args.pc_netcheck.read_text(encoding="utf-8")),
        robot_latencies=parse_netcheck_latencies(args.robot_netcheck.read_text(encoding="utf-8")),
        private_regions=private_regions,
        peer_is_direct=args.peer_direct,
        improvement_ratio=args.minimum_improvement_percent / 100.0,
    )
    print(json.dumps(asdict(decision), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
