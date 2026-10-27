#!/usr/bin/env python3
"""Replay a MobileVLA decision through the non-actuating terrain supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.pc.go2_mapping_locomotion.supervisor import (  # noqa: E402
    LocomotionSupervisor,
    MotionLease,
    TerrainObservation,
)
from mapping.pc.go2_mapping_locomotion.vla_bridge import (  # noqa: E402
    MobileVLATerrainCoordinator,
)
from mapping.pc.go2_mapping_vla.offline import parse_mobilevla_decision  # noqa: E402


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON input is unreadable: {path}") from exc


def _terrain(payload: dict[str, Any]) -> TerrainObservation:
    required = (
        "state",
        "confidence",
        "riser_height_m",
        "tread_depth_m",
        "width_m",
        "pitch_deg",
        "step_count",
        "top_landing_supported",
        "bottom_landing_supported",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("terrain evidence is missing: " + ", ".join(missing))
    return TerrainObservation(**{key: payload[key] for key in required})


def replay(
    decision_payload: str | dict[str, Any],
    terrain_payload: dict[str, Any],
    *,
    terrain_task: str,
    operator_confirmed: bool = False,
    lease_id: str | None = None,
    controller_id: str = "pgtt-go2",
) -> dict[str, Any]:
    decision = parse_mobilevla_decision(decision_payload)
    supervisor = LocomotionSupervisor(approved_controllers={controller_id})
    if operator_confirmed:
        supervisor.confirm_operator()
    if lease_id:
        supervisor.set_lease(MotionLease(lease_id, "offline-replay"))
    supervisor.select_controller(controller_id)
    result = MobileVLATerrainCoordinator(supervisor).request(
        decision,
        terrain=_terrain(terrain_payload),
        terrain_task=terrain_task,
    )
    return {
        "schema_version": 1,
        "model_intent": result.model_intent,
        "terrain_task": result.terrain_task,
        "supervisor": {
            "state": result.supervisor_decision.state,
            "command": result.supervisor_decision.command,
            "blocking_reasons": list(result.supervisor_decision.blocking_reasons),
        },
        "executable": False,
        "transport_authorized": False,
        "navigation_activation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--terrain-json", type=Path, required=True)
    parser.add_argument("--terrain-task", required=True)
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--lease-id")
    parser.add_argument("--controller-id", default="pgtt-go2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision_payload = _read_json(args.decision_json)
    terrain_payload = _read_json(args.terrain_json)
    if not isinstance(terrain_payload, dict):
        raise ValueError("terrain JSON must be an object")
    result = replay(
        decision_payload,
        terrain_payload,
        terrain_task=args.terrain_task,
        operator_confirmed=args.operator_confirmed,
        lease_id=args.lease_id,
        controller_id=args.controller_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print("MOBILEVLA_TERRAIN_REPLAY " + json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
