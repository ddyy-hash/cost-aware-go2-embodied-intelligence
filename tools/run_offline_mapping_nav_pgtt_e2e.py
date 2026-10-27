"""Run the offline mapping -> Nav2 bridge -> PGTT acceptance chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


def build_e2e_report(
    *,
    mapping: Mapping[str, Any],
    nav2: Mapping[str, Any],
    vla: Mapping[str, Any],
    locomotion: Mapping[str, Any],
    robustness: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine layer results without promoting the offline chain to hardware."""
    layers = {
        "mapping": dict(mapping),
        "nav2": dict(nav2),
        "vla": dict(vla),
        "locomotion": dict(locomotion),
        "robustness": dict(robustness),
        "safety": dict(safety),
    }
    return {
        "schema_version": 1,
        "pipeline": "coarse_map_to_nav2_bridge_to_pgtt_mujoco",
        "simulation_backend": "MuJoCo_CPU",
        "layers": layers,
        "accepted": all(bool(layer.get("accepted")) for layer in layers.values()),
        "ros_nav2_runtime": False,
        "executable": False,
        "transport_authorized": False,
    }


class _SimulatedNav2Client:
    """Minimal Nav2 action contract used only for offline bridge lifecycle tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.cancelled = False

    def wait_for_server(self, timeout_sec: float) -> bool:
        return timeout_sec > 0

    def send_goal(self, goal: Mapping[str, Any]) -> bool:
        if not goal.get("waypoints"):
            return False
        self.sent.append(dict(goal))
        return True

    def poll_result(self) -> str:
        return "SUCCEEDED"

    def cancel_goal(self) -> None:
        self.cancelled = True


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def run(
    *,
    map_path: Path,
    output_dir: Path,
    locomotion_episodes: int | None = None,
    locomotion_steps: int = 1000,
    robustness_episodes: int = 5,
    reuse_physics: bool = False,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from mapping.pc.go2_mapping_dashboard.coarse_map import build_task_graph, validate_coarse_map
    from mapping.pc.go2_mapping_dashboard.mission import compile_mission, update_mission_task
    from mapping.pc.go2_mapping_dashboard.nav2_map import write_nav2_map
    from mapping.pc.go2_mapping_locomotion.supervisor import LocomotionSupervisor, MotionLease, TerrainObservation
    from mapping.pc.go2_mapping_locomotion.vla_bridge import MobileVLATerrainCoordinator
    from mapping.pc.go2_mapping_vla.offline import parse_mobilevla_decision
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import NavigationBridgeStore
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.nav2_map_preflight import validate_nav2_map
    from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_executor import NavigationExecutionAdapter
    from tools.run_locomotion_safety_contract_matrix import run as run_safety
    from tools.run_pgtt_mujoco_acceptance_matrix import run_matrix as run_locomotion
    from tools.run_pgtt_mujoco_robustness_matrix import run_matrix as run_robustness

    output_dir.mkdir(parents=True, exist_ok=True)
    coarse = validate_coarse_map(_read_json(map_path))
    nav_root = output_dir / "nav2-map"
    exported = write_nav2_map(coarse, nav_root)
    preflight = validate_nav2_map(nav_root / "map.yaml")
    mapping_report = {
        "accepted": True,
        "map_id": preflight["map_id"],
        "coordinate_frame": preflight["coordinate_frame"],
        "source_revision": preflight["source_revision"],
        "width": preflight["width"],
        "height": preflight["height"],
        "resolution": preflight["resolution"],
        "nav2_yaml": str(exported["yaml_path"]),
    }

    mission = compile_mission(build_task_graph(coarse), mission_id="offline-e2e-mission")
    task = next(item for item in mission["tasks"] if item["kind"] == "floor_coverage")
    mission = update_mission_task(
        mission,
        task["task_id"],
        lease_id="offline-e2e-lease",
        mode="AUTONOMOUS",
        expected_revision=task["revision"],
    )
    task = next(item for item in mission["tasks"] if item["task_id"] == task["task_id"])
    (output_dir / "mission.json").write_text(json.dumps(mission, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    store = NavigationBridgeStore(output_dir / "navigation")
    store.compile_from_mission(mission, task)
    client = _SimulatedNav2Client()
    executor = NavigationExecutionAdapter(store=store, client_factory=lambda _plan: client)
    active = executor.execute_once()
    done = executor.execute_once()
    nav_report = {
        "accepted": done["state"] == "DONE" and bool(client.sent),
        "goal_state": done["state"],
        "active_state": active["state"],
        "waypoint_count": len(client.sent[0]["waypoints"]) if client.sent else 0,
        "mission_id": mission["mission_id"],
        "task_id": task["task_id"],
    }
    decision = parse_mobilevla_decision(
        {
            "intent": "inspect_room",
            "target": "room-1",
            "confidence": 0.95,
            "reasoning_summary": "offline e2e semantic replay",
            "constraints": {},
        }
    )
    terrain = TerrainObservation("STAIR_UP", 0.95, 0.12, 0.30, 0.90, 22.0, 6, True, True)
    supervisor = LocomotionSupervisor(approved_controllers={"pgtt-go2"})
    supervisor.confirm_operator()
    supervisor.set_lease(MotionLease("offline-e2e-lease", "offline-e2e"))
    supervisor.select_controller("pgtt-go2")
    vla_decision = MobileVLATerrainCoordinator(supervisor).request(
        decision,
        terrain=terrain,
        terrain_task="STAIR_ASCEND",
    ).supervisor_decision
    vla_report = {
        "accepted": vla_decision.state == "READY",
        "state": vla_decision.state,
        "controller": (vla_decision.command or {}).get("controller"),
        "intent": decision.intent,
    }

    locomotion_path = output_dir / "pgtt-mujoco-acceptance-matrix.json"
    robustness_path = output_dir / "pgtt-mujoco-robustness-matrix.json"
    safety_path = output_dir / "locomotion-safety-contract-matrix.json"
    if reuse_physics and locomotion_path.is_file() and robustness_path.is_file() and safety_path.is_file():
        locomotion = _read_json(locomotion_path)
        robustness = _read_json(robustness_path)
        safety = _read_json(safety_path)
    else:
        locomotion = run_locomotion(episodes=locomotion_episodes, steps=locomotion_steps, output=locomotion_path)
        robustness = run_robustness(episodes=robustness_episodes, steps=locomotion_steps, output=robustness_path)
        safety = run_safety(safety_path)
    report = build_e2e_report(
        mapping=mapping_report,
        nav2=nav_report,
        vla=vla_report,
        locomotion={"accepted": bool(locomotion["accepted"]), "scenarios": locomotion["scenarios"]},
        robustness={"accepted": bool(robustness["accepted"]), "scenarios": robustness["scenarios"]},
        safety={"accepted": bool(safety["accepted"]), "checks": safety["checks"]},
    )
    report_path = output_dir / "offline-mapping-nav-pgtt-e2e.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, default=ROOT / "mapping-data" / "pc-stack" / "coarse-map.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "offline-mapping-nav-pgtt-e2e")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--robustness-episodes", type=int, default=5)
    parser.add_argument("--reuse-physics", action="store_true", help="reuse existing accepted physics artifacts")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.steps <= 0 or args.robustness_episodes <= 0:
        parser.error("steps and robustness-episodes must be positive")
    report = run(
        map_path=args.map_path,
        output_dir=args.output_dir,
        locomotion_episodes=args.episodes,
        locomotion_steps=args.steps,
        robustness_episodes=args.robustness_episodes,
        reuse_physics=args.reuse_physics,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    print("OFFLINE_MAPPING_NAV_PGTT_E2E_PASS" if report["accepted"] else "OFFLINE_MAPPING_NAV_PGTT_E2E_FAIL")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
