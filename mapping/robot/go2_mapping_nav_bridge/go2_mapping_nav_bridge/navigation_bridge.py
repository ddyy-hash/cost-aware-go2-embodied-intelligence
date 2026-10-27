"""Compile mission tasks into navigation goals for the Nav2 runtime."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping


_MISSION_ID_RE = "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_TASK_ID_RE = "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_LEASE_ID_RE = "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_GOAL_ID_RE = "^(?:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._:-]{0,127})$"
_ALLOWED_MODES = {"MANUAL", "ASSISTED", "AUTONOMOUS"}
_TERMINAL_STATES = {"DONE", "CANCELLED", "QUARANTINED"}
_ACTIVE_STATES = {"NAV_READY", "NAV_ACTIVE", "NAV_PAUSED", "NAV_BLOCKED", "MANUAL_OVERRIDE", "SAFETY_STOP", "LOCALIZATION_REQUIRED"}


class NavigationBridgeError(RuntimeError):
    pass


def _regex(pattern: str):
    import re

    return re.compile(pattern, re.ASCII)


_MISSION_ID = _regex(_MISSION_ID_RE)
_TASK_ID = _regex(_TASK_ID_RE)
_LEASE_ID = _regex(_LEASE_ID_RE)
_GOAL_ID = _regex(_GOAL_ID_RE)


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(descriptor, "wb", closefd=True) as target:
            target.write(_canonical_bytes(payload))
            target.flush()
            import os

            os.fsync(target.fileno())
        Path(temp_name).replace(path)
        import os

        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        body = path.read_bytes()
        payload = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NavigationBridgeError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict) or body != _canonical_bytes(payload):
        raise NavigationBridgeError(f"{label} is not canonical")
    return payload


def _identifier(value: Any, pattern, label: str) -> str:
    result = str(value)
    if pattern.fullmatch(result) is None:
        raise NavigationBridgeError(f"{label} is invalid")
    return result


def _ensure_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NavigationBridgeError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise NavigationBridgeError(f"{label} must be finite")
    return number


def _point(value: Any, label: str) -> dict[str, float]:
    if isinstance(value, Mapping):
        x = _ensure_number(value.get("x"), f"{label}.x")
        y = _ensure_number(value.get("y"), f"{label}.y")
        yaw = _ensure_number(value.get("yaw", 0.0), f"{label}.yaw")
        frame = str(value.get("frame") or "SESSION_ODOM_LOCAL")
        return {"x": x, "y": y, "yaw": yaw, "frame": frame}
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise NavigationBridgeError(f"{label} must be a point")
    x = _ensure_number(value[0], f"{label}.x")
    y = _ensure_number(value[1], f"{label}.y")
    yaw = _ensure_number(value[2], f"{label}.yaw") if len(value) > 2 else 0.0
    frame = str(value[3]) if len(value) > 3 else "SESSION_ODOM_LOCAL"
    return {"x": x, "y": y, "yaw": yaw, "frame": frame}


def _normalize_points(values: Any, label: str) -> list[dict[str, float]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise NavigationBridgeError(f"{label} must be a list")
    return [_point(item, f"{label}[{index}]") for index, item in enumerate(values)]


def _normalise_mode(value: Any) -> str:
    mode = str(value or "ASSISTED")
    if mode not in _ALLOWED_MODES:
        raise NavigationBridgeError("mode is unsupported")
    return mode


def _goal_id(mission_id: str, task_id: str, task_revision: int, source_sha: str | None) -> str:
    canonical = json.dumps(
        {
            "mission_id": mission_id,
            "task_id": task_id,
            "task_revision": task_revision,
            "source_coarse_map_sha256": source_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _speed_for_task(task_kind: str, mode: str) -> float:
    base = {
        "perimeter": 0.16,
        "corners": 0.12,
        "feature": 0.10,
        "floor_coverage": 0.18,
    }.get(task_kind, 0.12)
    if mode == "MANUAL":
        return 0.0
    if mode == "ASSISTED":
        return round(base, 3)
    return round(min(0.22, base * 1.1), 3)


def _task_state(task_status: str, mode: str, lease_id: str | None) -> tuple[str, str | None]:
    if task_status in _TERMINAL_STATES:
        if task_status == "DONE":
            return "DONE", "TASK_DONE"
        return "NAV_BLOCKED", f"TASK_{task_status}"
    if task_status == "ACTIVE":
        if mode == "MANUAL":
            return "MANUAL_OVERRIDE", "MANUAL_TASK"
        if mode == "AUTONOMOUS" and not lease_id:
            return "NAV_BLOCKED", "LEASE_REQUIRED"
        return "NAV_ACTIVE", None
    if task_status == "PAUSED":
        return "NAV_PAUSED", None
    if mode == "MANUAL":
        return "MANUAL_OVERRIDE", "MANUAL_TASK"
    if mode == "AUTONOMOUS" and not lease_id:
        return "NAV_BLOCKED", "LEASE_REQUIRED"
    return "NAV_READY", "OPERATOR_CONFIRMATION_REQUIRED" if mode == "ASSISTED" else None


def _mission_task_record(mission: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    mission_id = _identifier(mission.get("mission_id"), _MISSION_ID, "mission_id")
    task_id = _identifier(task.get("task_id"), _TASK_ID, "task_id")
    task_revision = int(task.get("revision", 1))
    if task_revision <= 0:
        raise NavigationBridgeError("task_revision must be positive")
    mode = str(task.get("mode") or "ASSISTED")
    if mode not in _ALLOWED_MODES:
        raise NavigationBridgeError("task mode is unsupported")
    task_status = str(task.get("status") or "READY")
    lease_id = task.get("lease_id")
    if lease_id is not None:
        lease_id = _identifier(lease_id, _LEASE_ID, "lease_id")
    room_polygon = task.get("room_polygon") or []
    waypoints = _normalize_points(task.get("waypoints"), "waypoints")
    inspection_targets = _normalize_points(task.get("inspection_targets"), "inspection_targets")
    geofence = _normalize_points(room_polygon, "room_polygon")
    state, reason = _task_state(task_status, mode, lease_id)
    return {
        "schema_version": 1,
        "mission_id": mission_id,
        "task_id": task_id,
        "task_kind": task.get("kind"),
        "goal_id": _goal_id(mission_id, task_id, task_revision, str(mission.get("source_coarse_map_sha256") or "")),
        "goal_authority": mode,
        "mode": mode,
        "state": state,
        "state_reason": reason,
        "task_status": task_status,
        "task_revision": task_revision,
        "lease_id": lease_id,
        "lease_state": "LEASED" if lease_id else "UNLEASED",
        "waypoints": waypoints,
        "inspection_targets": inspection_targets,
        "geofence": {
            "frame": mission.get("coordinate_frame") or "SESSION_ODOM_LOCAL",
            "polygon": deepcopy(_normalize_points(task.get("room_polygon") or geofence, "room_polygon")),
        },
        "max_speed_mps": _speed_for_task(str(task.get("kind") or ""), mode),
        "max_duration_s": int(max(30, int(task.get("estimated_minutes", 1)) * 60)),
        "safety_revision": mission.get("source_coarse_map_sha256"),
        "mission_source_sha256": mission.get("source_coarse_map_sha256"),
        "goal_available": state in {"NAV_READY", "NAV_ACTIVE", "NAV_PAUSED"},
        "operator_confirmation_required": mode == "ASSISTED",
        "bridge_revision": 1,
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }


def compile_navigation_goal(
    mission: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    goal_id: str | None = None,
    lease_id: str | None = None,
    max_speed_mps: float | None = None,
    max_duration_s: int | None = None,
) -> dict[str, Any]:
    record = _mission_task_record(mission, task)
    if goal_id is not None:
        record["goal_id"] = _identifier(goal_id, _GOAL_ID, "goal_id")
    if lease_id is not None:
        record["lease_id"] = _identifier(lease_id, _LEASE_ID, "lease_id")
        record["lease_state"] = "LEASED"
    if max_speed_mps is not None:
        record["max_speed_mps"] = round(_ensure_number(max_speed_mps, "max_speed_mps"), 3)
    if max_duration_s is not None:
        if isinstance(max_duration_s, bool) or not isinstance(max_duration_s, int) or max_duration_s <= 0:
            raise NavigationBridgeError("max_duration_s must be a positive integer")
        record["max_duration_s"] = int(max_duration_s)
    if record["mode"] == "AUTONOMOUS" and not record.get("lease_id"):
        record["state"] = "NAV_BLOCKED"
        record["state_reason"] = "LEASE_REQUIRED"
        record["goal_available"] = False
    return record


def compile_explicit_navigation_goal(payload: Mapping[str, Any]) -> dict[str, Any]:
    mission_id = payload.get("mission_id")
    task_id = payload.get("task_id")
    if mission_id is not None:
        mission_id = _identifier(mission_id, _MISSION_ID, "mission_id")
    if task_id is not None:
        task_id = _identifier(task_id, _TASK_ID, "task_id")
    mode = _normalise_mode(payload.get("mode", "ASSISTED"))
    task_revision = payload.get("task_revision", 1)
    if isinstance(task_revision, bool) or not isinstance(task_revision, int) or task_revision <= 0:
        raise NavigationBridgeError("task_revision must be a positive integer")
    goal_id = payload.get("goal_id")
    if goal_id is not None:
        goal_id = _identifier(goal_id, _GOAL_ID, "goal_id")
    waypoints = _normalize_points(payload.get("waypoints"), "waypoints")
    inspection_targets = _normalize_points(payload.get("inspection_targets"), "inspection_targets")
    geofence_payload = payload.get("geofence") or {}
    if not isinstance(geofence_payload, Mapping):
        raise NavigationBridgeError("geofence must be an object")
    geofence = {
        "frame": str(geofence_payload.get("frame") or "SESSION_ODOM_LOCAL"),
        "polygon": _normalize_points(geofence_payload.get("polygon"), "geofence.polygon"),
    }
    lease_id = payload.get("lease_id")
    if lease_id is not None:
        lease_id = _identifier(lease_id, _LEASE_ID, "lease_id")
    task_status = str(payload.get("task_status") or "READY")
    state, reason = _task_state(task_status, mode, lease_id)
    if not waypoints and not inspection_targets:
        raise NavigationBridgeError("navigation goal requires waypoints or inspection_targets")
    goal_id = goal_id or _goal_id(mission_id or "explicit-mission", task_id or "explicit-task", task_revision, str(payload.get("safety_revision") or ""))
    max_speed = payload.get("max_speed_mps")
    max_duration = payload.get("max_duration_s")
    record = {
        "schema_version": 1,
        "mission_id": mission_id,
        "task_id": task_id,
        "task_kind": payload.get("task_kind"),
        "goal_id": goal_id,
        "goal_authority": mode,
        "mode": mode,
        "state": state,
        "state_reason": reason,
        "task_status": task_status,
        "task_revision": task_revision,
        "lease_id": lease_id,
        "lease_state": "LEASED" if lease_id else "UNLEASED",
        "waypoints": waypoints,
        "inspection_targets": inspection_targets,
        "geofence": geofence,
        "max_speed_mps": round(_ensure_number(max_speed, "max_speed_mps"), 3) if max_speed is not None else _speed_for_task(str(payload.get("task_kind") or ""), mode),
        "max_duration_s": int(max_duration) if max_duration is not None else max(30, int(payload.get("estimated_minutes", 1)) * 60),
        "safety_revision": payload.get("safety_revision"),
        "mission_source_sha256": payload.get("mission_source_sha256"),
        "goal_available": state in {"NAV_READY", "NAV_ACTIVE", "NAV_PAUSED"},
        "operator_confirmation_required": mode == "ASSISTED",
        "bridge_revision": 1,
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }
    if record["mode"] == "AUTONOMOUS" and not record.get("lease_id"):
        record["state"] = "NAV_BLOCKED"
        record["state_reason"] = "LEASE_REQUIRED"
        record["goal_available"] = False
    return record


class NavigationBridgeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "navigation-current.json"
        self._lock = threading.RLock()

    def _default(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "state": "NAV_READY",
            "state_reason": "NO_GOAL",
            "goal_available": False,
            "goal_id": None,
            "mission_id": None,
            "task_id": None,
            "task_kind": None,
            "task_revision": None,
            "task_status": None,
            "mode": None,
            "lease_id": None,
            "lease_state": "UNLEASED",
            "waypoints": [],
            "inspection_targets": [],
            "geofence": {"frame": "SESSION_ODOM_LOCAL", "polygon": []},
            "max_speed_mps": 0.0,
            "max_duration_s": 0,
            "goal_authority": None,
            "operator_confirmation_required": False,
            "safety_revision": None,
            "mission_source_sha256": None,
            "action_feedback_count": 0,
            "action_feedback_updated_at_unix": None,
            "bridge_revision": 0,
            "created_at_unix": None,
            "updated_at_unix": None,
        }

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                return self._default()
            payload = _read_json(self.path, "navigation bridge")
            if payload.get("schema_version") != 1:
                raise NavigationBridgeError("navigation bridge schema version is unsupported")
            return payload

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = deepcopy(dict(payload))
        record["schema_version"] = 1
        record["updated_at_unix"] = time.time()
        if not record.get("created_at_unix"):
            record["created_at_unix"] = record["updated_at_unix"]
        with self._lock:
            _write_atomic(self.path, record)
            return deepcopy(record)

    def status(self) -> dict[str, Any]:
        return self.load()

    def compile_from_mission(
        self,
        mission: Mapping[str, Any],
        task: Mapping[str, Any],
        *,
        goal_id: str | None = None,
        lease_id: str | None = None,
        max_speed_mps: float | None = None,
        max_duration_s: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = compile_navigation_goal(
                mission,
                task,
                goal_id=goal_id,
                lease_id=lease_id,
                max_speed_mps=max_speed_mps,
                max_duration_s=max_duration_s,
            )
            record["bridge_revision"] = 1
            return self.save(record)

    def compile_from_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = compile_explicit_navigation_goal(payload)
            record["bridge_revision"] = 1
            return self.save(record)

    def _mutate(self, mutator) -> dict[str, Any]:
        with self._lock:
            current = self.load()
            updated = mutator(deepcopy(current))
            updated["bridge_revision"] = int(updated.get("bridge_revision", 0)) + 1
            return self.save(updated)

    def update_lease(self, lease_id: str | None) -> dict[str, Any]:
        normalized = None if lease_id in (None, "") else _identifier(lease_id, _LEASE_ID, "lease_id")

        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            current["lease_id"] = normalized
            current["lease_state"] = "LEASED" if normalized else "UNLEASED"
            if current["mode"] == "AUTONOMOUS":
                if normalized:
                    if current["state"] == "NAV_BLOCKED" and current.get("state_reason") == "LEASE_REQUIRED":
                        current["state"] = "NAV_READY"
                        current["state_reason"] = "OPERATOR_CONFIRMATION_REQUIRED"
                        current["goal_available"] = True
                else:
                    if current["state"] == "NAV_ACTIVE":
                        raise NavigationBridgeError("active autonomous goal cannot release its lease")
                    current["state"] = "NAV_BLOCKED"
                    current["state_reason"] = "LEASE_REQUIRED"
                    current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def activate(self) -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            if current["state"] == "NAV_ACTIVE":
                return current
            if current["state"] not in {"NAV_READY", "MANUAL_OVERRIDE"}:
                raise NavigationBridgeError("navigation goal is not ready")
            if current["mode"] == "MANUAL":
                raise NavigationBridgeError("manual tasks do not become autonomous navigation goals")
            if current["mode"] == "AUTONOMOUS" and not current.get("lease_id"):
                raise NavigationBridgeError("AUTONOMOUS mode requires a lease_id")
            current["state"] = "NAV_ACTIVE"
            current["state_reason"] = None
            current["goal_available"] = True
            return current

        return self._mutate(mutate)

    def pause(self) -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            if current["state"] != "NAV_ACTIVE":
                raise NavigationBridgeError("navigation goal is not active")
            current["state"] = "NAV_PAUSED"
            current["state_reason"] = None
            return current

        return self._mutate(mutate)

    def resume(self) -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            if current["state"] != "NAV_PAUSED":
                raise NavigationBridgeError("navigation goal is not paused")
            current["state"] = "NAV_ACTIVE"
            current["state_reason"] = None
            return current

        return self._mutate(mutate)

    def cancel(self) -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            if current["state"] in {"DONE", "CANCELLED"}:
                return current
            current["state"] = "CANCELLED"
            current["state_reason"] = "OPERATOR_CANCELLED"
            current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def done(self, *, reason: str = "NAV2_SUCCEEDED") -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            if current["state"] in {"DONE", "CANCELLED"}:
                return current
            current["state"] = "DONE"
            current["state_reason"] = reason
            current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def record_feedback(self, count: int) -> dict[str, Any]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise NavigationBridgeError("action feedback count must be a non-negative integer")

        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            previous = int(current.get("action_feedback_count", 0))
            if count < previous:
                raise NavigationBridgeError("action feedback count cannot move backwards")
            if current["state"] == "NAV_ACTIVE" and count > previous:
                current["action_feedback_count"] = count
                current["action_feedback_updated_at_unix"] = time.time()
            return current

        return self._mutate(mutate)

    def block(self, *, reason: str = "NAV2_UNAVAILABLE") -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            current["state"] = "NAV_BLOCKED"
            current["state_reason"] = reason
            current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def manual_override(self, *, reason: str = "MANUAL_TASK") -> dict[str, Any]:
        """Record that operator-controlled mode owns the robot for this goal."""

        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            current["state"] = "MANUAL_OVERRIDE"
            current["state_reason"] = reason
            current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def safety_stop(self, *, reason: str = "SAFETY_STOP") -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            current["state"] = "SAFETY_STOP"
            current["state_reason"] = reason
            current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def acknowledge_safety_stop(self) -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            if current["state"] != "SAFETY_STOP":
                raise NavigationBridgeError("navigation goal is not in safety stop")
            current["state"] = "MANUAL_OVERRIDE"
            current["state_reason"] = "SAFETY_ACKNOWLEDGED"
            current["goal_available"] = False
            return current

        return self._mutate(mutate)

    def localization_required(self) -> dict[str, Any]:
        def mutate(current: dict[str, Any]) -> dict[str, Any]:
            current["state"] = "LOCALIZATION_REQUIRED"
            current["state_reason"] = "LOCALIZATION_LOST"
            current["goal_available"] = False
            return current

        return self._mutate(mutate)
