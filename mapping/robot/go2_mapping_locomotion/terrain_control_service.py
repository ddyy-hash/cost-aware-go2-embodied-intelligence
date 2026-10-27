"""High-level terrain-control facade for the authenticated Go2 service.

This layer owns the operator-facing terrain state and audit records. It does
not publish ``rt/lowcmd``; a compatible native executor must consume the
handoff only after its own SDK and lease gates pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from .request_gate import RequestGateError, validate_controller_request
from .vendor_sport_adapter import Go2VendorSportAdapter


CONFIRMATION = "GO2_PGTT_FIELD_TEST_CONFIRMED"
QUALIFIED = {"STAIR_UP", "STAIR_DOWN", "ROUGH_TERRAIN"}


class TerrainControlSupervisor:
    def __init__(self, *, status_path: Path, control_status_path: Path, motion_helper: Path,
                 vendor_adapter: Path | None = None, executor: Path | None = None) -> None:
        self.status_path = Path(status_path)
        self.control_status_path = Path(control_status_path)
        self.audit_path = self.control_status_path.with_name("terrain-control-audit.jsonl")
        self.request_path = self.control_status_path.with_name("terrain-control-request.json")
        self.motion_helper = Path(motion_helper)
        self.vendor_adapter = Go2VendorSportAdapter(
            vendor_adapter or Path("/home/unitree/go2_mapping/runtime/bin/go2_vendor_sport_adapter")
        )
        self.executor = Path(executor) if executor else Path("/home/unitree/go2_mapping/runtime/bin/go2_native_terrain_executor")
        self.executor_status = self.control_status_path.with_name("native-terrain-executor-status.json")
        self._executor_process: subprocess.Popen[str] | None = None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _router(self) -> dict[str, Any] | None:
        return self._read_json(self.status_path)

    def _write_atomic(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _emit(self, *, state: str, reason: str, transport_authorized: bool = False,
              motion_lease_available: bool = False) -> dict[str, Any]:
        previous = self._read_json(self.control_status_path) or {}
        sequence = int(previous.get("sequence", 0)) + 1
        route = self._router() or {}
        payload: dict[str, Any] = {
            "schema_version": 1,
            "sequence": sequence,
            "state": state,
            "reason": reason,
            "route_state": route.get("state"),
            "policy_level": route.get("policy_level"),
            "transport_authorized": bool(transport_authorized),
            "motion_helper_available": self.motion_helper.is_file() and os.access(self.motion_helper, os.X_OK),
            "vendor_sport_available": self.vendor_adapter.available,
            "motion_lease_available": bool(motion_lease_available),
            "updated_at_unix": time.time(),
        }
        self._write_atomic(self.control_status_path, payload)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            audit.flush()
            os.fsync(audit.fileno())
        return payload

    def status(self) -> dict[str, Any]:
        current = self._read_json(self.control_status_path)
        if current is not None:
            current = dict(current)
            current["motion_helper_available"] = self.motion_helper.is_file() and os.access(self.motion_helper, os.X_OK)
            current["vendor_sport_available"] = self.vendor_adapter.available
            current.setdefault("motion_lease_available", False)
            current["executor_running"] = bool(self._executor_process is not None and self._executor_process.poll() is None)
            current["motion_lease_available"] = current["executor_running"]
            if self.executor_status.is_file():
                executor_state = self._read_json(self.executor_status)
                if executor_state is not None:
                    current["executor_state"] = executor_state.get("state")
                    current["executor_blocker"] = executor_state.get("blocker")
                    current["transport_authorized"] = bool(executor_state.get("transport_authorized", False))
            return current
        if not self.vendor_adapter.available:
            return self._emit(state="ENHANCED_WALK_BLOCKED", reason="VENDOR_SPORT_INTERFACE_UNAVAILABLE")
        return self._emit(state="NATIVE_READY", reason="NOT_ARMED")

    def arm(self, confirmation: Any) -> dict[str, Any]:
        if confirmation != CONFIRMATION:
            return self._emit(state="NATIVE_READY", reason="FIELD_CONFIRMATION_REQUIRED")
        if not self.vendor_adapter.available:
            return self._emit(state="ENHANCED_WALK_BLOCKED", reason="VENDOR_SPORT_INTERFACE_UNAVAILABLE")
        route = self._router() or {}
        if route.get("state") not in QUALIFIED or route.get("executable") is not True:
            return self._emit(state="NATIVE_READY", reason="TERRAIN_ROUTE_NOT_EXECUTABLE")
        if not (self.executor.is_file() and os.access(self.executor, os.X_OK)):
            return self._emit(state="ENHANCED_WALK_BLOCKED", reason="EXECUTOR_UNAVAILABLE")
        if self._executor_process is not None and self._executor_process.poll() is None:
            return self._emit(state="ENHANCED_WALK", reason="FIELD_SESSION_ALREADY_ARMED", transport_authorized=True)
        self._executor_process = subprocess.Popen(
            [str(self.executor), "--confirmation", CONFIRMATION, "--router-status", str(self.status_path),
             "--request-path", str(self.request_path), "--status-file", str(self.executor_status)],
            start_new_session=True,
            text=True,
        )
        return self._emit(state="HANDOFF_PREP", reason="FIELD_SESSION_ARMED", transport_authorized=False,
                          motion_lease_available=False)

    def stop(self, reason: str = "OPERATOR_STOP") -> dict[str, Any]:
        if self._executor_process is not None and self._executor_process.poll() is None:
            self._executor_process.terminate()
            try:
                self._executor_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._executor_process.kill()
                self._executor_process.wait(timeout=1.0)
        self._executor_process = None
        try:
            self.request_path.unlink()
        except FileNotFoundError:
            pass
        return self._emit(state="NATIVE_READY", reason=str(reason) or "OPERATOR_STOP")

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            bounded = validate_controller_request(payload)
        except RequestGateError as exc:
            raise ValueError(str(exc)) from exc
        response = self.status()
        request_record = dict(bounded)
        request_record["schema_version"] = 1
        request_record["updated_at_unix"] = time.time()
        self._write_atomic(self.request_path, request_record)
        if not response["vendor_sport_available"]:
            result = self._emit(
                state="ENHANCED_WALK_BLOCKED",
                reason="MOTION_LEASE_UNAVAILABLE",
            )
        elif not response["motion_lease_available"]:
            result = self._emit(
                state="ENHANCED_WALK_BLOCKED",
                reason="MOTION_LEASE_UNAVAILABLE",
            )
        elif response.get("executor_state") != "ENHANCED_WALK":
            result = self._emit(
                state="HANDOFF_PREP",
                reason="WAITING_FOR_DEADMAN_AND_ROUTE",
                motion_lease_available=True,
            )
        else:
            result = self._emit(
                state="ENHANCED_WALK",
                reason="HIGH_LEVEL_REQUEST_ACCEPTED",
                transport_authorized=True,
                motion_lease_available=True,
            )
        result["request"] = bounded
        return result


__all__ = ["CONFIRMATION", "TerrainControlSupervisor"]
