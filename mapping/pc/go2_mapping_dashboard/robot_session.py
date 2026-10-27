"""Thread-safe robot capture heartbeat state for the operator console."""
from __future__ import annotations

import math
import re
import threading
import time
from typing import Optional


_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TASK_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SUPPORTED_PREVIEW_POLICIES = {
    "realtime_preview_priority_high_fidelity_remaining_capacity",
    # Keep an already-running release observable while it is being replaced.
    "strict_dynamic_5pct_preview_95pct_high_fidelity",
}


class RobotSessionTelemetry:
    def __init__(self, max_age_s: float = 30.0) -> None:
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.max_age_s = float(max_age_s)
        self._lock = threading.Lock()
        self._payload: Optional[dict] = None
        self._received_at: Optional[float] = None

    def ingest(self, payload: dict, *, now: Optional[float] = None) -> dict:
        session_id = str(payload.get("session_id", ""))
        mode = str(payload.get("mode", ""))
        state = str(payload.get("state", ""))
        capture_limit_gib = int(payload.get("capture_limit_gib", 0))
        preview_policy = str(payload.get("preview_policy", ""))
        mission_id = payload.get("mission_id")
        task_id = payload.get("task_id")
        task_revision = payload.get("task_revision")
        lease_id = payload.get("lease_id")
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("session_id is unsafe")
        if mode not in {"full", "baseline"}:
            raise ValueError("mode must be full or baseline")
        if state not in {"STARTING", "RECORDING", "FINALIZING", "STOPPED"}:
            raise ValueError("state must be STARTING, RECORDING, FINALIZING, or STOPPED")
        if not 1 <= capture_limit_gib <= 40:
            raise ValueError("capture_limit_gib must be within [1, 40]")
        if preview_policy not in _SUPPORTED_PREVIEW_POLICIES:
            raise ValueError("preview_policy is unsupported")
        if mission_id is not None and _IDENTIFIER.fullmatch(str(mission_id)) is None:
            raise ValueError("mission_id is unsafe")
        if task_id is not None and _TASK_IDENTIFIER.fullmatch(str(task_id)) is None:
            raise ValueError("task_id is unsafe")
        if task_revision is not None and (isinstance(task_revision, bool) or not isinstance(task_revision, int) or task_revision <= 0):
            raise ValueError("task_revision must be a positive integer")
        if lease_id is not None and _IDENTIFIER.fullmatch(str(lease_id)) is None:
            raise ValueError("lease_id is unsafe")
        received_at = time.monotonic() if now is None else float(now)
        normalized = {
            "session_id": session_id,
            "mode": mode,
            "state": state,
            "capture_limit_gib": capture_limit_gib,
            "preview_policy": preview_policy,
            "mission_id": None if mission_id is None else str(mission_id),
            "task_id": None if task_id is None else str(task_id),
            "task_revision": task_revision,
            "lease_id": None if lease_id is None else str(lease_id),
            "preview_share": None,
            "high_fidelity_share": None,
        }
        with self._lock:
            self._payload = normalized
            self._received_at = received_at
        return self.snapshot(now=received_at)

    def snapshot(self, *, now: Optional[float] = None) -> dict:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            payload = None if self._payload is None else dict(self._payload)
            received_at = self._received_at
        if payload is None or received_at is None:
            return {
                "state": "UNKNOWN",
                "capture_active": False,
                "session_id": None,
                "mode": None,
                "capture_limit_gib": None,
                "preview_policy": "realtime_preview_priority_high_fidelity_remaining_capacity",
                "mission_id": None,
                "task_id": None,
                "task_revision": None,
                "lease_id": None,
                "preview_share": None,
                "high_fidelity_share": None,
                "heartbeat_age_ms": None,
            }
        age_s = max(0.0, current - received_at)
        stale = age_s > self.max_age_s
        payload["heartbeat_age_ms"] = int(age_s * 1000.0)
        payload["capture_active"] = payload["state"] == "RECORDING" and not stale
        if stale:
            payload["state"] = "STALE"
        return payload
