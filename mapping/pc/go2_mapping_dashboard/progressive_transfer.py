"""Read PC-receiver transport telemetry without fabricating unavailable data."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Optional


class ProgressiveTransferTelemetry:
    def __init__(self, status_path: Optional[Path] = None, max_age_s: float = 30.0) -> None:
        self.status_path = Path(status_path) if status_path else None
        self.max_age_s = float(max_age_s)

    def snapshot(self, now_ms: Optional[int] = None) -> dict:
        unavailable = {
            "state": "AWAITING_HIGH_FIDELITY_ARTIFACT",
            "active": False,
            "app_goodput_mbps": None,
            "accepted_chunks": 0,
            "verified_artifacts": 0,
            "verified_bytes": 0,
            "last_verified_age_ms": None,
            "updated_at_ms": None,
            "status_age_ms": None,
            "total_budget_mbps": None,
            "preview_budget_mbps": None,
            "high_fidelity_budget_mbps": None,
            "observed_capacity_mbps": None,
        }
        if self.status_path is None or not self.status_path.is_file():
            return unavailable
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
            high_fidelity = payload["high_fidelity"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return unavailable
        state = {**unavailable, **{key: value for key, value in high_fidelity.items() if key in unavailable}}
        current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        updated_at_ms = state.get("updated_at_ms")
        if not isinstance(updated_at_ms, (int, float)):
            return state
        age_ms = max(0, current_ms - int(updated_at_ms))
        state["status_age_ms"] = age_ms
        state["active"] = age_ms <= int(self.max_age_s * 1000)
        if not state["active"]:
            state["state"] = "STALE"
            state["app_goodput_mbps"] = None
            state["total_budget_mbps"] = None
            state["preview_budget_mbps"] = None
            state["high_fidelity_budget_mbps"] = None
            state["observed_capacity_mbps"] = None
        return state
