import json
import os
import time
from pathlib import Path


class SessionTelemetry:
    """Serves only an explicitly written mapping-session state; never invents map data."""

    def __init__(self, status_path=None):
        configured = status_path or os.environ.get("GO2_MAPPING_SESSION_STATUS")
        self.status_path = Path(configured) if configured else None
        self.sequence = 0

    def snapshot(self):
        self.sequence += 1
        state = self._load_state()
        if state is not None:
            state["sequence"] = self.sequence
            state["timestamp_ms"] = int(time.time() * 1000)
            return state
        return {
            "source": "AWAITING VERIFIED LIDAR",
            "sequence": self.sequence,
            "timestamp_ms": int(time.time() * 1000),
            "robot": {"position": [0.0, 0.0, 0.0], "yaw": 0.0, "speed_mps": 0.0},
            "mapping": {
                # A dashboard with no verified registration is only an
                # operational preview, never an as-built deliverable.
                "as_built_authority": False,
                "points": 0,
                "coverage_m2": 0.0,
                "update_hz": 0.0,
                "residual_cm": 0.0,
            },
            "transport": {"goodput_mbps": 0.0, "latency_ms": 0, "buffer_gib": 0.0},
            "bim": {"state": "unregistered", "support_groups": 0},
        }

    def _load_state(self):
        if self.status_path is None or not self.status_path.is_file():
            return None
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        required = {"source", "robot", "mapping", "transport", "bim"}
        return payload if required.issubset(payload) else None
