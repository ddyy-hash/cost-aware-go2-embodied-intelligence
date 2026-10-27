"""Truthful application-level telemetry for the Go2 live keyframe stream."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Optional


@dataclass(frozen=True)
class _Arrival:
    received_at: float
    payload_bytes: int


class LiveTransportMetrics:
    """Tracks PC-observed stream health without assuming clock synchronization."""

    def __init__(self, window_seconds: float = 30.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._arrivals: Deque[_Arrival] = deque()
        self._last_ack_rtt_ms: Optional[float] = None
        self._duplicate_frames = 0
        self._camera_deferred_by_bandwidth = False
        self._camera_depth_deferred_by_bandwidth = False
        self._last_received_at: Optional[float] = None

    def record(
        self,
        payload_bytes: int,
        acknowledgement_rtt_ms: Optional[float],
        received_at: Optional[float] = None,
        duplicate: bool = False,
        camera_deferred_by_bandwidth: bool = False,
        camera_depth_deferred_by_bandwidth: bool = False,
    ) -> None:
        if payload_bytes <= 0:
            raise ValueError("payload_bytes must be positive")
        if acknowledgement_rtt_ms is not None:
            if not math.isfinite(acknowledgement_rtt_ms) or acknowledgement_rtt_ms < 0:
                raise ValueError("acknowledgement_rtt_ms must be finite and non-negative")
            self._last_ack_rtt_ms = round(float(acknowledgement_rtt_ms), 1)
        now = time.monotonic() if received_at is None else received_at
        self._arrivals.append(_Arrival(now, payload_bytes))
        self._last_received_at = now
        if duplicate:
            self._duplicate_frames += 1
        self._camera_deferred_by_bandwidth = bool(camera_deferred_by_bandwidth)
        self._camera_depth_deferred_by_bandwidth = bool(camera_depth_deferred_by_bandwidth)
        self._trim(now)

    def snapshot(self, now: Optional[float] = None) -> dict:
        current = time.monotonic() if now is None else now
        self._trim(current)
        if not self._arrivals:
            return {
                "app_goodput_mbps": None,
                "keyframe_hz": None,
                "ack_rtt_ms": self._last_ack_rtt_ms,
                "last_frame_age_ms": None if self._last_received_at is None else int(max(0.0, current - self._last_received_at) * 1000.0),
                "duplicate_frames": self._duplicate_frames,
                "camera_deferred_by_bandwidth": self._camera_deferred_by_bandwidth,
                "camera_depth_deferred_by_bandwidth": self._camera_depth_deferred_by_bandwidth,
            }

        first = self._arrivals[0].received_at
        observation_seconds = max(current - first, 1.0)
        total_bytes = sum(arrival.payload_bytes for arrival in self._arrivals)
        return {
            "app_goodput_mbps": round(total_bytes * 8.0 / observation_seconds / 1_000_000.0, 3),
            "keyframe_hz": round(len(self._arrivals) / observation_seconds, 2),
            "ack_rtt_ms": self._last_ack_rtt_ms,
            "last_frame_age_ms": int(max(0.0, current - self._arrivals[-1].received_at) * 1000.0),
            "duplicate_frames": self._duplicate_frames,
            "camera_deferred_by_bandwidth": self._camera_deferred_by_bandwidth,
            "camera_depth_deferred_by_bandwidth": self._camera_depth_deferred_by_bandwidth,
        }

    def _trim(self, now: float) -> None:
        while self._arrivals and now - self._arrivals[0].received_at > self.window_seconds:
            self._arrivals.popleft()
