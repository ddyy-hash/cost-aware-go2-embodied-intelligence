"""Conservative LiDAR clock alignment for the live, provisional LIO stream.

The raw capture stream keeps the timestamp supplied by the Unitree DDS
message.  This helper is deliberately used only on a separate live LIO topic:
it translates a stable LiDAR-clock offset into the host/IMU clock and keeps a
small guard interval so LIO-SAM has received the IMU samples for a scan.
"""
from __future__ import annotations

from collections import deque
from statistics import median
from typing import Deque


NANOSECONDS_PER_SECOND = 1_000_000_000


class HeaderProgressGate:
    """Require source timestamps to advance without assuming a host epoch."""

    def __init__(self, *, minimum_progress_ms: float) -> None:
        if minimum_progress_ms <= 0.0:
            raise ValueError("minimum_progress_ms must be positive")
        self._minimum_progress_ns = int(round(minimum_progress_ms * 1_000_000))
        self._first_timestamp_ns: int | None = None

    def observe(self, timestamp_ns: int) -> bool:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
            return False
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = timestamp_ns
            return False
        return timestamp_ns - self._first_timestamp_ns >= self._minimum_progress_ns


class LioCloudClockAligner:
    """Align a stable source clock to host arrival time without following jitter."""

    def __init__(self, guard_ms: float = 160.0, max_offset_step_ms: float = 250.0, window_size: int = 31) -> None:
        if guard_ms < 0:
            raise ValueError("guard_ms must be non-negative")
        if max_offset_step_ms <= 0:
            raise ValueError("max_offset_step_ms must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._guard_ns = int(round(guard_ms * 1_000_000))
        self._max_offset_step_ns = int(round(max_offset_step_ms * 1_000_000))
        self._offsets_ns: Deque[int] = deque(maxlen=window_size)

    def align(self, source_stamp_ns: int, arrival_stamp_ns: int) -> int:
        """Return a host-clock scan-start timestamp suitable for LIO only."""
        if arrival_stamp_ns <= 0:
            raise ValueError("arrival_stamp_ns must be positive")
        if source_stamp_ns <= 0:
            return max(1, arrival_stamp_ns - self._guard_ns)

        candidate_offset_ns = arrival_stamp_ns - source_stamp_ns
        if not self._offsets_ns:
            self._offsets_ns.append(candidate_offset_ns)
        else:
            baseline_offset_ns = int(round(median(self._offsets_ns)))
            if abs(candidate_offset_ns - baseline_offset_ns) <= self._max_offset_step_ns:
                self._offsets_ns.append(candidate_offset_ns)

        offset_ns = int(round(median(self._offsets_ns)))
        return max(1, source_stamp_ns + offset_ns - self._guard_ns)
