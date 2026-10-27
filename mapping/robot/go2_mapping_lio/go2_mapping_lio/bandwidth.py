"""Application-layer bandwidth contracts for Go2 mapping transport."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Optional


@dataclass(frozen=True)
class DualQueueBudget:
    """Reserve the current real-time preview demand before bulk transfer."""

    total_mbps: float
    preview_demand_mbps: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_mbps) or self.total_mbps <= 0:
            raise ValueError("total_mbps must be finite and positive")
        # No fixed percentage is reserved. The live producer publishes its
        # measured RGB/RGB-D demand; until that heartbeat arrives, preview has
        # no claimed demand and the caller may apply an explicit startup floor.
        demand = 0.0 if self.preview_demand_mbps is None else self.preview_demand_mbps
        if not math.isfinite(demand) or demand < 0:
            raise ValueError("preview_demand_mbps must be finite and non-negative")
        object.__setattr__(self, "preview_demand_mbps", float(demand))

    @property
    def preview_mbps(self) -> float:
        return min(self.total_mbps, self.preview_demand_mbps)

    @property
    def high_fidelity_mbps(self) -> float:
        return self.total_mbps - self.preview_mbps

    @property
    def preview_policy(self) -> str:
        return "realtime_preview_priority_high_fidelity_remaining_capacity"

    @property
    def effective_preview_share(self) -> float:
        return self.preview_mbps / self.total_mbps

    @property
    def preview_share(self) -> float:
        return self.effective_preview_share

    @property
    def effective_high_fidelity_share(self) -> float:
        return self.high_fidelity_mbps / self.total_mbps

    @property
    def preview_share_percent(self) -> float:
        return self.effective_preview_share * 100.0

    @property
    def high_fidelity_share_percent(self) -> float:
        return self.effective_high_fidelity_share * 100.0

    @property
    def preview_bytes_per_second(self) -> float:
        return self.preview_mbps * 1_000_000.0 / 8.0

    @property
    def high_fidelity_bytes_per_second(self) -> float:
        return self.high_fidelity_mbps * 1_000_000.0 / 8.0


class PreviewDemandStore:
    """Atomic, short-lived reservation published by the latest-only preview path."""

    def __init__(self, path: Path, *, max_age_s: float = 3.0) -> None:
        self.path = Path(path)
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.max_age_s = float(max_age_s)
        self._publish_lock = threading.Lock()

    def publish(self, *, required_mbps: float, payload_bytes: int, cadence_hz: float, now: Optional[float] = None) -> None:
        if not math.isfinite(required_mbps) or required_mbps < 0:
            raise ValueError("required_mbps must be finite and non-negative")
        if payload_bytes < 0 or not math.isfinite(cadence_hz) or cadence_hz <= 0:
            raise ValueError("preview demand payload is invalid")
        updated_at = time.time() if now is None else float(now)
        payload = {
            "schema_version": 1,
            "updated_at": updated_at,
            "required_mbps": float(required_mbps),
            "payload_bytes": int(payload_bytes),
            "cadence_hz": float(cadence_hz),
            "policy": "realtime_preview_priority_high_fidelity_remaining_capacity",
        }
        with self._publish_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, self.path)

    def read(self, *, now: Optional[float] = None) -> Optional[float]:
        current = time.time() if now is None else float(now)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            updated_at = float(payload["updated_at"])
            required_mbps = float(payload["required_mbps"])
            if int(payload["schema_version"]) != 1 or current < updated_at or current - updated_at > self.max_age_s:
                return None
            if not math.isfinite(required_mbps) or required_mbps < 0:
                return None
            return required_mbps
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


class TokenBucket:
    """Reservation-based token bucket with deterministic time injection for tests."""

    def __init__(
        self,
        rate_bytes_per_second: float,
        burst_seconds: float = 1.0,
        now: Optional[float] = None,
    ) -> None:
        if not math.isfinite(rate_bytes_per_second) or rate_bytes_per_second <= 0:
            raise ValueError("rate_bytes_per_second must be finite and positive")
        if not math.isfinite(burst_seconds) or burst_seconds <= 0:
            raise ValueError("burst_seconds must be finite and positive")
        self.rate_bytes_per_second = float(rate_bytes_per_second)
        self.capacity_bytes = self.rate_bytes_per_second * float(burst_seconds)
        self._tokens = self.capacity_bytes
        self._updated_at = time.monotonic() if now is None else float(now)
        self._lock = threading.RLock()

    def set_rate(self, rate_bytes_per_second: float, now: Optional[float] = None) -> None:
        """Adopt a measured rate without granting a fresh burst of tokens."""
        if not math.isfinite(rate_bytes_per_second) or rate_bytes_per_second <= 0:
            raise ValueError("rate_bytes_per_second must be finite and positive")
        with self._lock:
            current = time.monotonic() if now is None else float(now)
            if not math.isfinite(current):
                raise ValueError("now must be finite")
            elapsed = max(0.0, current - self._updated_at)
            self._tokens = min(self.capacity_bytes, self._tokens + elapsed * self.rate_bytes_per_second)
            self._updated_at = max(current, self._updated_at)
            burst_seconds = self.capacity_bytes / self.rate_bytes_per_second
            self.rate_bytes_per_second = float(rate_bytes_per_second)
            self.capacity_bytes = self.rate_bytes_per_second * burst_seconds
            self._tokens = min(self._tokens, self.capacity_bytes)

    def delay_for(self, byte_count: int, now: Optional[float] = None) -> float:
        """Reserve bytes and return the delay required before they may be sent.

        Reservations are ordered even when the caller invokes this method again
        before sleeping, preventing a burst of preview frames from borrowing the
        high-fidelity queue's allocation.
        """
        if byte_count < 0:
            raise ValueError("byte_count must not be negative")
        with self._lock:
            current = time.monotonic() if now is None else float(now)
            if not math.isfinite(current):
                raise ValueError("now must be finite")
            elapsed = max(0.0, current - self._updated_at)
            self._tokens = min(self.capacity_bytes, self._tokens + elapsed * self.rate_bytes_per_second)
            self._updated_at = max(current, self._updated_at)
            if byte_count <= self._tokens:
                self._tokens -= byte_count
                return 0.0
            queued_delay = max(0.0, self._updated_at - current)
            service_delay = (byte_count - self._tokens) / self.rate_bytes_per_second
            self._tokens = 0.0
            self._updated_at = max(current, self._updated_at) + service_delay
            return queued_delay + service_delay


@dataclass(frozen=True)
class GoodputSample:
    byte_count: int
    elapsed_s: float

    @property
    def goodput_mbps(self) -> float:
        return self.byte_count * 8.0 / self.elapsed_s / 1_000_000.0


class VerifiedAckGoodputWindow:
    """Aggregate verified bytes across concurrent upload workers."""

    def __init__(self, *, minimum_interval_s: float = 5.0, now: Optional[float] = None) -> None:
        if not math.isfinite(minimum_interval_s) or minimum_interval_s <= 0:
            raise ValueError("minimum_interval_s must be finite and positive")
        started_at = time.monotonic() if now is None else float(now)
        if not math.isfinite(started_at):
            raise ValueError("now must be finite")
        self.minimum_interval_s = float(minimum_interval_s)
        self._started_at = started_at
        self._pending_bytes = 0
        self._lock = threading.Lock()

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._pending_bytes

    def note_verified(self, byte_count: int, *, now: Optional[float] = None) -> Optional[GoodputSample]:
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("now must be finite")
        with self._lock:
            if current < self._started_at:
                raise ValueError("now must not move backwards")
            self._pending_bytes += byte_count
            elapsed_s = current - self._started_at
            if elapsed_s < self.minimum_interval_s:
                return None
            sample = GoodputSample(self._pending_bytes, elapsed_s)
            self._pending_bytes = 0
            self._started_at = current
            return sample


class AdaptiveBandwidthController:
    """AIMD-like link estimator driven by verified high-fidelity chunks.

    Congestion causes an immediate reduction to 90% of observed usable
    throughput. A fully utilized interval probes upward by a bounded fraction.
    The measured total is passed through :class:`DualQueueBudget`; the current
    real-time preview demand is reserved before durable high-fidelity data.
    """

    def __init__(
        self,
        *,
        ceiling_mbps: float,
        floor_mbps: float = 0.1,
        initial_total_mbps: Optional[float] = None,
        probe_gain: float = 0.10,
        congestion_ratio: float = 0.85,
        safety_factor: float = 0.90,
        preview_demand_mbps: Optional[float] = None,
    ) -> None:
        values = (ceiling_mbps, floor_mbps, probe_gain, congestion_ratio, safety_factor)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("adaptive bandwidth parameters must be finite and positive")
        if floor_mbps > ceiling_mbps:
            raise ValueError("floor_mbps must not exceed ceiling_mbps")
        if not 0 < probe_gain <= 1:
            raise ValueError("probe_gain must be in (0, 1]")
        if not 0 < congestion_ratio <= 1 or not 0 < safety_factor <= 1:
            raise ValueError("congestion_ratio and safety_factor must be in (0, 1]")
        initial = floor_mbps if initial_total_mbps is None else initial_total_mbps
        if not math.isfinite(initial) or not floor_mbps <= initial <= ceiling_mbps:
            raise ValueError("initial_total_mbps must be within the controller bounds")
        self.ceiling_mbps = float(ceiling_mbps)
        self.floor_mbps = float(floor_mbps)
        self.probe_gain = float(probe_gain)
        self.congestion_ratio = float(congestion_ratio)
        self.safety_factor = float(safety_factor)
        self._slow_start = initial_total_mbps is None
        self._preview_demand_mbps = None if preview_demand_mbps is None else float(preview_demand_mbps)
        self._budget = self._make_budget(float(initial))

    def _make_budget(self, total_mbps: float) -> DualQueueBudget:
        return DualQueueBudget(total_mbps, self._preview_demand_mbps)

    def set_preview_demand(self, required_mbps: float) -> DualQueueBudget:
        if not math.isfinite(required_mbps) or required_mbps < 0:
            raise ValueError("required_mbps must be finite and non-negative")
        self._preview_demand_mbps = float(required_mbps)
        self._budget = self._make_budget(self._budget.total_mbps)
        return self._budget

    @property
    def budget(self) -> DualQueueBudget:
        return self._budget

    def observe_high_fidelity_transfer(self, *, byte_count: int, elapsed_s: float) -> DualQueueBudget:
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError("elapsed_s must be finite and positive")
        observed_high_mbps = byte_count * 8.0 / elapsed_s / 1_000_000.0
        expected_high_mbps = self._budget.high_fidelity_mbps
        utilization = observed_high_mbps / expected_high_mbps
        if utilization < self.congestion_ratio:
            total_mbps = self._total_for_high_fidelity(observed_high_mbps * self.safety_factor)
            self._slow_start = False
        else:
            gain = 1.0 if self._slow_start else self.probe_gain
            total_mbps = self._budget.total_mbps * (1.0 + gain)
        total_mbps = min(self.ceiling_mbps, max(self.floor_mbps, total_mbps))
        self._budget = self._make_budget(total_mbps)
        return self._budget

    def _total_for_high_fidelity(self, target_high_mbps: float) -> float:
        return target_high_mbps + (self._preview_demand_mbps or 0.0)

    def observe_capacity_probe(self, *, byte_count: int, elapsed_s: float) -> DualQueueBudget:
        """Adopt a conservative total budget from one unthrottled durable block."""
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError("elapsed_s must be finite and positive")
        observed_high_mbps = byte_count * 8.0 / elapsed_s / 1_000_000.0
        # The probe bypasses both queue token buckets and measures the whole
        # application path, before reserving preview bandwidth.
        total_mbps = observed_high_mbps * self.safety_factor
        total_mbps = min(self.ceiling_mbps, max(self.floor_mbps, total_mbps))
        self._slow_start = False
        self._budget = self._make_budget(total_mbps)
        return self._budget


class BandwidthAllocationStore:
    """Atomic shared state between the durable uploader and live preview."""

    def __init__(self, path: Path, *, max_age_s: float = 30.0) -> None:
        self.path = Path(path)
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.max_age_s = float(max_age_s)

    def publish(
        self,
        budget: DualQueueBudget,
        *,
        observed_high_fidelity_mbps: float,
        sample_elapsed_s: float,
        route_id: str,
        ack_rtt_ms: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        updated_at = time.time() if now is None else float(now)
        if not route_id or len(route_id) > 512 or any(char in route_id for char in "\r\n"):
            raise ValueError("route_id must be a non-empty single-line identity")
        if ack_rtt_ms is not None and (not math.isfinite(ack_rtt_ms) or ack_rtt_ms < 0):
            raise ValueError("ack_rtt_ms must be finite and non-negative")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (observed_high_fidelity_mbps, sample_elapsed_s, updated_at)
        ):
            raise ValueError("bandwidth state values must be finite and non-negative")
        payload = {
            "schema_version": 3,
            "updated_at": updated_at,
            "route_id": route_id,
            "total_mbps": budget.total_mbps,
            "preview_mbps": budget.preview_mbps,
            "high_fidelity_mbps": budget.high_fidelity_mbps,
            "preview_policy": budget.preview_policy,
            "preview_demand_mbps": budget.preview_demand_mbps,
            "preview_share": budget.effective_preview_share,
            "high_fidelity_share": budget.effective_high_fidelity_share,
            "observed_high_fidelity_mbps": observed_high_fidelity_mbps,
            "sample_elapsed_s": sample_elapsed_s,
            "ack_rtt_ms": ack_rtt_ms,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.path)

    def read(
        self,
        *,
        now: Optional[float] = None,
        expected_route_id: Optional[str] = None,
    ) -> Optional[DualQueueBudget]:
        current = time.time() if now is None else float(now)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            updated_at = float(payload["updated_at"])
            if current < updated_at or current - updated_at > self.max_age_s:
                return None
            if int(payload["schema_version"]) != 3:
                return None
            if expected_route_id is not None and str(payload["route_id"]) != expected_route_id:
                return None
            demand = payload.get("preview_demand_mbps")
            if demand is None and "preview_mbps" in payload:
                demand = payload["preview_mbps"]
            return DualQueueBudget(float(payload["total_mbps"]), None if demand is None else float(demand))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
