"""Bounded latest-frame worker for optional PC-side monocular assistance."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .depth_anything import DepthAnythingProvider, RelativeDepthFrame
from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


@dataclass(frozen=True)
class DepthAnythingWorkItem:
    frame_sha256: str
    source_generation_id: str
    source_revision_id: str
    calibration_sha256: str
    physical_depth_validation_revision: str
    model_revision_id: str
    assistance_config_sha256: str
    rectified_rgb: np.ndarray
    anchors: tuple[Any, ...]

    def __post_init__(self) -> None:
        for name in (
            "frame_sha256",
            "source_generation_id",
            "source_revision_id",
            "calibration_sha256",
            "physical_depth_validation_revision",
            "model_revision_id",
            "assistance_config_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        rgb = np.array(self.rectified_rgb, dtype=np.uint8, copy=True, order="C")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rectified_rgb must be an H by W by 3 uint8 array")
        rgb.setflags(write=False)
        object.__setattr__(self, "rectified_rgb", rgb)
        object.__setattr__(self, "anchors", tuple(self.anchors))

    @property
    def identity_key(self) -> tuple[str, str, str, str]:
        return (
            self.source_generation_id,
            self.model_revision_id,
            self.assistance_config_sha256,
            self.frame_sha256,
        )

    @property
    def generation_key(self) -> tuple[str, str, str]:
        return (
            self.source_generation_id,
            self.model_revision_id,
            self.assistance_config_sha256,
        )


@dataclass(frozen=True)
class DepthAnythingWorkResult:
    accepted: bool
    fit_accepted: bool
    assistance_coverage: float
    blockers: tuple[str, ...]
    augmentation_reference: Mapping[str, Any] | None
    validation_metrics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        coverage = float(self.assistance_coverage)
        if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("assistance_coverage must be finite and in [0, 1]")
        object.__setattr__(self, "assistance_coverage", coverage)
        object.__setattr__(self, "blockers", tuple(dict.fromkeys(self.blockers)))
        if self.accepted and (not self.fit_accepted or self.augmentation_reference is None):
            raise ValueError("accepted assistance requires an accepted fit and immutable reference")


class DepthAnythingWorker:
    """Keeps one pending frame and isolates all optional assistance failures."""

    def __init__(
        self,
        provider: DepthAnythingProvider | None,
        processor: Callable[[DepthAnythingWorkItem, RelativeDepthFrame], DepthAnythingWorkResult | None],
        *,
        enabled: bool = True,
        assistance_config_sha256: str | None = None,
    ) -> None:
        self._provider = provider
        self._processor = processor
        self._enabled = bool(enabled)
        self.assistance_config_sha256 = (
            load_reconstruction_config().appearance_sha256
            if assistance_config_sha256 is None
            else assistance_config_sha256
        )
        self._condition = threading.Condition(threading.RLock())
        self._pending: DepthAnythingWorkItem | None = None
        self._generation_key: tuple[str, str, str] | None = None
        self._processed_keys: set[tuple[str, str, str, str]] = set()
        self._thread: threading.Thread | None = None
        self._closing = False
        self._submitted_frames = 0
        self._processed_frames = 0
        self._failed_frames = 0
        self._coalesced_frames = 0
        self._evidence_drops = 0
        self._idempotent_replays = 0
        self._identity_invalidations = 0
        self._last_inference_ms: float | None = None
        self._last_error: str | None = None
        self._fit_accepted = False
        self._assistance_coverage = 0.0
        self._augmentation_reference: dict[str, Any] | None = None
        self._validation_metrics: dict[str, Any] | None = None
        self._blockers: list[str] = []
        if provider is None:
            self._record_blocker("DEPTH_ANYTHING_PROVIDER_UNAVAILABLE")
        if not self._enabled:
            self._record_blocker("DEPTH_ANYTHING_CPU_ASSISTANCE_DISABLED")

    @property
    def provider(self) -> DepthAnythingProvider | None:
        return self._provider

    @property
    def model_revision_id(self) -> str | None:
        value = None if self._provider is None else getattr(self._provider, "model_revision_id", None)
        return value if isinstance(value, str) and _SHA256_RE.fullmatch(value) else None

    def submit(self, item: DepthAnythingWorkItem) -> bool:
        if not isinstance(item, DepthAnythingWorkItem):
            raise TypeError("item must be a DepthAnythingWorkItem")
        with self._condition:
            if not self._enabled or self._provider is None:
                return False
            if not item.anchors:
                self._evidence_drops += 1
                self._record_blocker("ACCEPTED_LIDAR_ANCHORS_REQUIRED")
                return False
            if item.identity_key in self._processed_keys or (
                self._pending is not None and self._pending.identity_key == item.identity_key
            ):
                self._idempotent_replays += 1
                return False
            if self._generation_key is None:
                self._generation_key = item.generation_key
            elif self._generation_key != item.generation_key:
                if self._pending is not None:
                    self._coalesced_frames += 1
                self._pending = None
                self._generation_key = item.generation_key
                self._processed_keys.clear()
                self._fit_accepted = False
                self._assistance_coverage = 0.0
                self._augmentation_reference = None
                self._validation_metrics = None
                self._identity_invalidations += 1
            if self._pending is not None:
                self._coalesced_frames += 1
            self._pending = item
            self._submitted_frames += 1
            self._condition.notify()
            return True

    def invalidate_source_generation(self, source_generation_id: str) -> None:
        if not isinstance(source_generation_id, str) or _SHA256_RE.fullmatch(source_generation_id) is None:
            raise ValueError("source_generation_id must be a lowercase SHA-256")
        with self._condition:
            if self._generation_key is None or self._generation_key[0] == source_generation_id:
                return
            self._pending = None
            self._generation_key = None
            self._processed_keys.clear()
            self._fit_accepted = False
            self._assistance_coverage = 0.0
            self._augmentation_reference = None
            self._validation_metrics = None
            self._identity_invalidations += 1

    def run_once(self) -> bool:
        with self._condition:
            if not self._enabled or self._provider is None or self._pending is None:
                return False
            item = self._pending
            self._pending = None
        started = time.perf_counter()
        try:
            relative = self._provider.infer(item.rectified_rgb, frame_sha256=item.frame_sha256)
            inference_ms = (time.perf_counter() - started) * 1000.0
            if relative.frame_sha256 != item.frame_sha256:
                raise ValueError("provider returned a stale frame identity")
            if relative.model_revision_id != item.model_revision_id:
                raise ValueError("provider model revision does not match queued work")
            result = self._processor(item, relative)
            if result is not None and not isinstance(result, DepthAnythingWorkResult):
                raise TypeError("processor must return DepthAnythingWorkResult or None")
        except Exception as exc:
            with self._condition:
                self._failed_frames += 1
                self._last_inference_ms = (time.perf_counter() - started) * 1000.0
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._fit_accepted = False
                self._augmentation_reference = None
                self._validation_metrics = None
                self._record_blocker("DEPTH_ANYTHING_INFERENCE_FAILED")
            return False

        with self._condition:
            self._last_inference_ms = inference_ms
            self._last_error = None
            self._processed_frames += 1
            self._processed_keys.add(item.identity_key)
            if result is None:
                self._fit_accepted = False
                self._assistance_coverage = 0.0
                self._augmentation_reference = None
                self._record_blocker("DEPTH_ANYTHING_ASSISTANCE_REJECTED")
                return False
            self._fit_accepted = result.fit_accepted
            self._assistance_coverage = result.assistance_coverage
            self._augmentation_reference = (
                None if result.augmentation_reference is None else dict(result.augmentation_reference)
            )
            self._validation_metrics = (
                None if result.validation_metrics is None else dict(result.validation_metrics)
            )
            for blocker in result.blockers:
                self._record_blocker(blocker)
            if not result.accepted:
                self._augmentation_reference = None
            return result.accepted

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            if not self._enabled or self._provider is None:
                return
            self._closing = False
            self._thread = threading.Thread(
                target=self._run_loop,
                name="go2-depth-anything-latest-frame",
                daemon=True,
            )
            self._thread.start()

    def close(self, timeout_s: float = 2.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout_s)))

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            if not self._enabled:
                state = "DEPTH_ANYTHING_DISABLED"
            elif self._provider is None:
                state = "DEPTH_ANYTHING_PROVIDER_UNAVAILABLE"
            elif self._pending is not None:
                state = "DEPTH_ANYTHING_PENDING"
            elif self._augmentation_reference is not None:
                state = "RGBD_AUGMENTATION_ACCEPTED"
            elif self._last_error is not None:
                state = "DEPTH_ANYTHING_FAILED"
            else:
                state = "DEPTH_ANYTHING_IDLE"
            generation = None if self._generation_key is None else self._generation_key[0]
            model = None if self._generation_key is None else self._generation_key[1]
            config = None if self._generation_key is None else self._generation_key[2]
            return {
                "state": state,
                "queue_capacity": 1,
                "pending_frames": int(self._pending is not None),
                "submitted_frames": self._submitted_frames,
                "processed_frames": self._processed_frames,
                "failed_frames": self._failed_frames,
                "coalesced_frames": self._coalesced_frames,
                "evidence_drops": self._evidence_drops,
                "idempotent_replays": self._idempotent_replays,
                "identity_invalidations": self._identity_invalidations,
                "source_generation_id": generation,
                "model_revision_id": model,
                "assistance_config_sha256": config,
                "last_inference_ms": self._last_inference_ms,
                "fit_accepted": self._fit_accepted,
                "assistance_coverage": self._assistance_coverage,
                "augmentation_reference": None
                if self._augmentation_reference is None
                else dict(self._augmentation_reference),
                "validation_metrics": None
                if self._validation_metrics is None
                else dict(self._validation_metrics),
                "last_error": self._last_error,
                "blockers": list(dict.fromkeys(self._blockers)),
            }

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or self._pending is not None)
                if self._closing:
                    return
            self.run_once()

    def _record_blocker(self, blocker: str) -> None:
        if blocker not in self._blockers:
            self._blockers.append(blocker)
