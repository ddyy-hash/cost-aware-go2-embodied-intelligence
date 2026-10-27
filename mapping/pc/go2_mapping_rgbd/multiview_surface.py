"""Bounded DA3 view windows and fail-closed neural surface publication gates."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Optional

import numpy as np

from mapping.pc.go2_mapping_live.live_surface_mesher import (
    MeshTileRevision,
    TriangleProvenance,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


def _require_real_numeric_dtype(array: np.ndarray, name: str) -> None:
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be numeric")


def _immutable_numeric_array(
    value: Any, dtype, *, shape_tail: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} has an invalid shape")
    _require_real_numeric_dtype(array, name)
    immutable = np.array(array, dtype=dtype, order="C", copy=True)
    immutable.setflags(write=False)
    return immutable


def _array_content_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": [int(size) for size in array.shape],
        "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _canonical_scalar(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "integer", "value": int(value)}
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("canonical scalar must be finite")
    return {"type": "float64-hex", "value": numeric.hex()}


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _freeze_metrics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metrics(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metrics(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metrics(item) for item in value)
    if isinstance(value, np.ndarray):
        immutable = np.array(value, copy=True)
        immutable.setflags(write=False)
        return immutable
    return value


@dataclass(frozen=True)
class Da3ViewObservation:
    frame_sha256: str
    independent_view_id: str
    timestamp_ns: int
    rectified_rgb: np.ndarray
    intrinsics: np.ndarray
    extrinsics_world_to_camera: np.ndarray
    aligned_depth_m: Optional[np.ndarray] = None
    building_id: Optional[str] = None
    session_id: Optional[str] = None
    submap_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_sha256, str) or _SHA256.fullmatch(self.frame_sha256) is None:
            raise ValueError("frame_sha256 must be a lowercase SHA-256")
        if not isinstance(self.independent_view_id, str) or not self.independent_view_id:
            raise ValueError("independent_view_id is required")
        if isinstance(self.timestamp_ns, bool) or int(self.timestamp_ns) <= 0:
            raise ValueError("timestamp_ns must be positive")
        rgb = np.asarray(self.rectified_rgb)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        extrinsics = np.asarray(self.extrinsics_world_to_camera, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rectified_rgb must be uint8 H by W by 3")
        if intrinsics.shape != (3, 3) or extrinsics.shape != (4, 4):
            raise ValueError("view camera matrices have invalid shapes")
        if (
            not np.isfinite(intrinsics).all()
            or not np.isfinite(extrinsics).all()
            or intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or not np.allclose(extrinsics[3], (0, 0, 0, 1), atol=1e-6)
        ):
            raise ValueError("view camera matrices must be finite calibrated transforms")
        for name, value in (
            ("rectified_rgb", rgb),
            ("intrinsics", intrinsics),
            ("extrinsics_world_to_camera", extrinsics),
        ):
            immutable = np.array(value, copy=True, order="C")
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        if self.aligned_depth_m is not None:
            depth = np.asarray(self.aligned_depth_m, dtype=np.float32)
            if depth.shape != rgb.shape[:2] or np.any(np.isinf(depth)) or np.any(depth[np.isfinite(depth)] <= 0):
                raise ValueError("aligned_depth_m must match RGB and contain positive finite depth or NaN")
            immutable_depth = np.array(depth, dtype=np.float32, order="C", copy=True)
            immutable_depth.setflags(write=False)
            object.__setattr__(self, "aligned_depth_m", immutable_depth)
        context = (self.building_id, self.session_id, self.submap_id)
        if any(value is not None for value in context):
            if any(not isinstance(value, str) or not value for value in context):
                raise ValueError("DA3 view Building, session, and submap context must be complete")


@dataclass(frozen=True)
class Da3ViewWindow:
    views: tuple[Da3ViewObservation, ...]
    reference_index: int
    window_sha256: str


class Da3WindowAssembler:
    def __init__(self, *, window_size: int = 5, overlap: int = 2) -> None:
        if window_size != 5 or overlap != 2:
            raise ValueError("DA3 surface windows are fixed at five views with two-view overlap")
        self.window_size = window_size
        self.overlap = overlap
        self._pending: list[Da3ViewObservation] = []
        self._seen: set[str] = set()
        self.duplicate_frames = 0

    @property
    def pending_view_count(self) -> int:
        return len(self._pending)

    def submit(self, observation: Da3ViewObservation) -> Optional[Da3ViewWindow]:
        if not isinstance(observation, Da3ViewObservation):
            raise TypeError("observation must be a Da3ViewObservation")
        if observation.frame_sha256 in self._seen:
            self.duplicate_frames += 1
            return None
        if self._pending and observation.timestamp_ns <= self._pending[-1].timestamp_ns:
            raise ValueError("DA3 observations must be submitted in timestamp order")
        self._seen.add(observation.frame_sha256)
        self._pending.append(observation)
        if len(self._pending) < self.window_size:
            return None
        views = tuple(self._pending[: self.window_size])
        identity = {
            "schema_version": 1,
            "frame_sha256s": [view.frame_sha256 for view in views],
            "independent_view_ids": [view.independent_view_id for view in views],
            "reference_index": self.window_size // 2,
        }
        window_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        self._pending = self._pending[self.window_size - self.overlap :]
        return Da3ViewWindow(views, self.window_size // 2, window_sha256)


class Da3MultiViewWorker:
    """Capacity-one DA3 worker whose failures disable only the active session."""

    def __init__(
        self,
        provider: Any,
        processor: Callable[[Da3ViewWindow, Any], Any],
    ) -> None:
        self._provider = provider
        self._processor = processor
        self._condition = threading.Condition(threading.RLock())
        self._pending: Optional[tuple[str, Da3ViewWindow]] = None
        self._active_session_id: Optional[str] = None
        self._disabled_session_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._closing = False
        self._submitted_windows = 0
        self._processed_windows = 0
        self._failed_windows = 0
        self._coalesced_windows = 0
        self._last_inference_ms: Optional[float] = None
        self._last_error: Optional[str] = None
        self._blockers: list[str] = []

    def submit(self, session_id: str, window: Da3ViewWindow) -> bool:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        if not isinstance(window, Da3ViewWindow):
            raise TypeError("window must be a Da3ViewWindow")
        with self._condition:
            if self._disabled_session_id == session_id:
                return False
            if self._active_session_id != session_id:
                self._pending = None
                self._active_session_id = session_id
                self._disabled_session_id = None
                self._last_error = None
                self._blockers = []
            if self._pending is not None:
                self._coalesced_windows += 1
            self._pending = (session_id, window)
            self._submitted_windows += 1
            self._condition.notify()
            return True

    def run_once(self) -> bool:
        with self._condition:
            if self._pending is None:
                return False
            session_id, window = self._pending
            self._pending = None
        images = [view.rectified_rgb for view in window.views]
        frame_sha256s = tuple(view.frame_sha256 for view in window.views)
        intrinsics = np.stack([view.intrinsics for view in window.views])
        extrinsics = np.stack([view.extrinsics_world_to_camera for view in window.views])
        started = time.perf_counter()
        try:
            prediction = self._provider.infer_multiview(
                images,
                frame_sha256s=frame_sha256s,
                intrinsics=intrinsics,
                extrinsics_world_to_camera=extrinsics,
            )
            elapsed = time.perf_counter() - started
            timeout_s = float(getattr(self._provider, "inference_timeout_s", 30.0))
            if elapsed > timeout_s:
                raise TimeoutError(f"DA3 window exceeded {timeout_s:.3f}s")
            self._processor(window, prediction)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            lowered = message.lower()
            if "out of memory" in lowered or "cuda oom" in lowered:
                blocker = "DA3_SESSION_DISABLED_OOM"
            elif isinstance(exc, TimeoutError) or "timeout" in lowered or "exceeded" in lowered:
                blocker = "DA3_SESSION_DISABLED_TIMEOUT"
            else:
                blocker = "DA3_SESSION_DISABLED_INFERENCE_FAILURE"
            with self._condition:
                self._failed_windows += 1
                self._last_inference_ms = (time.perf_counter() - started) * 1000.0
                self._last_error = message
                self._disabled_session_id = session_id
                if blocker not in self._blockers:
                    self._blockers.append(blocker)
            return False
        with self._condition:
            self._processed_windows += 1
            self._last_inference_ms = (time.perf_counter() - started) * 1000.0
            self._last_error = None
        return True

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closing = False
            self._thread = threading.Thread(target=self._run_loop, name="go2-da3-multiview", daemon=True)
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
            state = "DA3_SESSION_DISABLED" if self._disabled_session_id is not None else (
                "DA3_MULTIVIEW_PENDING" if self._pending is not None else "DA3_MULTIVIEW_IDLE"
            )
            return {
                "state": state,
                "queue_capacity": 1,
                "pending_windows": int(self._pending is not None),
                "active_session_id": self._active_session_id,
                "disabled_session_id": self._disabled_session_id,
                "submitted_windows": self._submitted_windows,
                "processed_windows": self._processed_windows,
                "failed_windows": self._failed_windows,
                "coalesced_windows": self._coalesced_windows,
                "last_inference_ms": self._last_inference_ms,
                "last_error": self._last_error,
                "blockers": list(self._blockers),
            }

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or self._pending is not None)
                if self._closing:
                    return
            self.run_once()


@dataclass(frozen=True)
class NeuralSurfaceCandidate:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_scale_m: np.ndarray
    supporting_frame_count: np.ndarray
    independent_view_count: np.ndarray
    holdout_median_residual_m: np.ndarray
    holdout_p95_residual_m: np.ndarray
    holdout_absolute_bias_m: np.ndarray
    uncertainty_m: np.ndarray
    observed_front: np.ndarray
    free_space_conflict: np.ndarray
    occlusion_conflict: np.ndarray
    depth_boundary_conflict: np.ndarray
    measured_surface_conflict: np.ndarray
    source_window_sha256: str
    near_object_size_holdout_error_mm: Optional[float] = None
    candidate_revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        positions = _immutable_numeric_array(
            self.positions, "<f4", shape_tail=(3,), name="positions"
        )
        normals = _immutable_numeric_array(
            self.normals, "<f4", shape_tail=(3,), name="normals"
        )
        if normals.shape != positions.shape or not np.isfinite(positions).all() or not np.isfinite(normals).all():
            raise ValueError("neural candidate geometry must be finite matching arrays")

        raw_triangles = np.asarray(self.triangles)
        if raw_triangles.ndim != 2 or raw_triangles.shape[1:] != (3,):
            raise ValueError("triangles has an invalid shape")
        if np.issubdtype(raw_triangles.dtype, np.bool_) or not np.issubdtype(
            raw_triangles.dtype, np.integer
        ):
            raise ValueError("triangles must contain integers")
        if len(raw_triangles) and np.any(raw_triangles < 0):
            raise ValueError("neural candidate triangle index cannot be negative")
        if len(raw_triangles) and int(raw_triangles.max()) >= len(positions):
            raise ValueError("neural candidate triangle index is out of range")
        triangles = np.array(raw_triangles, dtype="<u4", order="C", copy=True)
        triangles.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "triangles", triangles)
        triangle_count = len(triangles)
        fields = {
            "triangle_scale_m": ("<f4", "numeric"),
            "supporting_frame_count": ("u1", "integer"),
            "independent_view_count": ("u1", "integer"),
            "holdout_median_residual_m": ("<f4", "numeric"),
            "holdout_p95_residual_m": ("<f4", "numeric"),
            "holdout_absolute_bias_m": ("<f4", "numeric"),
            "uncertainty_m": ("<f4", "numeric"),
            "observed_front": (bool, "boolean"),
            "free_space_conflict": (bool, "boolean"),
            "occlusion_conflict": (bool, "boolean"),
            "depth_boundary_conflict": (bool, "boolean"),
            "measured_surface_conflict": (bool, "boolean"),
        }
        for name, (dtype, kind) in fields.items():
            raw_values = np.asarray(getattr(self, name))
            if raw_values.shape != (triangle_count,):
                raise ValueError(f"{name} must contain one value per triangle")
            if kind == "numeric":
                _require_real_numeric_dtype(raw_values, name)
            elif kind == "integer":
                if np.issubdtype(raw_values.dtype, np.bool_) or not np.issubdtype(
                    raw_values.dtype, np.integer
                ):
                    raise ValueError(f"{name} must contain integers")
                if np.any(raw_values < 0) or np.any(raw_values > np.iinfo(np.uint8).max):
                    raise ValueError(f"{name} must fit in an unsigned byte")
            elif not np.issubdtype(raw_values.dtype, np.bool_):
                raise ValueError(f"{name} must contain boolean values")
            immutable = np.array(raw_values, dtype=dtype, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        for name in (
            "triangle_scale_m",
            "holdout_median_residual_m",
            "holdout_p95_residual_m",
            "holdout_absolute_bias_m",
            "uncertainty_m",
        ):
            values = getattr(self, name)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if np.any(self.triangle_scale_m <= 0):
            raise ValueError("triangle_scale_m must be positive")
        if not isinstance(self.source_window_sha256, str) or _SHA256.fullmatch(self.source_window_sha256) is None:
            raise ValueError("source_window_sha256 must be a lowercase SHA-256")
        if self.near_object_size_holdout_error_mm is not None and (
            not math.isfinite(self.near_object_size_holdout_error_mm)
            or self.near_object_size_holdout_error_mm < 0
        ):
            raise ValueError("near_object_size_holdout_error_mm must be finite and non-negative")
        if self.near_object_size_holdout_error_mm is not None:
            object.__setattr__(
                self,
                "near_object_size_holdout_error_mm",
                float(self.near_object_size_holdout_error_mm),
            )
        object.__setattr__(
            self, "candidate_revision_id", self.recompute_candidate_revision_id()
        )

    def recompute_candidate_revision_id(self) -> str:
        array_fields = (
            "positions",
            "normals",
            "triangles",
            "triangle_scale_m",
            "supporting_frame_count",
            "independent_view_count",
            "holdout_median_residual_m",
            "holdout_p95_residual_m",
            "holdout_absolute_bias_m",
            "uncertainty_m",
            "observed_front",
            "free_space_conflict",
            "occlusion_conflict",
            "depth_boundary_conflict",
            "measured_surface_conflict",
        )
        identity = {
            "schema": "go2-neural-surface-candidate-content-v1",
            "arrays": {
                name: _array_content_identity(getattr(self, name))
                for name in array_fields
            },
            "source_window_sha256": self.source_window_sha256,
            "near_object_size_holdout_error_mm": _canonical_scalar(
                self.near_object_size_holdout_error_mm
            ),
        }
        return _identity_sha256(identity)

    @property
    def content_revision_id(self) -> str:
        return self.candidate_revision_id

    def replaced(self, **changes: Any) -> "NeuralSurfaceCandidate":
        return replace(self, **changes)


@dataclass(frozen=True)
class NeuralSurfaceDecision:
    accepted_triangle_indices: np.ndarray
    triangle_provenance: np.ndarray
    triangle_uncertainty_m: np.ndarray
    candidate_revision_id: str
    gate_policy_revision_id: str
    blockers: tuple[str, ...]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        indices = np.asarray(self.accepted_triangle_indices)
        if indices.ndim != 1:
            raise ValueError("accepted triangle indices must be one-dimensional")
        if np.issubdtype(indices.dtype, np.bool_) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("accepted triangle indices must contain integers")
        if len(indices) and np.any(indices < 0):
            raise ValueError("accepted triangle indices cannot be negative")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("accepted triangle indices cannot contain duplicates")
        immutable_indices = np.array(indices, order="C", copy=True)
        immutable_indices.setflags(write=False)
        object.__setattr__(self, "accepted_triangle_indices", immutable_indices)

        provenance = np.asarray(self.triangle_provenance)
        if provenance.shape != (len(indices),):
            raise ValueError("neural decision provenance has an invalid shape")
        if np.issubdtype(provenance.dtype, np.bool_) or not np.issubdtype(
            provenance.dtype, np.integer
        ):
            raise ValueError("neural decision provenance must contain integers")
        if np.any(provenance != int(TriangleProvenance.NEURAL_MULTIVIEW)):
            raise ValueError("neural decision provenance is invalid")
        immutable_provenance = np.array(provenance, dtype="u1", order="C", copy=True)
        immutable_provenance.setflags(write=False)
        object.__setattr__(self, "triangle_provenance", immutable_provenance)

        raw_uncertainty = np.asarray(self.triangle_uncertainty_m)
        if raw_uncertainty.shape != (len(indices),):
            raise ValueError("neural decision uncertainty has an invalid shape")
        if (
            not np.issubdtype(raw_uncertainty.dtype, np.number)
            or np.issubdtype(raw_uncertainty.dtype, np.bool_)
            or np.issubdtype(raw_uncertainty.dtype, np.complexfloating)
        ):
            raise ValueError("neural decision uncertainty must be numeric")
        uncertainty = np.array(raw_uncertainty, dtype="<f4", order="C", copy=True)
        if (
            not np.isfinite(uncertainty).all()
            or np.any(uncertainty < 0)
            or np.any(uncertainty > 1.0)
        ):
            raise ValueError("neural decision uncertainty is invalid")
        uncertainty.setflags(write=False)
        object.__setattr__(self, "triangle_uncertainty_m", uncertainty)

        for name in ("candidate_revision_id", "gate_policy_revision_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")

        blockers = tuple(self.blockers)
        if any(not isinstance(blocker, str) or not blocker for blocker in blockers):
            raise ValueError("neural decision blockers are invalid")
        object.__setattr__(self, "blockers", blockers)
        if not isinstance(self.metrics, Mapping):
            raise TypeError("neural decision metrics must be a mapping")
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))

    def validate_for(self, candidate: NeuralSurfaceCandidate) -> np.ndarray:
        if not isinstance(candidate, NeuralSurfaceCandidate):
            raise TypeError("candidate must be a NeuralSurfaceCandidate")
        indices = self.accepted_triangle_indices
        triangle_count = len(candidate.triangles)
        if len(indices) and int(indices.max()) >= triangle_count:
            raise ValueError("accepted triangle index is out of range for candidate")
        if self.metrics.get("source_window_sha256") != candidate.source_window_sha256:
            raise ValueError("neural decision source window does not match candidate")
        candidate_revision_id = candidate.recompute_candidate_revision_id()
        if candidate.candidate_revision_id != candidate_revision_id:
            raise ValueError("neural candidate revision failed integrity validation")
        if self.candidate_revision_id != candidate_revision_id:
            raise ValueError("neural decision candidate revision does not match candidate")
        gate_policy_revision_id = NeuralSurfaceGate.policy_revision_id()
        if self.gate_policy_revision_id != gate_policy_revision_id:
            raise ValueError("neural decision gate policy revision is stale")

        def require_metric_count(name: str, expected: int) -> None:
            value = self.metrics.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be an integer")
            if int(value) != expected:
                raise ValueError(f"{name} does not match candidate")

        require_metric_count("candidate_triangle_count", triangle_count)
        require_metric_count("accepted_triangle_count", len(indices))
        if "rejected_triangle_count" in self.metrics:
            require_metric_count(
                "rejected_triangle_count", triangle_count - len(indices)
            )
        selected = np.asarray(indices, dtype=np.intp)
        expected_uncertainty = np.asarray(candidate.uncertainty_m[selected], dtype="<f4")
        if not np.array_equal(self.triangle_uncertainty_m, expected_uncertainty):
            raise ValueError("neural decision does not match candidate uncertainty")
        return selected


class NeuralSurfaceGate:
    __slots__ = ()

    gate_algorithm_schema_version = 1
    minimum_frames = 3
    minimum_independent_views = 2
    maximum_holdout_median_m = 0.025
    maximum_holdout_p95_m = 0.060
    maximum_absolute_bias_m = 0.015
    maximum_uncertainty_m = 0.030

    @classmethod
    def policy_revision_id(cls) -> str:
        identity = {
            "schema": "go2-neural-surface-gate-policy-v1",
            "gate_algorithm_schema_version": _canonical_scalar(
                cls.gate_algorithm_schema_version
            ),
            "combination": "accept-only-when-every-rule-passes;ascending-indices",
            "rules": (
                {
                    "field": "supporting_frame_count",
                    "operator": ">=",
                    "threshold": _canonical_scalar(cls.minimum_frames),
                    "blocker": "MINIMUM_THREE_FRAME_SUPPORT_REQUIRED",
                },
                {
                    "field": "independent_view_count",
                    "operator": ">=",
                    "threshold": _canonical_scalar(cls.minimum_independent_views),
                    "blocker": "MINIMUM_TWO_INDEPENDENT_VIEWS_REQUIRED",
                },
                {
                    "field": "holdout_median_residual_m",
                    "operator": "<=",
                    "threshold": _canonical_scalar(cls.maximum_holdout_median_m),
                    "blocker": "HOLDOUT_MEDIAN_RESIDUAL_EXCEEDED",
                },
                {
                    "field": "holdout_p95_residual_m",
                    "operator": "<=",
                    "threshold": _canonical_scalar(cls.maximum_holdout_p95_m),
                    "blocker": "HOLDOUT_P95_RESIDUAL_EXCEEDED",
                },
                {
                    "field": "holdout_absolute_bias_m",
                    "operator": "<=",
                    "threshold": _canonical_scalar(cls.maximum_absolute_bias_m),
                    "blocker": "HOLDOUT_ABSOLUTE_BIAS_EXCEEDED",
                },
                {
                    "field": "uncertainty_m",
                    "operator": "<=",
                    "threshold": _canonical_scalar(cls.maximum_uncertainty_m),
                    "blocker": "NEURAL_UNCERTAINTY_EXCEEDED",
                },
                {
                    "field": "observed_front",
                    "operator": "is-true",
                    "blocker": "UNOBSERVED_BACK_SURFACE",
                },
                {
                    "field": "free_space_conflict",
                    "operator": "is-false",
                    "blocker": "FREE_SPACE_CONFLICT",
                },
                {
                    "field": "occlusion_conflict",
                    "operator": "is-false",
                    "blocker": "OCCLUSION_CONFLICT",
                },
                {
                    "field": "depth_boundary_conflict",
                    "operator": "is-false",
                    "blocker": "DEPTH_BOUNDARY_CONFLICT",
                },
                {
                    "field": "measured_surface_conflict",
                    "operator": "is-false",
                    "blocker": "MEASURED_SURFACE_CONFLICT",
                },
            ),
        }
        return _identity_sha256(identity)

    def evaluate(self, candidate: NeuralSurfaceCandidate) -> NeuralSurfaceDecision:
        if not isinstance(candidate, NeuralSurfaceCandidate):
            raise TypeError("candidate must be a NeuralSurfaceCandidate")
        policy = NeuralSurfaceGate
        count = len(candidate.triangles)
        accepted = np.ones(count, dtype=bool)
        blockers: list[str] = []

        def reject(mask: np.ndarray, blocker: str) -> None:
            nonlocal accepted
            if np.any(mask):
                accepted &= ~mask
                blockers.append(blocker)

        reject(
            candidate.supporting_frame_count < policy.minimum_frames,
            "MINIMUM_THREE_FRAME_SUPPORT_REQUIRED",
        )
        reject(
            candidate.independent_view_count < policy.minimum_independent_views,
            "MINIMUM_TWO_INDEPENDENT_VIEWS_REQUIRED",
        )
        reject(
            candidate.holdout_median_residual_m > policy.maximum_holdout_median_m,
            "HOLDOUT_MEDIAN_RESIDUAL_EXCEEDED",
        )
        reject(
            candidate.holdout_p95_residual_m > policy.maximum_holdout_p95_m,
            "HOLDOUT_P95_RESIDUAL_EXCEEDED",
        )
        reject(
            candidate.holdout_absolute_bias_m > policy.maximum_absolute_bias_m,
            "HOLDOUT_ABSOLUTE_BIAS_EXCEEDED",
        )
        reject(
            candidate.uncertainty_m > policy.maximum_uncertainty_m,
            "NEURAL_UNCERTAINTY_EXCEEDED",
        )
        reject(~candidate.observed_front, "UNOBSERVED_BACK_SURFACE")
        reject(candidate.free_space_conflict, "FREE_SPACE_CONFLICT")
        reject(candidate.occlusion_conflict, "OCCLUSION_CONFLICT")
        reject(candidate.depth_boundary_conflict, "DEPTH_BOUNDARY_CONFLICT")
        reject(candidate.measured_surface_conflict, "MEASURED_SURFACE_CONFLICT")

        indices = np.flatnonzero(accepted).astype("<u4")
        provenance = np.full(
            len(indices), int(TriangleProvenance.NEURAL_MULTIVIEW), dtype="u1"
        )
        uncertainty = np.asarray(candidate.uncertainty_m[indices], dtype="<f4")
        metrics = {
            "source_window_sha256": candidate.source_window_sha256,
            "candidate_triangle_count": count,
            "accepted_triangle_count": len(indices),
            "rejected_triangle_count": count - len(indices),
            "near_object_size_holdout_error_mm": candidate.near_object_size_holdout_error_mm,
        }
        return NeuralSurfaceDecision(
            accepted_triangle_indices=indices,
            triangle_provenance=provenance,
            triangle_uncertainty_m=uncertainty,
            candidate_revision_id=candidate.recompute_candidate_revision_id(),
            gate_policy_revision_id=policy.policy_revision_id(),
            blockers=tuple(dict.fromkeys(blockers)),
            metrics=metrics,
        )


class HybridSurfaceAssembler:
    """Append only gated neural faces after measured faces and preserve evidence order."""

    @staticmethod
    def merge(
        measured: MeshTileRevision,
        neural: NeuralSurfaceCandidate,
        decision: NeuralSurfaceDecision,
    ) -> MeshTileRevision:
        if not isinstance(decision, NeuralSurfaceDecision):
            raise TypeError("decision must be a NeuralSurfaceDecision")
        indices = decision.validate_for(neural)
        measured_positions = np.asarray(measured.positions, dtype="<f4")
        measured_normals = np.asarray(measured.normals, dtype="<f4")
        measured_triangles = np.asarray(measured.triangles, dtype="<u4")
        measured_scales = np.asarray(measured.triangle_scale_m, dtype="<f4")
        measured_provenance = (
            np.full(len(measured_triangles), int(TriangleProvenance.MEASURED_LIDAR), dtype="u1")
            if measured.triangle_provenance is None
            else np.asarray(measured.triangle_provenance, dtype="u1")
        )
        measured_uncertainty = (
            measured_scales.copy()
            if measured.triangle_uncertainty_m is None
            else np.asarray(measured.triangle_uncertainty_m, dtype="<f4")
        )
        neural_triangles = np.asarray(neural.triangles[indices], dtype="<u4") + len(measured_positions)
        positions = np.concatenate((measured_positions, neural.positions.astype("<f4", copy=False)))
        normals = np.concatenate((measured_normals, neural.normals.astype("<f4", copy=False)))
        triangles = np.concatenate((measured_triangles, neural_triangles))
        scales = np.concatenate((measured_scales, neural.triangle_scale_m[indices].astype("<f4", copy=False)))
        provenance = np.concatenate((measured_provenance, decision.triangle_provenance.astype("u1", copy=False)))
        uncertainty = np.concatenate((measured_uncertainty, decision.triangle_uncertainty_m.astype("<f4", copy=False)))
        digest = hashlib.sha256()
        digest.update(json.dumps(measured.tile_id.as_tuple(), separators=(",", ":")).encode("ascii"))
        for values in (positions, normals, triangles, scales, provenance, uncertainty):
            digest.update(np.ascontiguousarray(values).tobytes())
        return MeshTileRevision(
            tile_id=measured.tile_id,
            revision_id=digest.hexdigest(),
            positions=positions,
            normals=normals,
            triangles=triangles,
            triangle_scale_m=scales,
            unresolved_surfel_count=measured.unresolved_surfel_count,
            triangle_provenance=provenance,
            triangle_uncertainty_m=uncertainty,
        )
