"""Bounded calibration-capture progress for live operator feedback."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import math
from threading import RLock

import cv2
import numpy as np

from .target import BoardDetection, TargetDetector, TargetSpecError


CAPTURE_SLOTS = ("estimation_1", "estimation_2", "estimation_3", "holdout")


@dataclass
class _Slot:
    accepted_frames: int = 0
    rejected_frames: int = 0
    board_instances: int = 0
    last_timestamp_ns: int | None = None
    normals: list[np.ndarray] = field(default_factory=list)


class CalibrationCaptureState:
    """Track detection progress without treating preview JPEGs as evidence."""

    def __init__(self, detector: TargetDetector, expected_size: tuple[int, int]) -> None:
        self.detector = detector
        self.expected_size = expected_size
        self._slots = {name: _Slot() for name in CAPTURE_SLOTS}
        self._active_slot = CAPTURE_SLOTS[0]
        self._last_result: dict | None = None
        self._lock = RLock()

    def select_slot(self, slot: str) -> None:
        if slot not in self._slots:
            raise ValueError(f"unknown calibration capture slot: {slot}")
        with self._lock:
            self._active_slot = slot

    def ingest_camera_payload(self, payload: dict) -> dict:
        color = payload.get("color")
        if not isinstance(color, dict):
            raise TargetSpecError("camera payload has no color frame")
        width = color.get("width")
        height = color.get("height")
        if (width, height) != self.expected_size:
            raise TargetSpecError(
                f"camera frame size {(width, height)} does not match the hashed CameraInfo {self.expected_size}"
            )
        try:
            encoded = base64.b64decode(color["jpeg_b64"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise TargetSpecError("camera payload JPEG is malformed") from exc
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise TargetSpecError("camera JPEG dimensions do not match its metadata")
        return self.ingest_image(image, int(payload.get("timestamp_ns", 0)))

    def ingest_image(self, image: np.ndarray, timestamp_ns: int) -> dict:
        detections = self.detector.detect_all(image)
        accepted = [value for value in detections if value.accepted]
        best = detections[0] if detections else self.detector.detect(image)
        with self._lock:
            slot = self._slots[self._active_slot]
            slot.last_timestamp_ns = timestamp_ns
            if accepted:
                slot.accepted_frames += 1
                slot.board_instances += len(accepted)
                normal = self._normal(accepted[0])
                if normal is not None:
                    slot.normals.append(normal)
                    slot.normals = slot.normals[-200:]
            else:
                slot.rejected_frames += 1
            self._last_result = {
                "slot": self._active_slot,
                "timestamp_ns": timestamp_ns,
                "accepted": bool(accepted),
                "board_instances": len(accepted),
                "decoded_fraction": best.decoded_fraction,
                "charuco_fraction": best.charuco_fraction,
                "reprojection_p95_px": best.reprojection_p95_px,
                "blocking_reasons": list(best.blocking_reasons),
            }
            return dict(self._last_result)

    def snapshot(self) -> dict:
        with self._lock:
            all_normals = [normal for slot in self._slots.values() for normal in slot.normals]
            orientation_count = self._independent_orientation_count(all_normals)
            blockers = []
            if any(slot.accepted_frames == 0 for slot in self._slots.values()):
                blockers.append("FOUR_CAPTURE_SLOTS_INCOMPLETE")
            if orientation_count < 3:
                blockers.append("BOARD_NORMAL_DIVERSITY_BELOW_THREE")
            blockers.extend(["CLOCK_FIT_PENDING", "ROBOT_MOTION_EXCITATION_PENDING", "HIGH_FIDELITY_EVIDENCE_PENDING"])
            return {
                "active_slot": self._active_slot,
                "slots": {
                    name: {
                        "accepted_frames": slot.accepted_frames,
                        "rejected_frames": slot.rejected_frames,
                        "board_instances": slot.board_instances,
                        "last_timestamp_ns": slot.last_timestamp_ns,
                    }
                    for name, slot in self._slots.items()
                },
                "board_pose_diversity": {
                    "independent_normal_orientations": orientation_count,
                    "required": 3,
                    "minimum_pairwise_separation_deg": 15.0,
                },
                "clock_fit_state": "PENDING",
                "motion_excitation_state": "PENDING",
                "last_detection": None if self._last_result is None else dict(self._last_result),
                "blockers": blockers,
                "preview_authoritative": False,
            }

    @staticmethod
    def _normal(detection: BoardDetection) -> np.ndarray | None:
        if detection.pose is None:
            return None
        rotation, _ = cv2.Rodrigues(np.asarray(detection.pose.rotation_vector, dtype=np.float64))
        normal = rotation @ np.asarray([0.0, 0.0, 1.0])
        return normal / max(float(np.linalg.norm(normal)), 1e-12)

    @staticmethod
    def _independent_orientation_count(normals: list[np.ndarray]) -> int:
        selected: list[np.ndarray] = []
        for normal in normals:
            if all(
                math.degrees(math.acos(float(np.clip(abs(np.dot(normal, previous)), -1.0, 1.0)))) >= 15.0
                for previous in selected
            ):
                selected.append(normal)
        return len(selected)
