"""Deterministic, non-destructive frame-local LiDAR filtering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from mapping.pc.go2_mapping_quality.reconstruction_config import FilteringGates

from .observations import FilteredFrame, LidarObservationFrame, RejectedObservation


class FrameFilter:
    def __init__(
        self,
        gates: FilteringGates,
        self_mask: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.gates = gates
        self.self_mask = self_mask

    def classify(self, frame: LidarObservationFrame) -> FilteredFrame:
        points = frame.points_xyz
        rejected: list[RejectedObservation] = []
        eligible: list[int] = []
        for index, point in enumerate(points):
            if not np.all(np.isfinite(point)):
                rejected.append(RejectedObservation(index, "NONFINITE_XYZ"))
                continue
            distance = float(np.linalg.norm(point))
            if not self.gates.accepts_range(range_m=distance):
                rejected.append(RejectedObservation(index, "RANGE_OUT_OF_BOUNDS", f"range_m={distance:g}"))
                continue
            eligible.append(index)

        if self.self_mask is not None and eligible:
            selected = np.asarray(self.self_mask(points[eligible]), dtype=bool)
            if selected.shape != (len(eligible),):
                raise ValueError("self_mask must return one boolean per candidate point")
            masked = [index for index, is_masked in zip(eligible, selected) if is_masked]
            rejected.extend(RejectedObservation(index, "SELF_MASK") for index in masked)
            eligible = [index for index, is_masked in zip(eligible, selected) if not is_masked]

        if not eligible:
            return self._result(frame, (), rejected)

        candidate_points = points[eligible]
        tree = cKDTree(candidate_points)
        neighbors = tree.query_ball_point(candidate_points, r=self.gates.radius_m)
        radius_accepted: list[int] = []
        for local_index, neighbor_indices in enumerate(neighbors):
            support = len(neighbor_indices) - 1
            source_index = eligible[local_index]
            if not self.gates.has_radius_support(neighbor_count=support):
                rejected.append(RejectedObservation(source_index, "RADIUS_OUTLIER", f"neighbor_count={support}"))
            else:
                radius_accepted.append(source_index)

        if not radius_accepted:
            return self._result(frame, (), rejected)

        if len(radius_accepted) < self.gates.statistical_neighbor_count + 1:
            for index in radius_accepted:
                rejected.append(
                    RejectedObservation(index, "STATISTICAL_SUPPORT_INSUFFICIENT", f"source_count={len(radius_accepted)}")
                )
            return self._result(frame, (), rejected)

        statistical_points = points[radius_accepted]
        statistical_tree = cKDTree(statistical_points)
        distances, _ = statistical_tree.query(
            statistical_points,
            k=self.gates.statistical_neighbor_count + 1,
            workers=1,
        )
        neighbor_distances = distances[:, 1:]
        local_means = np.mean(neighbor_distances, axis=1)
        center = float(np.median(local_means))
        spread = float(1.4826 * np.median(np.abs(local_means - center)))
        if spread <= 1e-12:
            spread = float(np.std(local_means))
        accepted: list[int] = []
        for source_index, distance in zip(radius_accepted, local_means):
            if self.gates.accepts_statistical_distance(distance=float(distance), mean=center, stddev=spread):
                accepted.append(source_index)
            else:
                rejected.append(RejectedObservation(source_index, "STATISTICAL_OUTLIER", f"mean_neighbor_distance={distance:g}"))
        return self._result(frame, accepted, rejected)

    @staticmethod
    def _result(
        frame: LidarObservationFrame,
        accepted: tuple[int, ...] | list[int],
        rejected: list[RejectedObservation],
    ) -> FilteredFrame:
        return FilteredFrame(
            source_count=len(frame.points_xyz),
            accepted_indices=np.asarray(sorted(accepted), dtype=np.int64),
            rejected=tuple(sorted(rejected, key=lambda item: (item.index, item.reason))),
            artifact_sha256=frame.artifact_sha256,
            frame_id=frame.frame_id,
            record_timestamp_ns=frame.record_timestamp_ns,
        )
