from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Dict, List, Mapping, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy import sparse
from scipy.spatial.transform import Rotation


_SPARSE_SOLVER_MIN_VARIABLE_KEYFRAMES = 64


def _rigid(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        matrix = np.eye(4)
        matrix[:3, 3] = array
    elif array.shape == (4, 4):
        matrix = np.array(array, copy=True)
    else:
        raise ValueError(f"{name} must be XYZ or a 4 by 4 transform")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must be a finite homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"{name} must be rigid")
    matrix.setflags(write=False)
    return matrix


def _relative(map_from_source: np.ndarray, map_from_target: np.ndarray) -> np.ndarray:
    return np.linalg.inv(map_from_source) @ map_from_target


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))


@dataclass(frozen=True)
class Keyframe:
    keyframe_id: str
    measured_pose_map_from_keyframe: np.ndarray
    calibration_id: str
    raw_asset_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measured_pose_map_from_keyframe",
            _rigid(self.measured_pose_map_from_keyframe, "measured_pose_map_from_keyframe"),
        )

    @property
    def measured_pose_xyz(self) -> Tuple[float, float, float]:
        return tuple(float(value) for value in self.measured_pose_map_from_keyframe[:3, 3])


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    translation_residual_m: float
    rotation_residual_deg: float
    measured_target_from_source: np.ndarray | None = None
    translation_sigma_m: float | None = None
    rotation_sigma_deg: float | None = None

    def __post_init__(self) -> None:
        if self.measured_target_from_source is not None:
            object.__setattr__(
                self,
                "measured_target_from_source",
                _rigid(self.measured_target_from_source, "measured_target_from_source"),
            )


@dataclass(frozen=True)
class PoseGraphSolution:
    poses_map_from_keyframe: Mapping[str, np.ndarray]
    pose_generation: str
    initial_cost: float
    final_cost: float
    accepted_loop_closures: int
    iterations: int


class PoseGraph:
    def __init__(self, loop_translation_limit_m: float, loop_rotation_limit_deg: float) -> None:
        if loop_translation_limit_m <= 0.0 or loop_rotation_limit_deg <= 0.0:
            raise ValueError("loop limits must be positive")
        self.loop_translation_limit_m = float(loop_translation_limit_m)
        self.loop_rotation_limit_deg = float(loop_rotation_limit_deg)
        self._keyframes: Dict[str, Keyframe] = {}
        self.edges: List[GraphEdge] = []

    def add_keyframe(self, keyframe_id: str, measured_pose_xyz, calibration_id: str, raw_asset_ref: str) -> None:
        if not keyframe_id or not calibration_id or not raw_asset_ref:
            raise ValueError("keyframe identity, calibration, and raw asset are required")
        candidate = Keyframe(keyframe_id, measured_pose_xyz, calibration_id, raw_asset_ref)
        existing = self._keyframes.get(keyframe_id)
        if existing is not None and (
            existing.calibration_id != candidate.calibration_id
            or existing.raw_asset_ref != candidate.raw_asset_ref
            or not np.array_equal(existing.measured_pose_map_from_keyframe, candidate.measured_pose_map_from_keyframe)
        ):
            raise ValueError("keyframe measurements are immutable")
        self._keyframes[keyframe_id] = candidate

    def keyframe(self, keyframe_id: str) -> Keyframe:
        return self._keyframes[keyframe_id]

    def add_lio_edge(self, source: str, target: str, translation_residual_m: float, rotation_residual_deg: float) -> None:
        self._require_nodes(source, target)
        self.edges.append(GraphEdge(source, target, "lio", translation_residual_m, rotation_residual_deg))

    def add_loop_edge(self, source: str, target: str, translation_residual_m: float, rotation_residual_deg: float) -> bool:
        self._require_nodes(source, target)
        if translation_residual_m > self.loop_translation_limit_m or rotation_residual_deg > self.loop_rotation_limit_deg:
            return False
        self.edges.append(GraphEdge(source, target, "verified_loop", translation_residual_m, rotation_residual_deg))
        return True

    def add_lio_constraint(
        self,
        source: str,
        target: str,
        measured_target_from_source,
        *,
        translation_sigma_m: float,
        rotation_sigma_deg: float,
    ) -> None:
        self._require_nodes(source, target)
        self._validate_sigma(translation_sigma_m, rotation_sigma_deg)
        measurement = _rigid(measured_target_from_source, "measured_target_from_source")
        self.edges.append(GraphEdge(
            source,
            target,
            "lio",
            0.0,
            0.0,
            measurement,
            float(translation_sigma_m),
            float(rotation_sigma_deg),
        ))

    def add_loop_constraint(
        self,
        source: str,
        target: str,
        measured_target_from_source,
        *,
        translation_sigma_m: float,
        rotation_sigma_deg: float,
    ) -> bool:
        self._require_nodes(source, target)
        self._validate_sigma(translation_sigma_m, rotation_sigma_deg)
        measurement = _rigid(measured_target_from_source, "measured_target_from_source")
        predicted = _relative(
            self._keyframes[source].measured_pose_map_from_keyframe,
            self._keyframes[target].measured_pose_map_from_keyframe,
        )
        correction = np.linalg.inv(measurement) @ predicted
        translation_residual = float(np.linalg.norm(correction[:3, 3]))
        rotation_residual = _rotation_angle_deg(correction[:3, :3])
        if translation_residual > self.loop_translation_limit_m or rotation_residual > self.loop_rotation_limit_deg:
            return False
        self.edges.append(GraphEdge(
            source,
            target,
            "verified_loop",
            translation_residual,
            rotation_residual,
            measurement,
            float(translation_sigma_m),
            float(rotation_sigma_deg),
        ))
        return True

    def optimize(self) -> PoseGraphSolution:
        if not self._keyframes:
            raise ValueError("pose graph has no keyframes")
        ids = list(self._keyframes)
        anchor = ids[0]
        variable_ids = ids[1:]
        constraints = sorted(
            (edge for edge in self.edges if edge.measured_target_from_source is not None),
            key=lambda edge: (
                edge.source,
                edge.target,
                edge.kind,
                tuple(edge.measured_target_from_source.reshape(-1).tolist()),
                edge.translation_sigma_m,
                edge.rotation_sigma_deg,
            ),
        )
        initial = (
            np.concatenate([
                self._pose_parameters(self._keyframes[key].measured_pose_map_from_keyframe)
                for key in variable_ids
            ])
            if variable_ids
            else np.empty(0, dtype=np.float64)
        )

        def poses(parameters: np.ndarray) -> dict[str, np.ndarray]:
            values = {anchor: np.array(self._keyframes[anchor].measured_pose_map_from_keyframe, copy=True)}
            for index, key in enumerate(variable_ids):
                values[key] = self._parameters_pose(parameters[index * 6 : (index + 1) * 6])
            return values

        def edge_residual(edge: GraphEdge, source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
            predicted = _relative(source_pose, target_pose)
            error = np.linalg.inv(edge.measured_target_from_source) @ predicted
            rotation_sigma_rad = math.radians(edge.rotation_sigma_deg)
            return np.concatenate((
                error[:3, 3] / edge.translation_sigma_m,
                Rotation.from_matrix(error[:3, :3]).as_rotvec() / rotation_sigma_rad,
            ))

        def residual(parameters: np.ndarray) -> np.ndarray:
            current = poses(parameters)
            if not constraints:
                return np.zeros(1, dtype=np.float64)
            values = np.empty(6 * len(constraints), dtype=np.float64)
            for index, edge in enumerate(constraints):
                values[index * 6 : (index + 1) * 6] = edge_residual(
                    edge,
                    current[edge.source],
                    current[edge.target],
                )
            return values

        variable_indices = {key: index for index, key in enumerate(variable_ids)}

        def dense_jacobian(parameters: np.ndarray) -> np.ndarray:
            """Accurate local finite differences for compact pose graphs."""

            current = poses(parameters)
            matrix = np.zeros((6 * len(constraints), len(parameters)), dtype=np.float64)
            relative_step = math.sqrt(np.finfo(np.float64).eps)
            for constraint_index, edge in enumerate(constraints):
                row = slice(constraint_index * 6, (constraint_index + 1) * 6)
                source_pose = current[edge.source]
                target_pose = current[edge.target]
                baseline = edge_residual(edge, source_pose, target_pose)
                for key in (edge.source, edge.target):
                    variable_index = variable_indices.get(key)
                    if variable_index is None:
                        continue
                    column_start = variable_index * 6
                    block = parameters[column_start : column_start + 6]
                    for offset in range(6):
                        value = float(block[offset])
                        step = relative_step * (-1.0 if value < 0.0 else 1.0) * max(1.0, abs(value))
                        perturbed_block = np.array(block, copy=True)
                        perturbed_block[offset] += step
                        perturbed_pose = self._parameters_pose(perturbed_block)
                        candidate = edge_residual(
                            edge,
                            perturbed_pose if key == edge.source else source_pose,
                            perturbed_pose if key == edge.target else target_pose,
                        )
                        matrix[row, column_start + offset] = (candidate - baseline) / step
            return matrix

        def jacobian_sparsity() -> sparse.csr_matrix:
            """Describe the two keyframe blocks touched by each SE(3) edge."""

            rows: list[int] = []
            columns: list[int] = []
            for constraint_index, edge in enumerate(constraints):
                row_start = constraint_index * 6
                for key in (edge.source, edge.target):
                    variable_index = variable_indices.get(key)
                    if variable_index is None:
                        continue
                    column_start = variable_index * 6
                    rows.extend(row_start + component for component in range(6) for _ in range(6))
                    columns.extend(column_start + component for _ in range(6) for component in range(6))
            return sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, columns)),
                shape=(6 * len(constraints), len(initial)),
            )

        initial_residual = residual(initial)
        if constraints and len(variable_ids):
            use_sparse_solver = len(variable_ids) >= _SPARSE_SOLVER_MIN_VARIABLE_KEYFRAMES
            optimization_kwargs = {
                "loss": "huber",
                "f_scale": 1.0,
                "max_nfev": 500,
                "xtol": 1e-12,
                "ftol": 1e-12,
                "gtol": 1e-12,
            }
            if use_sparse_solver:
                # A dense trust-region solve grows quadratically with the
                # number of keyframes. Sparse coloring differentiates only
                # non-overlapping pose blocks per residual evaluation.
                optimization_kwargs["jac"] = "2-point"
                optimization_kwargs["jac_sparsity"] = jacobian_sparsity()
                optimization_kwargs["tr_solver"] = "lsmr"
            else:
                optimization_kwargs["jac"] = dense_jacobian
            optimized = least_squares(
                residual,
                initial,
                **optimization_kwargs,
            )
            if not optimized.success or not np.all(np.isfinite(optimized.x)):
                raise RuntimeError(f"pose graph optimization failed: {optimized.message}")
            final_parameters = optimized.x
            iterations = int(optimized.nfev)
        else:
            final_parameters = initial
            iterations = 0
        final_residual = residual(final_parameters)
        final_poses = poses(final_parameters)
        readonly = {}
        for key, value in final_poses.items():
            value.setflags(write=False)
            readonly[key] = value
        generation_payload = {
            "poses": {key: readonly[key].reshape(-1).tolist() for key in sorted(readonly)},
            "constraints": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind,
                    "measurement": edge.measured_target_from_source.reshape(-1).tolist(),
                    "translation_sigma_m": edge.translation_sigma_m,
                    "rotation_sigma_deg": edge.rotation_sigma_deg,
                }
                for edge in constraints
            ],
        }
        generation = hashlib.sha256(
            json.dumps(generation_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()
        return PoseGraphSolution(
            MappingProxyType(readonly),
            generation,
            float(np.dot(initial_residual, initial_residual)),
            float(np.dot(final_residual, final_residual)),
            sum(edge.kind == "verified_loop" for edge in constraints),
            iterations,
        )

    def _require_nodes(self, source: str, target: str) -> None:
        if source not in self._keyframes or target not in self._keyframes:
            raise KeyError("edge endpoint is not a keyframe")
        if source == target:
            raise ValueError("self edges are not allowed")

    @staticmethod
    def _validate_sigma(translation_sigma_m: float, rotation_sigma_deg: float) -> None:
        if not all(math.isfinite(value) and value > 0.0 for value in (translation_sigma_m, rotation_sigma_deg)):
            raise ValueError("constraint sigmas must be finite and positive")

    @staticmethod
    def _pose_parameters(pose: np.ndarray) -> np.ndarray:
        return np.concatenate((Rotation.from_matrix(np.array(pose[:3, :3], copy=True)).as_rotvec(), pose[:3, 3]))

    @staticmethod
    def _parameters_pose(parameters: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        pose[:3, 3] = parameters[3:6]
        return pose
