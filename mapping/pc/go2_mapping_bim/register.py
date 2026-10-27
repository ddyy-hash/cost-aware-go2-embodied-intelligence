from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Sequence

import numpy as np


class RegistrationDecision(str, Enum):
    ACCEPTED = "accepted"
    UNREGISTERED = "unregistered"


@dataclass(frozen=True)
class RegistrationResult:
    decision: RegistrationDecision
    geometry_authority: str
    reasons: tuple
    quality_level: str = "UNREGISTERED"
    alignment_manifest: dict | None = None


def evaluate_registration(
    groups: Sequence[str],
    support_m: float,
    median_residual_m: float,
    p95_residual_m: float,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
    *,
    design_model_sha256: str | None = None,
    holdout_groups: Sequence[str] = (),
    holdout_p95_residual_m: float | None = None,
    transform_map_from_design: Sequence[float] | np.ndarray | None = None,
    rigid_se3_only: bool = True,
) -> RegistrationResult:
    reasons = []
    if len(set(groups)) < 3:
        reasons.append("fewer than three independent structural groups")
    if support_m < 12:
        reasons.append("insufficient structural support")
    if median_residual_m > 0.05 or p95_residual_m > 0.10:
        reasons.append("residual threshold exceeded")
    if translation_sigma_m > 0.05 or rotation_sigma_deg > 1.0:
        reasons.append("transform uncertainty threshold exceeded")
    if design_model_sha256 is not None and not _is_sha256(design_model_sha256):
        reasons.append("design model SHA-256 is invalid")
    train_groups = tuple(dict.fromkeys(str(group) for group in groups))
    excluded_groups = tuple(dict.fromkeys(str(group) for group in holdout_groups))
    if holdout_groups and set(train_groups) & set(excluded_groups):
        reasons.append("holdout structural groups overlap training groups")
    if design_model_sha256 is not None and not excluded_groups:
        reasons.append("excluded holdout structural group is required")
    if holdout_groups and holdout_p95_residual_m is None:
        reasons.append("holdout residual evidence is required")
    if holdout_p95_residual_m is not None and holdout_p95_residual_m > 0.10:
        reasons.append("holdout residual threshold exceeded")
    if not rigid_se3_only:
        reasons.append("non-rigid BIM deformation is forbidden")
    transform = _rigid_transform(transform_map_from_design)
    if transform_map_from_design is not None and transform is None:
        reasons.append("design alignment transform is not a rigid SE(3)")
    manifest = None
    if not reasons and design_model_sha256 is not None and excluded_groups:
        manifest = {
            "schema_version": 1,
            "state": "BIM_REGISTERED_VALIDATED",
            "geometry_authority": "LIDAR_MEASURED",
            "design_model_sha256": design_model_sha256.lower(),
            "training_structural_groups": list(train_groups),
            "holdout_structural_groups": list(excluded_groups),
            "median_residual_m": float(median_residual_m),
            "p95_residual_m": float(p95_residual_m),
            "holdout_p95_residual_m": None if holdout_p95_residual_m is None else float(holdout_p95_residual_m),
            "translation_sigma_m": float(translation_sigma_m),
            "rotation_sigma_deg": float(rotation_sigma_deg),
            "rigid_se3_only": True,
            "transform_map_from_design": None if transform is None else transform.reshape(-1).tolist(),
        }
    return RegistrationResult(
        RegistrationDecision.UNREGISTERED if reasons else RegistrationDecision.ACCEPTED,
        "as_built_map",
        tuple(reasons),
        "BIM_REGISTERED_VALIDATED" if manifest is not None else ("ACCEPTED_RIGID_ALIGNMENT" if not reasons else "UNREGISTERED"),
        manifest,
    )


def alignment_manifest_sha256(manifest: dict) -> str:
    """Content address a validated BIM alignment decision."""
    if manifest.get("state") != "BIM_REGISTERED_VALIDATED" or manifest.get("geometry_authority") != "LIDAR_MEASURED":
        raise ValueError("only a validated measured BIM alignment can be hashed")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _rigid_transform(value) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        return None
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        return None
    return matrix
