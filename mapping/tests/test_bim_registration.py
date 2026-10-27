import numpy as np
import pytest

from mapping.pc.go2_mapping_bim import alignment_manifest_sha256, evaluate_registration
from mapping.pc.go2_mapping_bim.register import RegistrationDecision


HASH = "a" * 64


def test_bim_alignment_requires_excluded_structural_holdout_and_preserves_lidar_authority():
    result = evaluate_registration(
        ("north-wall", "east-wall", "column-grid"),
        support_m=15.0,
        median_residual_m=0.02,
        p95_residual_m=0.05,
        translation_sigma_m=0.02,
        rotation_sigma_deg=0.4,
        design_model_sha256=HASH,
        holdout_groups=("south-wall",),
        holdout_p95_residual_m=0.07,
        transform_map_from_design=np.eye(4),
    )

    assert result.decision is RegistrationDecision.ACCEPTED
    assert result.quality_level == "BIM_REGISTERED_VALIDATED"
    assert result.alignment_manifest["geometry_authority"] == "LIDAR_MEASURED"
    assert result.alignment_manifest["rigid_se3_only"] is True
    assert len(alignment_manifest_sha256(result.alignment_manifest)) == 64


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"holdout_groups": (), "holdout_p95_residual_m": 0.07}, "excluded holdout"),
        ({"holdout_groups": ("north-wall",), "holdout_p95_residual_m": 0.07}, "overlap"),
        ({"holdout_groups": ("south-wall",), "holdout_p95_residual_m": 0.11}, "holdout residual"),
        ({"holdout_groups": ("south-wall",), "holdout_p95_residual_m": 0.07, "rigid_se3_only": False}, "non-rigid"),
    ],
)
def test_bim_alignment_fail_closed(kwargs, reason):
    result = evaluate_registration(
        ("north-wall", "east-wall", "column-grid"),
        support_m=15.0,
        median_residual_m=0.02,
        p95_residual_m=0.05,
        translation_sigma_m=0.02,
        rotation_sigma_deg=0.4,
        design_model_sha256=HASH,
        **kwargs,
    )
    assert result.decision is RegistrationDecision.UNREGISTERED
    assert any(reason.lower() in item.lower() for item in result.reasons)


def test_bim_alignment_rejects_non_rigid_transform_and_bad_design_hash():
    result = evaluate_registration(
        ("north-wall", "east-wall", "column-grid"),
        support_m=15.0,
        median_residual_m=0.02,
        p95_residual_m=0.05,
        translation_sigma_m=0.02,
        rotation_sigma_deg=0.4,
        design_model_sha256="bad",
        holdout_groups=("south-wall",),
        holdout_p95_residual_m=0.07,
        transform_map_from_design=np.diag([2.0, 1.0, 1.0, 1.0]),
    )

    assert result.decision is RegistrationDecision.UNREGISTERED
    assert "SHA-256" in " ".join(result.reasons)
    assert "rigid SE(3)" in " ".join(result.reasons)
