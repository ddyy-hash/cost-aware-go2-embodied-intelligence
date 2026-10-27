import numpy as np
import pytest

from mapping.pc.go2_mapping_rgbd import ConsensusAnchor
from mapping.pc.go2_mapping_rgbd.monocular_validation import (
    AnchorPartition,
    fit_metric_depth,
)


def _partition(*, scale=1.25, bias=0.02, holdout_offset=0.0, constant_model=False):
    fit = []
    holdout = []
    for index in range(240):
        model = 1.0 + index / 240.0 if not constant_model else 1.0
        metric_inverse = scale * model + bias
        fit.append(
            ConsensusAnchor(
                pixel_x=index % 80,
                pixel_y=index // 80,
                lidar_point_index=index,
                model_inverse_depth=model,
                metric_depth_m=1.0 / metric_inverse,
                point_camera_xyz=(0.0, 0.0, 1.0),
                frame_sha256="b" * 64,
                accepted_region_id=f"cell-{index % 24}",
            )
        )
    for index in range(64):
        model = 1.0 + (index + 3) / 64.0 if not constant_model else 1.0
        metric_inverse = scale * model + bias
        holdout.append(
            ConsensusAnchor(
                pixel_x=index % 80,
                pixel_y=20 + index // 80,
                lidar_point_index=1000 + index,
                model_inverse_depth=model,
                metric_depth_m=1.0 / metric_inverse + holdout_offset,
                point_camera_xyz=(0.0, 0.0, 1.0),
                frame_sha256="b" * 64,
                accepted_region_id=f"cell-{index % 24}",
            )
        )
    return AnchorPartition(tuple(fit), tuple(holdout), tuple(range(24)), (0, 1, 2, 3), 7, True, ())


def test_metric_fit_is_robust_positive_and_validates_untouched_holdout():
    partition = _partition()
    noisy = list(partition.fit)
    noisy[3] = ConsensusAnchor(**{**noisy[3].__dict__, "metric_depth_m": 4.0})
    noisy_partition = AnchorPartition(tuple(noisy), partition.holdout, partition.occupied_cells, partition.quadrants_in_fit, partition.seed64, True, ())

    fit = fit_metric_depth(noisy_partition, model_revision_id="a" * 64)

    assert fit.accepted is True
    assert fit.scale == pytest.approx(1.25, rel=1e-5)
    assert fit.bias == pytest.approx(0.02, rel=1e-5)
    assert fit.iterations <= 50
    assert 3 not in fit.inlier_ids
    assert fit.holdout_median_absolute_residual_m <= 0.025
    assert fit.holdout_p95_absolute_residual_m <= 0.060
    assert fit.holdout_absolute_bias_m <= 0.015
    assert len(fit.revision_id) == 64


def test_metric_fit_rejects_negative_scale_singular_model_and_holdout_failure():
    negative = fit_metric_depth(_partition(scale=-1.0, bias=3.0), model_revision_id="a" * 64)
    assert negative.accepted is False
    assert "NON_POSITIVE_SCALE" in negative.blockers

    singular = fit_metric_depth(_partition(constant_model=True), model_revision_id="a" * 64)
    assert singular.accepted is False
    assert "SINGULAR_METRIC_FIT" in singular.blockers

    holdout = fit_metric_depth(_partition(holdout_offset=0.2), model_revision_id="a" * 64)
    assert holdout.accepted is False
    assert "HOLDOUT_RESIDUAL_GATE_FAILED" in holdout.blockers


def test_metric_fit_rejects_unusable_partition_and_non_finite_values():
    empty = fit_metric_depth(AnchorPartition((), (), (), (), 0, False, ("BLOCKED",)), model_revision_id="a" * 64)
    assert empty.accepted is False
    assert "BLOCKED" in empty.blockers

    broken = list(_partition().fit)
    broken[0] = ConsensusAnchor(**{**broken[0].__dict__, "model_inverse_depth": np.nan})
    result = fit_metric_depth(
        AnchorPartition(tuple(broken), _partition().holdout, tuple(range(24)), (0, 1, 2, 3), 7, True, ()),
        model_revision_id="a" * 64,
    )
    assert result.accepted is False
    assert "NON_FINITE_ANCHOR_VALUES" in result.blockers
