import hashlib
import json
from pathlib import Path

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_json_bytes, load_reconstruction_config
from mapping.pc.go2_mapping_rgbd import ConsensusAnchor
from mapping.pc.go2_mapping_rgbd.monocular_validation import (
    AnchorPartition,
    AnchorSeedIdentity,
    bootstrap_stability,
    compute_metric_depth_uncertainty,
    fit_metric_depth,
)


def _partition():
    fit = []
    holdout = []
    for index in range(240):
        model = 1.0 + index / 240.0
        inverse = 1.25 * model + 0.02
        fit.append(ConsensusAnchor(index % 80, index // 80, index, model, 1.0 / inverse, (0.0, 0.0, 1.0), "b" * 64, f"cell-{index % 24}"))
    for index in range(64):
        model = 1.0 + (index + 3) / 64.0
        inverse = 1.25 * model + 0.02
        holdout.append(ConsensusAnchor(index % 80, 20 + index // 80, 1000 + index, model, 1.0 / inverse, (0.0, 0.0, 1.0), "b" * 64, f"cell-{index % 24}"))
    return AnchorPartition(tuple(fit), tuple(holdout), tuple(range(24)), (0, 1, 2, 3), 7, True, ())


def _seed():
    return AnchorSeedIdentity("a" * 64, "b" * 64, load_reconstruction_config().appearance_sha256, 80, 60)


def test_bootstrap_is_deterministic_cell_preserving_and_stable():
    partition = _partition()
    fit = fit_metric_depth(partition, model_revision_id="a" * 64)
    result = bootstrap_stability(partition, fit, model_revision_id="a" * 64, seed_identity=_seed())
    repeated = bootstrap_stability(partition, fit, model_revision_id="a" * 64, seed_identity=_seed())

    payload = {
        "appearance_config_sha256": load_reconstruction_config().appearance_sha256,
        "domain": "bootstrap-v1",
        "frame_sha256": ["b" * 64],
        "model_revision_id": "a" * 64,
    }
    expected_seed = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "little")
    assert result.seed64 == expected_seed
    assert result.replicates == 1000
    assert result.accepted is True
    assert result.scale_stddev_fraction <= 0.02
    assert np.array_equal(result.parameter_samples, repeated.parameter_samples)
    assert result.parameter_samples.flags.writeable is False
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "depth_anything" / "pcg64-golden.json").read_text(encoding="utf-8")
    )
    assert result.seed64 == fixture["bootstrap_seed64"]
    assert [list(sample) for sample in result.first_cell_samples] == fixture["bootstrap_first_cell_samples"]


def test_uncertainty_counts_components_once_and_requires_accepted_boundary():
    partition = _partition()
    fit = fit_metric_depth(partition, model_revision_id="a" * 64)
    bootstrap = bootstrap_stability(partition, fit, model_revision_id="a" * 64, seed_identity=_seed())
    model = np.full((2, 2), 1.2, dtype=np.float64)
    physical = np.asarray([[np.nan, 1.0 / (1.25 * 1.2 + 0.02)], [np.nan, np.nan]])
    distance = np.asarray([[0.10, 0.10], [0.20, 0.10]], dtype=np.float64)
    boundary = np.asarray([[True, True], [True, False]])

    result = compute_metric_depth_uncertainty(
        model,
        fit,
        bootstrap,
        physical_depth_m=physical,
        anchor_distance_m=distance,
        boundary_accepted=boundary,
    )

    assert result.metric_depth_m.shape == (2, 2)
    assert result.sigma_physical_m[0, 0] == 0.0
    assert result.valid[0, 0] is np.bool_ or bool(result.valid[0, 0])
    assert result.valid[1, 1] is False or bool(result.valid[1, 1]) is False
    assert result.sigma_total_m[0, 0] <= 0.03
    assert "BOUNDARY_SUPPORT_REQUIRED" in result.blockers
