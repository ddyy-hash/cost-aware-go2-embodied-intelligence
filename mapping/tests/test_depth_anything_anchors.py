import hashlib
import json
from pathlib import Path

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_json_bytes
from mapping.pc.go2_mapping_rgbd import ConsensusAnchor
from mapping.pc.go2_mapping_rgbd.monocular_validation import (
    AnchorSeedIdentity,
    stratified_holdout,
)


def _anchors(cells=48, per_cell=6):
    values = []
    for cell in range(cells):
        row, column = divmod(cell, 8)
        for offset in range(per_cell):
            values.append(
                ConsensusAnchor(
                    pixel_x=column * 10 + 1 + offset,
                    pixel_y=row * 10 + 1,
                    lidar_point_index=cell * per_cell + offset,
                    model_inverse_depth=1.0 + offset * 0.01,
                    metric_depth_m=1.0,
                    point_camera_xyz=(0.0, 0.0, 1.0),
                    frame_sha256="b" * 64,
                    accepted_region_id=f"tile-{cell}",
                )
            )
    return tuple(values)


def _identity():
    return AnchorSeedIdentity(
        model_revision_id="a" * 64,
        frame_sha256="b" * 64,
        appearance_config_sha256="c" * 64,
        image_width=80,
        image_height=60,
    )


def test_stratified_holdout_uses_exact_seed_allocation_and_disjoint_sets():
    identity = _identity()
    partition = stratified_holdout(_anchors(), seed_identity=identity)

    expected_payload = {
        "appearance_config_sha256": "c" * 64,
        "domain": "holdout-v1",
        "frame_sha256": "b" * 64,
        "model_revision_id": "a" * 64,
    }
    digest = hashlib.sha256(canonical_json_bytes(expected_payload)).digest()
    assert partition.seed64 == int.from_bytes(digest[:8], "little", signed=False)
    assert partition.accepted is True
    assert partition.blockers == ()
    assert len(partition.holdout) == 64
    assert len(partition.fit) == 224
    assert partition.occupied_cells == tuple(range(48))
    fit_ids = {anchor.lidar_point_index for anchor in partition.fit}
    holdout_ids = {anchor.lidar_point_index for anchor in partition.holdout}
    assert fit_ids.isdisjoint(holdout_ids)
    assert fit_ids | holdout_ids == set(range(288))
    counts = [sum(anchor.accepted_region_id == f"tile-{cell}" for anchor in partition.holdout) for cell in range(48)]
    assert counts == [2] * 16 + [1] * 32
    assert partition.quadrants_in_fit == (0, 1, 2, 3)


def test_stratified_holdout_is_repeatable_and_rejects_sparse_coverage():
    first = stratified_holdout(_anchors(), seed_identity=_identity())
    second = stratified_holdout(tuple(reversed(_anchors())), seed_identity=_identity())
    assert tuple(anchor.lidar_point_index for anchor in first.holdout) == tuple(
        anchor.lidar_point_index for anchor in second.holdout
    )

    sparse = stratified_holdout(_anchors(cells=10, per_cell=30), seed_identity=_identity())
    assert sparse.accepted is False
    assert "INSUFFICIENT_ANCHOR_GRID_COVERAGE" in sparse.blockers

    too_few = stratified_holdout(_anchors(cells=48, per_cell=5)[:240], seed_identity=_identity())
    assert too_few.accepted is False
    assert "INSUFFICIENT_CONSENSUS_ANCHORS" in too_few.blockers


def test_pcg64_holdout_matches_checked_in_golden_vector():
    fixture = Path(__file__).parent / "fixtures" / "depth_anything" / "pcg64-golden.json"
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    partition = stratified_holdout(_anchors(), seed_identity=_identity())

    assert partition.seed64 == expected["holdout_seed64"]
    assert [anchor.lidar_point_index for anchor in partition.holdout] == expected["holdout_ids"]
