import pytest

from mapping.pc.go2_mapping_appearance.tile_identity import (
    ArtifactRef,
    PrimitiveRecord,
    build_lod_record,
    tile_appearance_revision,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3"))


def _revision(**changes):
    values = {
        "tile_revision_id": HASHES[0],
        "geometry_revision_id": HASHES[1],
        "source_generation_id": HASHES[2],
        "calibration_id": "calibration-1",
        "calibration_sha256": HASHES[3],
        "physical_depth_validation_revision": HASHES[4],
        "pose_revision_id": HASHES[5],
        "model_revision_id": None,
        "assistance_config_sha256": None,
        "source_frame_sha256": (HASHES[7], HASHES[6]),
        "appearance_config_sha256": HASHES[8],
        "charting": {"implementation": "xatlas", "version": "0.0.11", "options": {"max_cost": 2.0}},
        "atlas_density_policy": {"minimum_mm_per_texel": 3, "maximum_mm_per_texel": 5},
        "padding_policy": {"pixels": 8, "mip_levels": 4},
        "baker_algorithm_version": "tile-atlas-baker-v1",
        "rgbd_augmented_revision_id": None,
        "contains_augmented_primitives": False,
        "measured_surface_coverage_ppm": 800000,
        "minimum_independent_view_count": 2,
    }
    values.update(changes)
    return tile_appearance_revision(**values)


def _lod(revision, *, artifact_hash=HASHES[0], atlas_index=0, source_layer="LIDAR_MEASURED"):
    artifacts = (
        ArtifactRef("render-vertices.bin", artifact_hash, 84),
        ArtifactRef("render-indices.bin", HASHES[1], 96),
        ArtifactRef("render-uvs.bin", HASHES[2], 84),
        ArtifactRef("atlas-0.png", HASHES[3], 128),
    )
    primitive = PrimitiveRecord(
        primitive_id=0,
        material="VALIDATED_UV_ATLAS",
        atlas_index=atlas_index,
        index_offset=0,
        index_count=3,
        source_layer=source_layer,
        source_triangle_ids=(0,),
    )
    return build_lod_record(
        tile_revision_id=revision.revision_core["tile_revision_id"],
        appearance_revision_id=revision.appearance_revision_id,
        policy={"name": "near", "mm_per_texel": 3, "maximum_distance_m": 2.0},
        vertex_count=3,
        triangle_count=1,
        atlas_count=1,
        artifacts=artifacts,
        primitives=(primitive,),
        rgbd_augmented_revision_id=revision.revision_core["rgbd_augmented_revision_id"],
    )


def test_revision_is_evidence_only_canonical_and_authorization_boundaries_are_exact():
    accepted = _revision()
    low_coverage = _revision(measured_surface_coverage_ppm=799999)
    one_view = _revision(minimum_independent_view_count=1)

    assert accepted.appearance_revision_id == canonical_sha256(accepted.revision_core)
    assert accepted.delivery_authorization.delivery_authorized is True
    assert accepted.delivery_authorization.required_coverage_ppm == 800000
    assert accepted.delivery_authorization.required_independent_view_count == 2
    assert low_coverage.delivery_authorization.delivery_authorized is False
    assert "MEASURED_SURFACE_COVERAGE_REQUIRED" in low_coverage.delivery_authorization.blockers
    assert one_view.delivery_authorization.delivery_authorized is False
    assert "INDEPENDENT_COLOR_VIEWS_REQUIRED" in one_view.delivery_authorization.blockers
    assert accepted.revision_core["source_frame_sha256"] == sorted((HASHES[6], HASHES[7]))
    assert not {"manifest_sha256", "lod_id", "artifacts"} & accepted.revision_core.keys()


def test_final_artifacts_change_lod_id_without_creating_an_appearance_digest_cycle():
    revision = _revision()
    first = _lod(revision)
    changed = _lod(revision, artifact_hash=HASHES[8])

    assert first.lod_id != changed.lod_id
    assert revision.appearance_revision_id == _revision().appearance_revision_id
    assert first.core["appearance_revision_id"] == revision.appearance_revision_id
    assert first.lod_id == canonical_sha256(first.core)
    assert "lod_id" not in first.core
    assert "manifest_sha256" not in first.core


@pytest.mark.parametrize(
    "change",
    [
        {"geometry_revision_id": HASHES[8]},
        {"source_generation_id": HASHES[8]},
        {"calibration_sha256": HASHES[8]},
        {"physical_depth_validation_revision": HASHES[8]},
        {"pose_revision_id": HASHES[8]},
        {"source_frame_sha256": (HASHES[6],)},
        {"appearance_config_sha256": HASHES[7]},
        {"charting": {"implementation": "xatlas", "version": "other", "options": {}}},
        {"atlas_density_policy": {"minimum_mm_per_texel": 4, "maximum_mm_per_texel": 5}},
        {"padding_policy": {"pixels": 4, "mip_levels": 3}},
        {"measured_surface_coverage_ppm": 900000},
        {"minimum_independent_view_count": 3},
    ],
)
def test_every_evidence_or_policy_change_changes_appearance_revision(change):
    assert _revision(**change).appearance_revision_id != _revision().appearance_revision_id


def test_augmented_revision_is_required_if_and_only_if_augmented_primitives_exist():
    with pytest.raises(ValueError, match="rgbd_augmented_revision_id"):
        _revision(contains_augmented_primitives=True)
    with pytest.raises(ValueError, match="rgbd_augmented_revision_id"):
        _revision(rgbd_augmented_revision_id=HASHES[7])

    augmented = _revision(
        contains_augmented_primitives=True,
        rgbd_augmented_revision_id=HASHES[7],
        model_revision_id=HASHES[6],
        assistance_config_sha256=HASHES[5],
    )
    lod = _lod(augmented, source_layer="RGBD_AUGMENTED")
    assert lod.core["rgbd_augmented_revision_id"] == HASHES[7]


def test_lod_rejects_self_hashes_invalid_primitive_ranges_and_atlas_sentinels():
    revision = _revision()
    with pytest.raises(ValueError, match="self|manifest|forbidden"):
        ArtifactRef("manifest_sha256", HASHES[0], 10)
    with pytest.raises(ValueError, match="multiple of three"):
        PrimitiveRecord(0, "BASE_MATERIAL", 65535, 0, 2, "LIDAR_MEASURED", ())
    with pytest.raises(ValueError, match="sentinel"):
        PrimitiveRecord(0, "BASE_MATERIAL", 0, 0, 3, "LIDAR_MEASURED", (0,))
    with pytest.raises(ValueError, match="atlas"):
        _lod(revision, atlas_index=2)
