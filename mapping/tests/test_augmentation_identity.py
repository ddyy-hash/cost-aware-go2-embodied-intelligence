import numpy as np

from mapping.pc.go2_mapping_rgbd.augmentation_identity import rgbd_augmented_identity


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2"))


def _identity(**changes):
    values = {
        "source_generation_id": HASHES[0],
        "source_revision_id": HASHES[1],
        "calibration_sha256": HASHES[2],
        "physical_depth_validation_revision": HASHES[3],
        "model_revision_id": HASHES[4],
        "assistance_config_sha256": HASHES[5],
        "source_frame_sha256": (HASHES[7], HASHES[6]),
        "positions": np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]),
        "normals": np.asarray([[0.0, 0.0, 1.0]] * 3),
        "triangles": np.asarray([[0, 1, 2]], dtype=np.int32),
        "uncertainty_m": np.asarray([0.02, 0.02, 0.02]),
        "source_vertex_ids": ("v0", "v1", "v2"),
        "source_triangle_ids": ("t0",),
    }
    values.update(changes)
    return rgbd_augmented_identity(**values)


def test_augmentation_identity_is_canonical_exact_and_source_ordered():
    identity = _identity()
    assert len(identity.revision_id) == 64
    assert identity.core["source_frame_sha256"] == sorted((HASHES[6], HASHES[7]))
    assert identity.core["vertex_count"] == 3
    assert identity.core["triangle_count"] == 1
    assert identity.positions_sha256
    assert identity.uncertainty_sha256


def test_every_geometry_identity_or_provenance_change_changes_revision():
    baseline = _identity().revision_id
    variants = [
        _identity(positions=np.asarray([[0.0, 0.0, 1.1], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]])).revision_id,
        _identity(normals=np.asarray([[0.0, 1.0, 0.0]] * 3)).revision_id,
        _identity(triangles=np.asarray([[2, 1, 0]], dtype=np.int32)).revision_id,
        _identity(uncertainty_m=np.asarray([0.021, 0.02, 0.02])).revision_id,
        _identity(source_vertex_ids=("v0", "v2", "v1")).revision_id,
        _identity(source_triangle_ids=("other",)).revision_id,
        _identity(model_revision_id=None).revision_id,
        _identity(source_generation_id="f" * 64).revision_id,
    ]
    assert all(value != baseline for value in variants)
