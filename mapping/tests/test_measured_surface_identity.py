import hashlib

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance.identity import measured_surface_identity
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _identity(**overrides):
    values = {
        "positions": np.array([[1.0, 2.0, 3.0], [-0.0, 0.25, -4.0]], dtype=np.float64),
        "normals": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        "triangles": np.array([[0, 1, 0]], dtype=np.int64),
        "pose_generation": HASH_A,
        "artifact_sha256": (HASH_C, HASH_B),
        "fusion_config_sha256": HASH_D,
    }
    values.update(overrides)
    return measured_surface_identity(**values)


def test_identity_hashes_exact_little_endian_payloads_and_canonical_core():
    identity = _identity()
    positions = np.asarray(
        [[1.0, 2.0, 3.0], [-0.0, 0.25, -4.0]], dtype="<f4", order="C"
    ).tobytes(order="C")
    normals = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype="<f4", order="C"
    ).tobytes(order="C")
    triangles = np.asarray([[0, 1, 0]], dtype="<u4", order="C").tobytes(order="C")

    assert identity.positions_sha256 == hashlib.sha256(positions).hexdigest()
    assert identity.normals_sha256 == hashlib.sha256(normals).hexdigest()
    assert identity.triangles_sha256 == hashlib.sha256(triangles).hexdigest()
    assert identity.core == {
        "schema_version": 1,
        "source_layer": "LIDAR_MEASURED",
        "pose_generation": HASH_A,
        "artifact_sha256": [HASH_B, HASH_C],
        "fusion_config_sha256": HASH_D,
        "vertex_count": 2,
        "triangle_count": 1,
        "positions_sha256": identity.positions_sha256,
        "normals_sha256": identity.normals_sha256,
        "triangles_sha256": identity.triangles_sha256,
    }
    assert identity.revision_id == canonical_sha256(identity.core)


@pytest.mark.parametrize(
    "override",
    [
        {"positions": np.array([[1.0, 2.0, 3.0001], [-0.0, 0.25, -4.0]])},
        {"normals": np.array([[0.0, 0.0, 1.0], [0.999, 0.0, 0.0]])},
        {"triangles": np.empty((0, 3), dtype=np.int64)},
        {"pose_generation": "e" * 64},
        {"artifact_sha256": (HASH_B, "e" * 64)},
        {"fusion_config_sha256": "e" * 64},
    ],
)
def test_identity_changes_when_any_identity_domain_changes(override):
    assert _identity(**override).revision_id != _identity().revision_id


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"positions": np.array([[np.nan, 0.0, 0.0]])}, "positions"),
        ({"normals": np.array([[0.0, np.inf, 0.0]])}, "normals"),
        ({"normals": np.array([[0.0, 0.0, 1.0]])}, "same shape"),
        ({"triangles": np.array([[-1, 0, 1]])}, "triangle indices"),
        ({"triangles": np.array([[0, 1, 2]])}, "triangle indices"),
        ({"artifact_sha256": ()}, "artifact_sha256"),
        ({"artifact_sha256": ("not-a-hash",)}, "artifact_sha256"),
        ({"pose_generation": "pg-1"}, "pose_generation"),
        ({"fusion_config_sha256": "config"}, "fusion_config_sha256"),
    ],
)
def test_identity_rejects_unusable_inputs(override, message):
    with pytest.raises(ValueError, match=message):
        _identity(**override)


def test_identity_is_independent_of_native_dtype_and_artifact_order():
    little = _identity()
    big = _identity(
        positions=np.asarray(
            [[1.0, 2.0, 3.0], [-0.0, 0.25, -4.0]], dtype=">f8"
        ),
        normals=np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=">f8"
        ),
        triangles=np.asarray([[0, 1, 0]], dtype=">i8"),
        artifact_sha256=(HASH_B, HASH_C),
    )

    assert big == little
