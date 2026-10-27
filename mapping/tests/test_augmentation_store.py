import json

import numpy as np
import pytest

from mapping.pc.go2_mapping_rgbd.augmentation import AugmentedLayer
from mapping.pc.go2_mapping_rgbd.augmentation_identity import rgbd_augmented_identity
from mapping.pc.go2_mapping_rgbd.augmentation_store import RgbdAugmentationStore


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2"))


def _layer(offset=0.0):
    return AugmentedLayer(
        name="RGBD_AUGMENTED",
        positions=np.asarray(
            [[offset, 0.0, 1.0], [0.1 + offset, 0.0, 1.0], [offset, 0.1, 1.0]]
        ),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 3),
        triangles=np.asarray([[0, 1, 2]], dtype=np.uint32),
        uncertainty_m=np.asarray([0.02, 0.02, 0.02]),
        source_vertex_ids=("v0", "v1", "v2"),
        source_triangle_ids=("t0",),
        geometry_authority=False,
        source="DEPTH_ANYTHING_METRIC_ASSISTED",
    )


def _identity(layer):
    return rgbd_augmented_identity(
        source_generation_id=HASHES[0],
        source_revision_id=HASHES[1],
        calibration_sha256=HASHES[2],
        physical_depth_validation_revision=HASHES[3],
        model_revision_id=HASHES[4],
        assistance_config_sha256=HASHES[5],
        source_frame_sha256=(HASHES[7], HASHES[6]),
        positions=layer.positions,
        normals=layer.normals,
        triangles=layer.triangles,
        uncertainty_m=layer.uncertainty_m,
        source_vertex_ids=layer.source_vertex_ids,
        source_triangle_ids=layer.source_triangle_ids,
    )


def test_store_publishes_reuses_and_restarts_with_exact_resolver(tmp_path):
    layer = _layer()
    identity = _identity(layer)
    store = RgbdAugmentationStore(tmp_path)

    first = store.publish(layer, identity, expected_source_generation_id=HASHES[0])
    repeated = store.publish(layer, identity, expected_source_generation_id=HASHES[0])
    restarted = RgbdAugmentationStore(tmp_path).resolve(HASHES[0], identity.revision_id)

    assert repeated == first == restarted
    assert first.manifest["manifest_sha256"] == first.manifest_sha256
    assert first.layer.geometry_authority is False
    assert first.layer.source_vertex_ids == layer.source_vertex_ids
    assert first.layer.triangles.tolist() == [[0, 1, 2]]
    assert first.root_path == tmp_path / "rgbd-augmentation" / HASHES[0] / identity.revision_id
    assert not list(tmp_path.rglob("*.partial"))


def test_store_rejects_stale_source_corruption_and_path_escape(tmp_path):
    layer = _layer()
    identity = _identity(layer)
    store = RgbdAugmentationStore(tmp_path)
    published = store.publish(layer, identity)

    with pytest.raises(FileNotFoundError):
        store.resolve("9" * 64, identity.revision_id)
    with pytest.raises(ValueError):
        store.resolve("../escape", identity.revision_id)

    payload = bytearray(published.positions_path.read_bytes())
    payload[-1] ^= 1
    published.positions_path.write_bytes(payload)
    with pytest.raises(ValueError, match="positions"):
        store.resolve(HASHES[0], identity.revision_id)


def test_store_rejects_conflicting_existing_revision(tmp_path):
    layer = _layer()
    identity = _identity(layer)
    store = RgbdAugmentationStore(tmp_path)
    published = store.publish(layer, identity)
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = "CONFLICTING_SOURCE"
    published.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        store.publish(layer, identity)


def test_store_cleans_interrupted_staging_but_never_needs_current_pointer(tmp_path):
    layer = _layer()
    identity = _identity(layer)
    store = RgbdAugmentationStore(tmp_path)
    published = store.publish(layer, identity)
    generation_root = published.root_path.parent
    abandoned = generation_root / ".rgbd-abandoned.partial"
    abandoned.mkdir()
    (abandoned / "positions.bin").write_bytes(b"partial")
    current = generation_root / "CURRENT.json"
    current.unlink()

    restarted = RgbdAugmentationStore(tmp_path)

    assert not abandoned.exists()
    assert restarted.resolve(HASHES[0], identity.revision_id) == published
