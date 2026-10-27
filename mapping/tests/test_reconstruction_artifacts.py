import json

import numpy as np

from mapping.pc.go2_mapping_fusion import MeasuredSurface
from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config
from mapping.pc.go2_mapping_reconstruction import LayerArtifactStore


def test_artifact_publish_is_atomic_and_content_addressed(tmp_path):
    surface = MeasuredSurface(
        positions=np.asarray([[0.0, 0.0, 0.0], [0.015, 0.0, 0.0], [0.0, 0.015, 0.0]]),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 3),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    identity = load_reconstruction_config().derived_identity(
        artifact_hashes=("a" * 64,),
        pose_generation="pg-1",
        calibration_hash="b" * 64,
        roi_hash="c" * 64,
        mask_hash="d" * 64,
        algorithm_version="fusion-v1",
    )

    published = LayerArtifactStore(tmp_path).publish(surface, identity)
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))

    assert published.ply_path.is_file()
    assert metadata["derived_cache_key"] == identity.cache_key
    assert metadata["ply_sha256"] == published.ply_sha256
    assert metadata["point_authority"] == "LIDAR_MEASURED"
    assert not list(tmp_path.rglob("*.partial"))
