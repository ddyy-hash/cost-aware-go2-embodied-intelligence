import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d")

from mapping.pc.go2_mapping_live.live_protocol import validate_live_keyframe
from mapping.pc.go2_mapping_live.live_surface import LiveSurfaceMap, TileId
from mapping.pc.go2_mapping_live.live_surface_mesher import LiveSurfaceMesher, TriangleProvenance
from mapping.robot.go2_mapping_lio.go2_mapping_lio.detail_surface_source import build_live_keyframe_message


SOURCE_HASH = hashlib.sha256(b"surface-config").hexdigest()


def _fixture_points() -> np.ndarray:
    door_y, door_z = np.meshgrid(np.arange(0.0, 0.161, 0.005), np.arange(0.0, 0.161, 0.005), indexing="ij")
    door = np.column_stack((np.zeros(door_y.size), door_y.ravel(), door_z.ravel()))

    angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    handle_y = np.arange(0.035, 0.126, 0.005)
    handle = np.asarray(
        [
            [0.055 + 0.010 * np.cos(theta), y, 0.080 + 0.010 * np.sin(theta)]
            for y in handle_y
            for theta in angle
        ]
    )
    leg_z = np.arange(0.0, 0.201, 0.005)
    leg = np.asarray(
        [
            [0.31 + 0.012 * np.cos(theta), 0.06 + 0.012 * np.sin(theta), z]
            for z in leg_z
            for theta in angle
        ]
    )
    return np.asarray(np.vstack((door, handle, leg)), dtype="<f4")


def _frame(sequence: int, points: np.ndarray, origin: tuple[float, float, float]):
    pose = np.eye(4)
    pose[:3, 3] = origin
    encoded = build_live_keyframe_message(
        building_id="building-a",
        session_id="session-a",
        map_id="map-a",
        producer_instance_id="producer-a",
        keyframe_sequence=sequence,
        timestamp_ns=1_000_000 + sequence,
        calibration_id="PREVIEW_UNCALIBRATED",
        source_config_sha256=SOURCE_HASH,
        points_xyz=points,
        pose_map_from_lidar=pose,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        point_cloud_source="/lio/cloud_deskewed",
        motion_compensated=True,
        points_in_map_frame=True,
    )
    return validate_live_keyframe(encoded.payload)


def test_scale_separated_bpa_keeps_handle_and_leg_faces_without_component_bridges():
    points = _fixture_points()
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    for sequence, origin in enumerate(((-0.5, 0.0, 0.08), (-0.5, 0.06, 0.08), (-0.5, -0.06, 0.08)), start=1):
        surface.integrate(_frame(sequence, points, origin))

    result = LiveSurfaceMesher().build(surface)

    assert result.triangle_count > 0
    assert len(result.triangle_provenance) == result.triangle_count
    assert set(result.triangle_provenance.tolist()) == {TriangleProvenance.MEASURED_LIDAR}
    assert len(result.triangle_uncertainty_m) == result.triangle_count
    assert np.all(result.triangle_uncertainty_m > 0)
    assert np.all(result.triangle_uncertainty_m <= 0.02)
    assert result.scale_sample_counts[0.005] > 0
    assert result.scale_sample_counts[0.02] > 0
    positions = result.positions
    triangles = result.triangles
    triangle_scales = result.triangle_scale_m
    centres = positions[triangles].mean(axis=1)
    assert np.any((centres[:, 0] > 0.04) & (centres[:, 0] < 0.07) & (centres[:, 1] < 0.14))
    assert np.any(centres[:, 0] > 0.28)
    for triangle, scale in zip(triangles, triangle_scales):
        vertices = positions[triangle]
        edges = np.linalg.norm(vertices[[1, 2, 0]] - vertices[[0, 1, 2]], axis=1)
        assert float(edges.max()) <= float(scale) * 3.05
        assert not (vertices[:, 0].min() < 0.01 and vertices[:, 0].max() > 0.04)


def test_empty_bpa_result_uses_bounded_patches_at_eligible_observed_surfels(monkeypatch):
    points = _fixture_points()
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    for sequence, origin in enumerate(
        ((-0.5, 0.0, 0.08), (-0.5, 0.06, 0.08), (-0.5, -0.06, 0.08)),
        start=1,
    ):
        surface.integrate(_frame(sequence, points, origin))
    eligible_sample_sets = []

    def reject_all_bpa_triangles(samples):
        if len(samples.positions):
            eligible_sample_sets.append(samples)
        return np.empty((0, 3), dtype="<u4")

    monkeypatch.setattr(
        LiveSurfaceMesher,
        "_triangulate",
        staticmethod(reject_all_bpa_triangles),
    )

    result = LiveSurfaceMesher().build(surface)

    eligible_centres = np.concatenate(
        [samples.positions for samples in eligible_sample_sets], axis=0
    )
    assert len(eligible_centres) > 0
    assert result.triangle_count == len(eligible_centres)
    assert len(result.positions) == result.triangle_count * 3
    triangle_vertices = result.positions[result.triangles]
    triangle_centres = triangle_vertices.mean(axis=1)
    centre_distances = np.linalg.norm(
        triangle_centres[:, None, :] - eligible_centres[None, :, :], axis=2
    )
    assert np.all(centre_distances.min(axis=1) < 1e-6)
    patch_radii = np.linalg.norm(
        triangle_vertices - triangle_centres[:, None, :], axis=2
    )
    assert np.all(patch_radii > 0)
    assert np.all(
        patch_radii <= result.triangle_scale_m[:, None] * 0.45 + 1e-7
    )
    face_normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    stored_normals = result.normals[result.triangles].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", face_normals, stored_normals) > 0.999)
    assert set(result.triangle_provenance.tolist()) == {
        TriangleProvenance.MEASURED_LIDAR
    }
    np.testing.assert_allclose(
        result.triangle_uncertainty_m,
        result.triangle_scale_m,
    )


def test_mesher_never_uses_poisson_or_fills_an_unsupported_gap():
    source = (
        Path(__file__).resolve().parents[2]
        / "mapping"
        / "pc"
        / "go2_mapping_live"
        / "live_surface_mesher.py"
    ).read_text(encoding="utf-8").lower()

    assert "create_from_point_cloud_ball_pivoting" in source
    assert "poisson" not in source


def test_requested_tile_filters_surfels_before_fine_metrics(monkeypatch):
    near = np.asarray([[0.10, 0.10, 0.10], [0.20, 0.10, 0.10]], dtype="<f4")
    far = np.asarray([[2.10, 0.10, 0.10], [2.20, 0.10, 0.10]], dtype="<f4")
    surface = LiveSurfaceMap(map_id="map-a", source_config_sha256=SOURCE_HASH)
    surface.integrate(_frame(1, np.vstack((near, far)), (-0.5, 0.0, 0.0)))
    captured = []

    def capture_metrics(surfels):
        captured.extend(surfels)
        return ()

    monkeypatch.setattr(LiveSurfaceMesher, "_fine_metrics", staticmethod(capture_metrics))
    monkeypatch.setattr(
        surface,
        "surfels",
        lambda: (_ for _ in ()).throw(AssertionError("tile meshing copied the full surface")),
    )

    LiveSurfaceMesher().build(surface, tiles={TileId(0, 0, 0)})

    assert captured
    assert max(float(surfel.centroid[0]) for surfel in captured) < 0.30
