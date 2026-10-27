"""Fail-closed bridge from validated RGB-D evidence to exact tile order."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256, is_sha256
from mapping.pc.go2_mapping_rgbd.texture import VertexColor

from .baker import TileAtlasBakeRequest
from .baker import TileAtlasBaker
from .tile_codec import decode_g2tm_v1


@dataclass(frozen=True)
class TileAppearanceScope:
    building_id: str
    floor_id: str
    component_id: str
    submap_id: str
    attachment_state: str
    publication_scope: str

    def __post_init__(self) -> None:
        for name in ("building_id", "floor_id", "component_id", "submap_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"appearance scope {name} must be non-empty")
        state = str(self.attachment_state).upper()
        if state not in {"ROOT", "ATTACHED", "DETACHED"}:
            raise ValueError("appearance scope attachment_state is invalid")
        object.__setattr__(self, "attachment_state", state)
        publication_scope = str(self.publication_scope).upper()
        if publication_scope not in {"LOCAL", "BUILDING"}:
            raise ValueError("appearance publication_scope is invalid")
        object.__setattr__(self, "publication_scope", publication_scope)

    @property
    def building_delivery_eligible(self) -> bool:
        return (
            self.publication_scope == "BUILDING"
            and self.attachment_state in {"ROOT", "ATTACHED"}
        )


class ValidatedTileAppearanceProcessor:
    """Reload and bake one queued immutable tile using current exact evidence."""

    def __init__(
        self,
        *,
        tile_store: Any,
        rgbd_refiner: Any,
        appearance_store: Any,
        preview_store: Any,
        appearance_config_sha256: str,
        calibration_sha256_resolver: Callable[[], str | None],
        publication_scope: str,
        atlas_dimension: int = 256,
        padding_pixels: int = 2,
        mip_filter_radius_pixels: int = 1,
    ) -> None:
        self.tile_store = tile_store
        self.rgbd_refiner = rgbd_refiner
        self.appearance_config_sha256 = _require_sha256(
            appearance_config_sha256, "appearance configuration"
        )
        self.calibration_sha256_resolver = calibration_sha256_resolver
        scope = str(publication_scope).upper()
        if scope not in {"LOCAL", "BUILDING"}:
            raise ValueError("appearance processor publication_scope is invalid")
        self.publication_scope = scope
        self.atlas_dimension = int(atlas_dimension)
        self.padding_pixels = int(padding_pixels)
        self.mip_filter_radius_pixels = int(mip_filter_radius_pixels)
        self.baker = TileAtlasBaker(
            source_loader=self.tile_store.load,
            appearance_store=appearance_store,
            preview_store=preview_store,
        )

    def scheduling_revision(self) -> str:
        snapshot = self.rgbd_refiner.snapshot()
        assistance = snapshot.get("depth_anything") or {}
        return canonical_sha256({
            "schema_version": 1,
            "publication_scope": self.publication_scope,
            "appearance_config_sha256": self.appearance_config_sha256,
            "rgbd_evidence": {
                name: snapshot.get(name)
                for name in (
                    "calibration_id",
                    "calibration_sha256",
                    "accepted_depth_frames",
                    "accepted_color_observations",
                    "depth_correspondence_count",
                    "blockers",
                    "texture",
                )
            },
            "augmentation_evidence": {
                name: assistance.get(name)
                for name in (
                    "source_generation_id",
                    "model_revision_id",
                    "assistance_config_sha256",
                    "fit_accepted",
                    "augmentation_reference",
                    "blockers",
                )
            },
        })

    def __call__(self, queued_metadata: Mapping[str, Any]):
        if not isinstance(queued_metadata, Mapping):
            raise TypeError("queued tile metadata must be a mapping")
        queued_evidence_revision = (
            queued_metadata.get("appearance_evidence_revision_id")
            or self.scheduling_revision()
        )
        if queued_evidence_revision != self.scheduling_revision():
            raise ValueError("queued appearance evidence revision is stale")
        revision = _require_sha256(queued_metadata.get("revision_id"), "queued tile revision")
        metadata, body = self.tile_store.load(revision)
        immutable_fields = (
            "revision_id",
            "geometry_revision_id",
            "source_generation_id",
            "pose_revision_id",
            "content_sha256",
            "building_id",
            "floor_id",
            "component_id",
            "submap_id",
            "attachment_state",
            "tile_id",
            "vertex_count",
            "triangle_count",
        )
        if any(queued_metadata.get(name) != metadata.get(name) for name in immutable_fields):
            raise ValueError("queue metadata does not match reloaded immutable tile")
        calibration = self.calibration_sha256_resolver()
        scope = TileAppearanceScope(
            building_id=metadata["building_id"],
            floor_id=metadata["floor_id"],
            component_id=metadata["component_id"],
            submap_id=metadata["submap_id"],
            attachment_state=metadata["attachment_state"],
            publication_scope=self.publication_scope,
        )
        request_kwargs = dict(
            metadata=metadata,
            tile_body=body,
            rgbd_refiner=self.rgbd_refiner,
            scope=scope,
            expected_calibration_sha256=calibration,
            pose_revision_id=metadata["pose_revision_id"],
            appearance_config_sha256=self.appearance_config_sha256,
            atlas_dimension=self.atlas_dimension,
            padding_pixels=self.padding_pixels,
            mip_filter_radius_pixels=self.mip_filter_radius_pixels,
        )
        baseline = {}

        def revalidate_evidence() -> None:
            if self.scheduling_revision() != queued_evidence_revision:
                raise ValueError("appearance evidence changed during bake")
            if self.calibration_sha256_resolver() != request_kwargs["expected_calibration_sha256"]:
                raise ValueError("appearance calibration changed during bake")
            fresh = build_validated_tile_bake_request(**request_kwargs)
            if _evidence_fingerprint(fresh) != baseline["fingerprint"]:
                raise ValueError("appearance evidence changed during bake")

        request = build_validated_tile_bake_request(
            **request_kwargs,
            evidence_revalidator=revalidate_evidence,
        )
        baseline["fingerprint"] = _evidence_fingerprint(request)
        return self.baker.bake(request)


def build_validated_tile_bake_request(
    *,
    metadata: Mapping[str, Any],
    tile_body: bytes,
    rgbd_refiner: Any,
    scope: TileAppearanceScope,
    expected_calibration_sha256: str,
    pose_revision_id: str,
    appearance_config_sha256: str,
    atlas_dimension: int = 256,
    padding_pixels: int = 2,
    mip_filter_radius_pixels: int = 1,
    evidence_revalidator: Callable[[], None] | None = None,
) -> TileAtlasBakeRequest:
    """Construct a bake request only from exact, aligned immutable evidence."""

    if not isinstance(metadata, Mapping):
        raise TypeError("tile metadata must be a mapping")
    if not isinstance(scope, TileAppearanceScope):
        raise TypeError("scope must be a TileAppearanceScope")
    body = bytes(tile_body)
    _require_sha256(metadata.get("revision_id"), "tile revision")
    _require_sha256(metadata.get("geometry_revision_id"), "geometry revision")
    generation = _require_sha256(metadata.get("source_generation_id"), "source generation")
    calibration = _require_sha256(expected_calibration_sha256, "expected calibration")
    pose = _require_sha256(pose_revision_id, "pose revision")
    config = _require_sha256(appearance_config_sha256, "appearance configuration")
    if metadata.get("pose_revision_id") != pose:
        raise ValueError("pose revision does not match immutable tile metadata")
    if metadata.get("content_sha256") != hashlib.sha256(body).hexdigest():
        raise ValueError("tile content hash does not match immutable metadata")
    for name in ("building_id", "floor_id", "component_id", "submap_id", "attachment_state"):
        expected = getattr(scope, name)
        actual = str(metadata.get(name, ""))
        if name == "attachment_state":
            actual = actual.upper()
        if actual != expected:
            raise ValueError(f"tile appearance scope mismatch for {name}")

    tile = decode_g2tm_v1(body)
    if metadata.get("vertex_count") != len(tile.positions) or metadata.get("triangle_count") != len(tile.triangles):
        raise ValueError("tile measured counts do not match immutable body")
    voxel_size = float(getattr(rgbd_refiner, "voxel_size_m", 0.0))
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("RGB-D refiner voxel size is invalid")
    # G2TM stores float32 positions. Recover exact voxel-boundary coordinates
    # that may round one ULP below the source float64 value during serialization.
    scaled = np.asarray(tile.positions, dtype=np.float64) / voxel_size
    quantization_tolerance = np.maximum(1e-7, np.abs(scaled) * np.finfo(np.float32).eps * 2.0)
    voxel_keys = np.floor(scaled + quantization_tolerance).astype(np.int64)
    spatial_ids = tuple(f"{x}:{y}:{z}" for x, y, z in voxel_keys)
    evidence = rgbd_refiner.appearance_evidence(spatial_ids)
    blockers = tuple(getattr(evidence, "blockers", ()))
    if not getattr(evidence, "ready", False):
        reason = ",".join(blockers) if blockers else "APPEARANCE_EVIDENCE_NOT_READY"
        raise ValueError(reason)
    if getattr(evidence, "measured_generation", None) != pose:
        raise ValueError("appearance evidence pose revision is stale")
    if getattr(evidence, "calibration_sha256", None) != calibration:
        raise ValueError("appearance evidence calibration does not match expected calibration")
    validation_revision = _require_sha256(
        getattr(evidence, "physical_depth_validation_revision", None),
        "physical depth validation revision",
    )
    source_frames = tuple(getattr(evidence, "source_frame_sha256", ()))
    if not source_frames or any(not is_sha256(value) or value != value.lower() for value in source_frames):
        raise ValueError("appearance evidence source frames are invalid")
    evidence_colors = tuple(getattr(evidence, "vertex_colors", ()))
    by_spatial_id = {color.vertex_id: color for color in evidence_colors}
    if len(by_spatial_id) != len(evidence_colors) or any(identifier not in by_spatial_id for identifier in spatial_ids):
        raise ValueError("appearance evidence does not cover exact tile vertex order")

    tile_colors = []
    accepted = []
    accepted_views = []
    for index, spatial_id in enumerate(spatial_ids):
        source = by_spatial_id[spatial_id]
        color = VertexColor(
            vertex_id=f"{metadata['revision_id']}:{index}",
            mean_linear_rgb=tuple(source.mean_linear_rgb),
            channel_stddev=tuple(source.channel_stddev),
            accepted_count=int(source.accepted_count),
            rejected_count=int(source.rejected_count),
            independent_views=int(source.independent_views),
            source_frame_sha256=tuple(source.source_frame_sha256),
            accepted_for_texture=bool(source.accepted_for_texture),
        )
        tile_colors.append(color)
        accepted.append(color.accepted_for_texture)
        if color.accepted_for_texture:
            accepted_views.append(color.independent_views)
    coverage_ppm = sum(accepted) * 1_000_000 // len(accepted)
    minimum_views = min(accepted_views, default=0)
    triangle_authorized = tuple(
        all(accepted[int(vertex)] for vertex in triangle)
        and min(tile_colors[int(vertex)].independent_views for vertex in triangle) >= 2
        for triangle in tile.triangles
    )
    triangle_colors = np.asarray(
        [
            np.mean(
                [np.asarray(tile_colors[int(vertex)].mean_linear_rgb, dtype=np.float64) for vertex in triangle],
                axis=0,
            )
            for triangle in tile.triangles
        ],
        dtype=np.float32,
    )
    return TileAtlasBakeRequest(
        tile_revision_id=metadata["revision_id"],
        geometry_revision_id=metadata["geometry_revision_id"],
        source_generation_id=generation,
        calibration_id=str(getattr(evidence, "calibration_id", "")),
        calibration_sha256=calibration,
        physical_depth_validation_revision=validation_revision,
        pose_revision_id=pose,
        source_frame_sha256=source_frames,
        appearance_config_sha256=config,
        measured_surface_coverage_ppm=coverage_ppm,
        minimum_independent_view_count=minimum_views,
        triangle_linear_rgb=triangle_colors,
        triangle_authorized=triangle_authorized,
        vertex_colors=tuple(tile_colors),
        atlas_dimension=atlas_dimension,
        padding_pixels=padding_pixels,
        mip_filter_radius_pixels=mip_filter_radius_pixels,
        building_delivery_eligible=scope.building_delivery_eligible,
        evidence_revalidator=evidence_revalidator,
    )


def _evidence_fingerprint(request: TileAtlasBakeRequest) -> str:
    return canonical_sha256(
        {
            "calibration_id": request.calibration_id,
            "calibration_sha256": request.calibration_sha256,
            "physical_depth_validation_revision": request.physical_depth_validation_revision,
            "pose_revision_id": request.pose_revision_id,
            "source_frame_sha256": request.source_frame_sha256,
            "measured_surface_coverage_ppm": request.measured_surface_coverage_ppm,
            "minimum_independent_view_count": request.minimum_independent_view_count,
            "triangle_linear_rgb": request.triangle_linear_rgb.tolist(),
            "triangle_authorized": request.triangle_authorized,
            "vertex_colors": [
                {
                    "vertex_id": color.vertex_id,
                    "mean_linear_rgb": color.mean_linear_rgb,
                    "channel_stddev": color.channel_stddev,
                    "accepted_count": color.accepted_count,
                    "rejected_count": color.rejected_count,
                    "independent_views": color.independent_views,
                    "source_frame_sha256": color.source_frame_sha256,
                    "accepted_for_texture": color.accepted_for_texture,
                }
                for color in request.vertex_colors
            ],
        }
    )


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not is_sha256(value) or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value
