"""Generation-keyed immutable tile atlas baker."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from mapping.pc.go2_mapping_rgbd.texture import VertexColor

from .atlas_codec import encode_render_indices_v1, encode_render_uvs_v1, encode_render_vertices_v1
from .charting import ChartedMeshV1, chart_mesh_v1
from .lod import build_lod_policies
from .rasterizer import RasterizedAtlas, rasterize_rgba8_atlas
from .tile_codec import DecodedTileV1, decode_g2tm_v1
from .tile_identity import ArtifactRef, PrimitiveRecord, TileAppearanceRevision, build_lod_record, tile_appearance_revision
from .tile_store import PublishedTileAppearance, TileAppearanceStore, TileLodPayload
from .tile_vertex_color import (
    PublishedTileVertexColor,
    TileVertexColorStore,
    build_tile_vertex_color_sidecar,
)


class StaleTileGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TileAtlasBakeRequest:
    tile_revision_id: str
    geometry_revision_id: str
    source_generation_id: str
    calibration_id: str
    calibration_sha256: str
    physical_depth_validation_revision: str
    pose_revision_id: str
    source_frame_sha256: tuple[str, ...]
    appearance_config_sha256: str
    measured_surface_coverage_ppm: int
    minimum_independent_view_count: int
    triangle_linear_rgb: np.ndarray
    triangle_authorized: tuple[bool, ...]
    vertex_colors: tuple[VertexColor, ...]
    atlas_dimension: int = 256
    padding_pixels: int = 2
    mip_filter_radius_pixels: int = 1
    near_mm_per_texel: float = 4.0
    model_revision_id: str | None = None
    assistance_config_sha256: str | None = None
    rgbd_augmented_revision_id: str | None = None
    augmented_triangle_linear_rgb: np.ndarray | None = None
    augmented_triangle_authorized: tuple[bool, ...] = ()
    building_delivery_eligible: bool = True
    evidence_revalidator: Callable[[], None] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        colors = np.ascontiguousarray(self.triangle_linear_rgb, dtype=np.float32)
        if colors.ndim != 2 or colors.shape[1:] != (3,) or not np.all(np.isfinite(colors)):
            raise ValueError("triangle_linear_rgb must be a finite N by 3 array")
        if np.any(colors < 0.0) or np.any(colors > 1.0):
            raise ValueError("triangle_linear_rgb must be normalized")
        authorized = tuple(bool(value) for value in self.triangle_authorized)
        if len(authorized) != len(colors):
            raise ValueError("triangle_authorized must align with triangle colors")
        colors.setflags(write=False)
        object.__setattr__(self, "triangle_linear_rgb", colors)
        object.__setattr__(self, "triangle_authorized", authorized)
        object.__setattr__(self, "source_frame_sha256", tuple(self.source_frame_sha256))
        object.__setattr__(self, "vertex_colors", tuple(self.vertex_colors))
        augmented = self.augmented_triangle_linear_rgb
        if augmented is not None:
            augmented = np.ascontiguousarray(augmented, dtype=np.float32)
            if augmented.ndim != 2 or augmented.shape[1:] != (3,) or not np.all(np.isfinite(augmented)):
                raise ValueError("augmented_triangle_linear_rgb must be a finite N by 3 array")
            if np.any(augmented < 0.0) or np.any(augmented > 1.0):
                raise ValueError("augmented_triangle_linear_rgb must be normalized")
            augmented.setflags(write=False)
        augmented_authorized = tuple(bool(value) for value in self.augmented_triangle_authorized)
        if augmented is not None and len(augmented_authorized) != len(augmented):
            raise ValueError("augmented_triangle_authorized must align with augmented colors")
        if self.rgbd_augmented_revision_id is None and (augmented is not None or augmented_authorized):
            raise ValueError("augmentation color evidence requires rgbd_augmented_revision_id")
        if self.building_delivery_eligible is not True and self.building_delivery_eligible is not False:
            raise ValueError("building_delivery_eligible must be boolean")
        object.__setattr__(self, "augmented_triangle_linear_rgb", augmented)
        object.__setattr__(self, "augmented_triangle_authorized", augmented_authorized)


@dataclass(frozen=True)
class TileAtlasBakeResult:
    revision: TileAppearanceRevision
    preview: PublishedTileVertexColor
    published: PublishedTileAppearance
    state: str
    building_authorized: bool
    export_authorized: bool


class TileAtlasBaker:
    def __init__(
        self,
        *,
        source_loader: Callable[[str], tuple[Mapping, bytes]],
        augmentation_resolver: Callable[[str, str], Any] | None = None,
        appearance_store: TileAppearanceStore,
        preview_store: TileVertexColorStore,
        progress_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._source_loader = source_loader
        self._augmentation_resolver = augmentation_resolver
        self._appearance_store = appearance_store
        self._preview_store = preview_store
        self._progress_hook = progress_hook

    def bake(self, request: TileAtlasBakeRequest) -> TileAtlasBakeResult:
        if not isinstance(request, TileAtlasBakeRequest):
            raise TypeError("request must be a TileAtlasBakeRequest")
        metadata, body, tile = self._load_and_validate(request)
        augmentation = self._resolve_augmentation(request)
        augmentation_fingerprint = self._augmentation_fingerprint(augmentation)
        if len(tile.triangles) != len(request.triangle_linear_rgb):
            raise ValueError("tile triangles and color evidence do not align")
        if len(tile.positions) != len(request.vertex_colors):
            raise ValueError("tile vertices and preview color evidence do not align")

        chart, raster = self._chart_and_raster(tile, request, augmentation)
        revision = tile_appearance_revision(
            tile_revision_id=request.tile_revision_id,
            geometry_revision_id=request.geometry_revision_id,
            source_generation_id=request.source_generation_id,
            calibration_id=request.calibration_id,
            calibration_sha256=request.calibration_sha256,
            physical_depth_validation_revision=request.physical_depth_validation_revision,
            pose_revision_id=request.pose_revision_id,
            model_revision_id=request.model_revision_id,
            assistance_config_sha256=request.assistance_config_sha256,
            source_frame_sha256=request.source_frame_sha256,
            appearance_config_sha256=request.appearance_config_sha256,
            charting=chart.charting_identity,
            atlas_density_policy={
                "minimum_mm_per_texel": 3,
                "maximum_mm_per_texel": 5,
                "selected_mm_per_texel": float(request.near_mm_per_texel),
            },
            padding_policy={
                "pixels": int(request.padding_pixels),
                "mip_filter_radius_pixels": int(request.mip_filter_radius_pixels),
            },
            baker_algorithm_version="tile-atlas-baker-v1",
            rgbd_augmented_revision_id=request.rgbd_augmented_revision_id,
            contains_augmented_primitives=request.rgbd_augmented_revision_id is not None,
            measured_surface_coverage_ppm=request.measured_surface_coverage_ppm,
            minimum_independent_view_count=request.minimum_independent_view_count,
        )

        self._revalidate(request, metadata, body, augmentation_fingerprint)
        preview = self._preview_store.publish(
            build_tile_vertex_color_sidecar(revision=revision, vertex_colors=request.vertex_colors)
        )
        self._progress("PREVIEW_PUBLISHED")

        lods = self._build_lods(request, chart, raster, revision)
        self._revalidate(request, metadata, body, augmentation_fingerprint)
        authorized = revision.delivery_authorization.delivery_authorized
        building_authorized = authorized and request.building_delivery_eligible
        published = self._appearance_store.publish(
            revision,
            lods,
            building_delivery_eligible=request.building_delivery_eligible,
        )
        self._progress("ATLAS_PUBLISHED")
        return TileAtlasBakeResult(
            revision,
            preview,
            published,
            "TEXTURED_READY" if authorized else "DIAGNOSTIC_ATLAS_READY",
            building_authorized,
            building_authorized,
        )

    def _load_and_validate(
        self, request: TileAtlasBakeRequest
    ) -> tuple[dict, bytes, DecodedTileV1]:
        metadata_raw, body_raw = self._source_loader(request.tile_revision_id)
        metadata = dict(metadata_raw)
        body = bytes(body_raw)
        if metadata.get("revision_id") != request.tile_revision_id:
            raise ValueError("source tile revision does not match bake request")
        if metadata.get("geometry_revision_id") != request.geometry_revision_id:
            raise ValueError("source geometry revision does not match bake request")
        if metadata.get("source_generation_id") != request.source_generation_id:
            raise ValueError("source generation does not match bake request")
        if metadata.get("content_sha256") != hashlib.sha256(body).hexdigest():
            raise ValueError("source tile content hash verification failed")
        tile = decode_g2tm_v1(body)
        if metadata.get("vertex_count") != len(tile.positions) or metadata.get("triangle_count") != len(tile.triangles):
            raise ValueError("source tile measured counts do not match binary")
        return metadata, body, tile

    def _revalidate(
        self,
        request: TileAtlasBakeRequest,
        baseline_metadata: Mapping,
        baseline_body: bytes,
        baseline_augmentation_fingerprint: tuple | None,
    ) -> None:
        try:
            metadata, body, _ = self._load_and_validate(request)
        except ValueError as exc:
            if "source generation" in str(exc):
                raise StaleTileGenerationError("source generation changed during appearance bake") from exc
            raise
        if metadata != dict(baseline_metadata) or body != baseline_body:
            raise StaleTileGenerationError("source tile changed during appearance bake")
        augmentation = self._resolve_augmentation(request)
        if self._augmentation_fingerprint(augmentation) != baseline_augmentation_fingerprint:
            raise StaleTileGenerationError("RGB-D augmentation changed during appearance bake")
        if request.evidence_revalidator is not None:
            request.evidence_revalidator()

    def _resolve_augmentation(self, request: TileAtlasBakeRequest) -> Any | None:
        revision = request.rgbd_augmented_revision_id
        if revision is None:
            return None
        if self._augmentation_resolver is None:
            raise ValueError("augmentation resolver is required for rgbd_augmented_revision_id")
        published = self._augmentation_resolver(request.source_generation_id, revision)
        if getattr(published, "source_generation_id", None) != request.source_generation_id:
            raise ValueError("augmentation source generation does not match bake request")
        if getattr(published, "rgbd_augmented_revision_id", None) != revision:
            raise ValueError("augmentation revision does not match bake request")
        layer = getattr(published, "layer", None)
        manifest = getattr(published, "manifest", {})
        if (
            layer is None
            or getattr(layer, "name", None) != "RGBD_AUGMENTED"
            or getattr(layer, "geometry_authority", None) is not False
            or not isinstance(manifest, Mapping)
            or manifest.get("geometry_authority") is not False
        ):
            raise ValueError("augmentation must remain RGBD_AUGMENTED and non-authoritative")
        return published

    @staticmethod
    def _augmentation_fingerprint(published: Any | None) -> tuple | None:
        if published is None:
            return None
        layer = published.layer
        arrays = (layer.positions, layer.normals, layer.triangles, layer.uncertainty_m)
        return (
            published.source_generation_id,
            published.rgbd_augmented_revision_id,
            getattr(published, "manifest_sha256", None),
            *(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for value in arrays),
            tuple(layer.source_vertex_ids),
            tuple(layer.source_triangle_ids),
            layer.source,
            layer.geometry_authority,
        )

    @staticmethod
    def _chart_and_raster(
        tile: DecodedTileV1, request: TileAtlasBakeRequest, augmentation: Any | None
    ) -> tuple[ChartedMeshV1, RasterizedAtlas]:
        positions = tile.positions
        normals = tile.normals
        triangles = tile.triangles
        source_vertex_ids = np.arange(len(tile.positions), dtype=np.uint32)
        source_triangle_ids = np.arange(len(tile.triangles), dtype=np.uint32)
        layers = ("LIDAR_MEASURED",) * len(tile.triangles)
        colors = request.triangle_linear_rgb
        authorized_values = request.triangle_authorized
        allowed_augmented_vertices = None
        allowed_augmented_triangles = None
        if augmentation is not None:
            layer = augmentation.layer
            augmented_colors = request.augmented_triangle_linear_rgb
            if augmented_colors is None or len(augmented_colors) != len(layer.triangles):
                raise ValueError("augmentation triangle color evidence does not match exact revision")
            vertex_offset = len(tile.positions)
            triangle_offset = len(tile.triangles)
            positions = np.concatenate((tile.positions, np.asarray(layer.positions, dtype=np.float32)))
            normals = np.concatenate((tile.normals, np.asarray(layer.normals, dtype=np.float32)))
            triangles = np.concatenate((
                tile.triangles,
                np.asarray(layer.triangles, dtype=np.uint32) + vertex_offset,
            ))
            source_vertex_ids = np.arange(len(positions), dtype=np.uint32)
            source_triangle_ids = np.arange(len(triangles), dtype=np.uint32)
            layers += ("RGBD_AUGMENTED",) * len(layer.triangles)
            colors = np.concatenate((request.triangle_linear_rgb, augmented_colors))
            authorized_values += request.augmented_triangle_authorized
            allowed_augmented_vertices = set(range(vertex_offset, len(positions)))
            allowed_augmented_triangles = set(range(triangle_offset, len(triangles)))
        materials = tuple("VALIDATED_UV_ATLAS" if value else "BASE_MATERIAL" for value in authorized_values)
        for _ in range(4):
            chart = chart_mesh_v1(
                positions,
                normals,
                triangles,
                source_vertex_ids=source_vertex_ids,
                source_triangle_ids=source_triangle_ids,
                triangle_materials=materials,
                triangle_source_layers=layers,
                allowed_augmented_vertex_ids=allowed_augmented_vertices,
                allowed_augmented_triangle_ids=allowed_augmented_triangles,
            )
            ordered_colors = colors[chart.source_triangle_ids]
            authorized = np.asarray(authorized_values, dtype=bool)[chart.source_triangle_ids]
            raster = rasterize_rgba8_atlas(
                chart.uvs,
                chart.render_triangles,
                ordered_colors,
                authorized,
                width=request.atlas_dimension,
                height=request.atlas_dimension,
                padding_pixels=request.padding_pixels,
                mip_filter_radius_pixels=request.mip_filter_radius_pixels,
            )
            if raster.triangle_materials == chart.triangle_materials:
                return chart, raster
            by_source = {int(source): material for source, material in zip(chart.source_triangle_ids, raster.triangle_materials)}
            materials = tuple(by_source[index] for index in range(len(triangles)))
        raise ValueError("atlas authorization did not converge to whole-triangle materials")

    @staticmethod
    def _build_lods(
        request: TileAtlasBakeRequest,
        chart: ChartedMeshV1,
        raster: RasterizedAtlas,
        revision: TileAppearanceRevision,
    ) -> tuple[TileLodPayload, ...]:
        policies = build_lod_policies(
            tuple(int(value) for value in chart.source_triangle_ids),
            near_mm_per_texel=request.near_mm_per_texel,
        )
        payloads = []
        for policy in policies:
            selected = set(policy.source_triangle_ids)
            augmented_sources = [
                int(source)
                for source, layer in zip(chart.source_triangle_ids, chart.triangle_source_layers)
                if layer == "RGBD_AUGMENTED"
            ]
            if augmented_sources and not selected.intersection(augmented_sources):
                selected.add(min(augmented_sources))
            faces = [index for index, source in enumerate(chart.source_triangle_ids) if int(source) in selected]
            groups: dict[tuple[str, int, str], list[int]] = {}
            for face in faces:
                key = (
                    raster.triangle_materials[face],
                    int(raster.triangle_atlas_indices[face]),
                    chart.triangle_source_layers[face],
                )
                groups.setdefault(key, []).append(face)
            ordered_faces = [face for key in sorted(groups) for face in groups[key]]
            used_old = []
            old_to_new = {}
            triangles = []
            primitives = []
            offset = 0
            for primitive_id, key in enumerate(sorted(groups)):
                primitive_faces = groups[key]
                source_ids = []
                for face in primitive_faces:
                    triangle = []
                    for old_value in chart.render_triangles[face]:
                        old = int(old_value)
                        if old not in old_to_new:
                            old_to_new[old] = len(used_old)
                            used_old.append(old)
                        triangle.append(old_to_new[old])
                    triangles.append(triangle)
                    source_ids.append(int(chart.source_triangle_ids[face]))
                material, atlas_index, layer = key
                count = len(primitive_faces) * 3
                primitives.append(PrimitiveRecord(
                    primitive_id, material, atlas_index, offset, count, layer, tuple(source_ids)
                ))
                offset += count
            used = np.asarray(used_old, dtype=np.uint32)
            render_vertices = encode_render_vertices_v1(
                request.tile_revision_id,
                revision.appearance_revision_id,
                chart.render_positions[used],
                chart.render_normals[used],
                chart.source_vertex_ids[used],
            )
            render_indices = encode_render_indices_v1(
                request.tile_revision_id,
                revision.appearance_revision_id,
                np.asarray(triangles, dtype=np.uint32),
                np.asarray([int(chart.source_triangle_ids[face]) for face in ordered_faces], dtype=np.uint32),
            )
            render_uvs = encode_render_uvs_v1(
                request.tile_revision_id,
                revision.appearance_revision_id,
                chart.uvs[used],
                chart.atlas_indices[used],
                chart.flags[used],
            )
            artifacts = {
                "render-vertices.bin": render_vertices,
                "render-indices.bin": render_indices,
                "render-uvs.bin": render_uvs,
                "atlas-0.png": raster.png_bytes,
            }
            references = tuple(
                ArtifactRef(name, hashlib.sha256(value).hexdigest(), len(value))
                for name, value in artifacts.items()
            )
            policy_core = {
                "name": policy.name,
                "order": policy.order,
                "mm_per_texel": policy.mm_per_texel,
                "minimum_distance_m": policy.minimum_distance_m,
                "maximum_distance_m": policy.maximum_distance_m,
                "source_triangle_ids": list(policy.source_triangle_ids),
            }
            record = build_lod_record(
                tile_revision_id=request.tile_revision_id,
                appearance_revision_id=revision.appearance_revision_id,
                policy=policy_core,
                vertex_count=len(used),
                triangle_count=len(triangles),
                atlas_count=1,
                artifacts=references,
                primitives=tuple(primitives),
                rgbd_augmented_revision_id=(
                    request.rgbd_augmented_revision_id
                    if any(value.source_layer == "RGBD_AUGMENTED" for value in primitives)
                    else None
                ),
            )
            payloads.append(TileLodPayload(record, artifacts))
        return tuple(payloads)

    def _progress(self, state: str) -> None:
        if self._progress_hook is not None:
            self._progress_hook(state)
