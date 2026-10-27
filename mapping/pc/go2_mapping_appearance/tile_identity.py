"""Acyclic content identities for immutable tile appearance generations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ATLAS_NAME_RE = re.compile(r"^atlas-(0|[1-9][0-9]*)\.png$", re.ASCII)
_ARTIFACT_NAMES = {"render-vertices.bin", "render-indices.bin", "render-uvs.bin", "vertex-color.bin"}
_FORBIDDEN_IDENTITY_FIELDS = {"manifest_sha256", "lod_id", "appearance_revision_id"}
_REQUIRED_COVERAGE_PPM = 800_000
_REQUIRED_INDEPENDENT_VIEWS = 2


@dataclass(frozen=True)
class DeliveryAuthorization:
    measured_surface_coverage_ppm: int
    minimum_independent_view_count: int
    required_coverage_ppm: int
    required_independent_view_count: int
    delivery_authorized: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class TileAppearanceRevision:
    revision_core: Mapping[str, Any]
    appearance_revision_id: str
    delivery_authorization: DeliveryAuthorization


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if self.name in _FORBIDDEN_IDENTITY_FIELDS:
            raise ValueError("artifact name is a forbidden self or manifest hash field")
        if self.name not in _ARTIFACT_NAMES and _ATLAS_NAME_RE.fullmatch(self.name) is None:
            raise ValueError("artifact name is not in the tile appearance allowlist")
        _sha256(self.sha256, "artifact sha256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ValueError("artifact byte count must be a non-negative integer")

    def core(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class PrimitiveRecord:
    primitive_id: int
    material: str
    atlas_index: int
    index_offset: int
    index_count: int
    source_layer: str
    source_triangle_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("primitive_id", "atlas_index", "index_offset", "index_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.material not in {"VALIDATED_UV_ATLAS", "BASE_MATERIAL"}:
            raise ValueError("primitive material is invalid")
        if self.source_layer not in {"LIDAR_MEASURED", "RGBD_AUGMENTED"}:
            raise ValueError("primitive source layer is invalid")
        if self.index_count == 0 or self.index_count % 3:
            raise ValueError("primitive index_count must be a positive multiple of three")
        source_ids = tuple(self.source_triangle_ids)
        if len(source_ids) != self.index_count // 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in source_ids
        ):
            raise ValueError("primitive source triangle IDs must match triangle order")
        if self.material == "BASE_MATERIAL" and self.atlas_index != 65535:
            raise ValueError("BASE_MATERIAL primitive requires atlas sentinel 65535")
        if self.material == "VALIDATED_UV_ATLAS" and self.atlas_index == 65535:
            raise ValueError("VALIDATED_UV_ATLAS primitive cannot use the base sentinel")
        object.__setattr__(self, "source_triangle_ids", source_ids)

    def core(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "material": self.material,
            "atlas_index": self.atlas_index,
            "index_offset": self.index_offset,
            "index_count": self.index_count,
            "source_layer": self.source_layer,
            "source_triangle_ids": list(self.source_triangle_ids),
        }


@dataclass(frozen=True)
class LodRecord:
    lod_id: str
    core: Mapping[str, Any]
    policy: Mapping[str, Any]
    artifacts: tuple[ArtifactRef, ...]
    primitives: tuple[PrimitiveRecord, ...]


def tile_appearance_revision(
    *,
    tile_revision_id: str,
    geometry_revision_id: str,
    source_generation_id: str,
    calibration_id: str,
    calibration_sha256: str,
    physical_depth_validation_revision: str,
    pose_revision_id: str,
    model_revision_id: str | None,
    assistance_config_sha256: str | None,
    source_frame_sha256: Sequence[str],
    appearance_config_sha256: str,
    charting: Mapping[str, Any],
    atlas_density_policy: Mapping[str, Any],
    padding_policy: Mapping[str, Any],
    baker_algorithm_version: str,
    rgbd_augmented_revision_id: str | None,
    contains_augmented_primitives: bool,
    measured_surface_coverage_ppm: int,
    minimum_independent_view_count: int,
) -> TileAppearanceRevision:
    frames = tuple(sorted(_sha256(value, "source frame sha256") for value in source_frame_sha256))
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("source_frame_sha256 must contain unique digests")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError("calibration_id must be non-empty")
    if not isinstance(baker_algorithm_version, str) or not baker_algorithm_version:
        raise ValueError("baker_algorithm_version must be non-empty")
    if contains_augmented_primitives is not True and contains_augmented_primitives is not False:
        raise ValueError("contains_augmented_primitives must be boolean")
    if contains_augmented_primitives != (rgbd_augmented_revision_id is not None):
        raise ValueError("rgbd_augmented_revision_id is required if and only if augmentation is present")
    augmentation = None if rgbd_augmented_revision_id is None else _sha256(
        rgbd_augmented_revision_id, "rgbd_augmented_revision_id"
    )
    if (model_revision_id is None) != (assistance_config_sha256 is None):
        raise ValueError("model and assistance configuration revisions must be supplied together")
    model = None if model_revision_id is None else _sha256(model_revision_id, "model_revision_id")
    assistance = None if assistance_config_sha256 is None else _sha256(
        assistance_config_sha256, "assistance_config_sha256"
    )
    coverage = _bounded_integer(measured_surface_coverage_ppm, 0, 1_000_000, "coverage ppm")
    views = _bounded_integer(minimum_independent_view_count, 0, 2**31 - 1, "independent view count")
    blockers: list[str] = []
    if coverage < _REQUIRED_COVERAGE_PPM:
        blockers.append("MEASURED_SURFACE_COVERAGE_REQUIRED")
    if views < _REQUIRED_INDEPENDENT_VIEWS:
        blockers.append("INDEPENDENT_COLOR_VIEWS_REQUIRED")
    authorization = DeliveryAuthorization(
        measured_surface_coverage_ppm=coverage,
        minimum_independent_view_count=views,
        required_coverage_ppm=_REQUIRED_COVERAGE_PPM,
        required_independent_view_count=_REQUIRED_INDEPENDENT_VIEWS,
        delivery_authorized=not blockers,
        blockers=tuple(blockers),
    )
    core: dict[str, Any] = {
        "schema_version": 1,
        "tile_revision_id": _sha256(tile_revision_id, "tile_revision_id"),
        "geometry_revision_id": _sha256(geometry_revision_id, "geometry_revision_id"),
        "source_generation_id": _sha256(source_generation_id, "source_generation_id"),
        "calibration_id": calibration_id,
        "calibration_sha256": _sha256(calibration_sha256, "calibration_sha256"),
        "physical_depth_validation_revision": _sha256(
            physical_depth_validation_revision, "physical_depth_validation_revision"
        ),
        "pose_revision_id": _sha256(pose_revision_id, "pose_revision_id"),
        "model_revision_id": model,
        "assistance_config_sha256": assistance,
        "source_frame_sha256": list(frames),
        "appearance_config_sha256": _sha256(appearance_config_sha256, "appearance_config_sha256"),
        "charting": _mapping(charting, "charting"),
        "atlas_density_policy": _mapping(atlas_density_policy, "atlas_density_policy"),
        "padding_policy": _mapping(padding_policy, "padding_policy"),
        "baker_algorithm_version": baker_algorithm_version,
        "rgbd_augmented_revision_id": augmentation,
        "delivery_evidence": {
            "measured_surface_coverage_ppm": coverage,
            "minimum_independent_view_count": views,
            "required_coverage_ppm": _REQUIRED_COVERAGE_PPM,
            "required_independent_view_count": _REQUIRED_INDEPENDENT_VIEWS,
        },
    }
    return TileAppearanceRevision(core, canonical_sha256(core), authorization)


def build_lod_record(
    *,
    tile_revision_id: str,
    appearance_revision_id: str,
    policy: Mapping[str, Any],
    vertex_count: int,
    triangle_count: int,
    atlas_count: int,
    artifacts: Sequence[ArtifactRef],
    primitives: Sequence[PrimitiveRecord],
    rgbd_augmented_revision_id: str | None,
) -> LodRecord:
    vertices = _bounded_integer(vertex_count, 0, 2**32 - 1, "vertex_count")
    triangles = _bounded_integer(triangle_count, 0, 2**32 - 1, "triangle_count")
    atlases = _bounded_integer(atlas_count, 0, 65535, "atlas_count")
    artifact_records = tuple(sorted(artifacts, key=lambda value: value.name))
    if any(not isinstance(value, ArtifactRef) for value in artifact_records):
        raise TypeError("artifacts must contain ArtifactRef records")
    if len({value.name for value in artifact_records}) != len(artifact_records):
        raise ValueError("artifact names must be unique")
    primitive_records = tuple(primitives)
    if any(not isinstance(value, PrimitiveRecord) for value in primitive_records):
        raise TypeError("primitives must contain PrimitiveRecord records")
    if tuple(value.primitive_id for value in primitive_records) != tuple(range(len(primitive_records))):
        raise ValueError("primitive IDs must be consecutive and ordered")
    expected_offset = 0
    for primitive in primitive_records:
        if primitive.index_offset != expected_offset:
            raise ValueError("primitive ranges must be contiguous and non-overlapping")
        expected_offset += primitive.index_count
        if primitive.material == "VALIDATED_UV_ATLAS" and primitive.atlas_index >= atlases:
            raise ValueError("primitive atlas index is out of range")
    if expected_offset != triangles * 3:
        raise ValueError("primitive ranges must cover every render index exactly once")
    augmented = any(value.source_layer == "RGBD_AUGMENTED" for value in primitive_records)
    if augmented != (rgbd_augmented_revision_id is not None):
        raise ValueError("LOD augmentation primitives and rgbd_augmented_revision_id disagree")
    augmentation = None if rgbd_augmented_revision_id is None else _sha256(
        rgbd_augmented_revision_id, "rgbd_augmented_revision_id"
    )
    core: dict[str, Any] = {
        "schema_version": 1,
        "tile_revision_id": _sha256(tile_revision_id, "tile_revision_id"),
        "appearance_revision_id": _sha256(appearance_revision_id, "appearance_revision_id"),
        "policy": _mapping(policy, "LOD policy"),
        "vertex_count": vertices,
        "triangle_count": triangles,
        "atlas_count": atlases,
        "artifacts": [value.core() for value in artifact_records],
        "primitives": [value.core() for value in primitive_records],
        "rgbd_augmented_revision_id": augmentation,
    }
    return LodRecord(
        lod_id=canonical_sha256(core),
        core=core,
        policy=core["policy"],
        artifacts=artifact_records,
        primitives=primitive_records,
    )


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    result = dict(value)
    if _FORBIDDEN_IDENTITY_FIELDS & result.keys():
        raise ValueError(f"{name} contains a forbidden self-reference field")
    canonical_sha256(result)
    return result


def _bounded_integer(value: int, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value
