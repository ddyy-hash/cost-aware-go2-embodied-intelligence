"""Immutable per-tile RGB8 validity previews bound to exact tile order."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import threading
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from mapping.pc.go2_mapping_rgbd.texture import VertexColor

from .tile_identity import TileAppearanceRevision


_MAGIC = b"G2TVCOL1"
_HEADER = struct.Struct("<8sHHI32s32s")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


@dataclass(frozen=True)
class TileVertexColorSidecar:
    tile_revision_id: str
    appearance_revision_id: str
    manifest: Mapping[str, Any]
    binary: bytes


@dataclass(frozen=True)
class ParsedTileVertexColor:
    vertex_count: int
    tile_revision_id: str
    appearance_revision_id: str
    rgb: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
        validity = np.asarray(self.validity, dtype=np.uint8).reshape(-1)
        if len(rgb) != self.vertex_count or len(validity) != self.vertex_count:
            raise ValueError("tile vertex-color arrays do not match vertex_count")
        rgb.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True)
class PublishedTileVertexColor:
    tile_revision_id: str
    appearance_revision_id: str
    manifest: Mapping[str, Any]
    root_path: Path
    manifest_path: Path
    binary_path: Path


def build_tile_vertex_color_sidecar(
    *,
    revision: TileAppearanceRevision,
    vertex_colors: Sequence[VertexColor],
) -> TileVertexColorSidecar:
    if not isinstance(revision, TileAppearanceRevision):
        raise TypeError("revision must be a TileAppearanceRevision")
    tile = _sha256(revision.revision_core.get("tile_revision_id"), "tile revision")
    appearance = _sha256(revision.appearance_revision_id, "appearance revision")
    colors = tuple(vertex_colors)
    if not colors:
        raise ValueError("vertex_colors must contain the exact tile vertex order")
    source_frames = set(revision.revision_core.get("source_frame_sha256", ()))
    records = bytearray()
    colored = 0
    for index, color in enumerate(colors):
        if not isinstance(color, VertexColor):
            raise TypeError("vertex_colors must contain VertexColor records")
        if color.vertex_id != f"{tile}:{index}":
            raise ValueError("vertex_colors must follow exact G2TM vertex order")
        if any(_sha256(value, "source frame") not in source_frames for value in color.source_frame_sha256):
            raise ValueError("tile preview source frame is outside the appearance revision")
        linear = np.asarray(color.mean_linear_rgb, dtype=np.float64)
        if linear.shape != (3,) or not np.all(np.isfinite(linear)):
            raise ValueError("mean_linear_rgb must contain three finite values")
        if color.accepted_for_texture:
            if np.any(linear < 0.0) or np.any(linear > 1.0):
                raise ValueError("accepted tile preview RGB must be normalized")
            records.extend(_linear_rgb8(linear))
            records.append(255)
            colored += 1
        else:
            records.extend(b"\0\0\0\0")
    binary = _HEADER.pack(
        _MAGIC,
        1,
        _HEADER.size,
        len(colors),
        bytes.fromhex(tile),
        bytes.fromhex(appearance),
    ) + bytes(records)
    content_sha = hashlib.sha256(binary).hexdigest()
    authorized = revision.delivery_authorization.delivery_authorized
    manifest = {
        "schema_version": 1,
        "tile_revision_id": tile,
        "source_generation_id": revision.revision_core["source_generation_id"],
        "appearance_revision_id": appearance,
        "vertex_order": "G2TM_V1_EXACT_INDEX_ORDER",
        "vertex_count": len(colors),
        "colored_vertex_count": colored,
        "state": "TILE_VERTEX_COLOR_READY" if authorized else "DIAGNOSTIC_VERTEX_COLOR_READY",
        "delivery_authorized": authorized,
        "building_authorized": False,
        "export_authorized": False,
        "blockers": list(revision.delivery_authorization.blockers),
        "vertex_color": {
            "format": "RGB8_VALIDITY8_TILE_V1",
            "sha256": content_sha,
            "bytes": len(binary),
        },
    }
    return TileVertexColorSidecar(tile, appearance, manifest, binary)


def parse_tile_vertex_color_v1(
    payload: bytes,
    *,
    expected_tile_revision_id: str | None = None,
    expected_appearance_revision_id: str | None = None,
) -> ParsedTileVertexColor:
    value = bytes(payload)
    if len(value) < _HEADER.size:
        raise ValueError("tile vertex-color payload is truncated")
    magic, version, header_bytes, count, tile_raw, appearance_raw = _HEADER.unpack_from(value)
    if magic != _MAGIC or version != 1 or header_bytes != _HEADER.size:
        raise ValueError("tile vertex-color header is invalid")
    if count <= 0 or len(value) != _HEADER.size + count * 4:
        raise ValueError("tile vertex-color length does not match vertex_count")
    tile = tile_raw.hex()
    appearance = appearance_raw.hex()
    if expected_tile_revision_id is not None and tile != _sha256(expected_tile_revision_id, "expected tile revision"):
        raise ValueError("tile vertex-color tile revision does not match")
    if expected_appearance_revision_id is not None and appearance != _sha256(
        expected_appearance_revision_id, "expected appearance revision"
    ):
        raise ValueError("tile vertex-color appearance revision does not match")
    records = np.frombuffer(value, dtype=np.uint8, offset=_HEADER.size).reshape(count, 4)
    validity = np.array(records[:, 3], copy=True)
    rgb = np.array(records[:, :3], copy=True)
    if np.any((validity != 0) & (validity != 255)):
        raise ValueError("tile vertex-color validity must be binary")
    if np.any(rgb[validity == 0] != 0):
        raise ValueError("invalid tile vertex-color records must have zero RGB")
    return ParsedTileVertexColor(count, tile, appearance, rgb, validity)


class TileVertexColorStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "appearance-previews"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        for candidate in self.root.rglob("*.partial"):
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)

    def publish(self, sidecar: TileVertexColorSidecar) -> PublishedTileVertexColor:
        if not isinstance(sidecar, TileVertexColorSidecar):
            raise TypeError("sidecar must be a TileVertexColorSidecar")
        tile = _sha256(sidecar.tile_revision_id, "tile revision")
        appearance = _sha256(sidecar.appearance_revision_id, "appearance revision")
        manifest_bytes = _canonical_bytes(sidecar.manifest)
        target = self.root / tile / appearance
        with self._lock:
            if target.exists():
                published = self.resolve(tile, appearance)
                if published.manifest_path.read_bytes() != manifest_bytes or published.binary_path.read_bytes() != sidecar.binary:
                    raise ValueError("existing tile preview conflicts with immutable content")
                return published
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.parent / f".{appearance}.{uuid.uuid4().hex}.partial"
            staging.mkdir()
            try:
                _write_fsync(staging / "manifest.json", manifest_bytes)
                _write_fsync(staging / "vertex-color.bin", sidecar.binary)
                _fsync_directory(staging)
                self._load(staging, tile, appearance)
                os.replace(staging, target)
                _fsync_directory(target.parent)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        return self.resolve(tile, appearance)

    def resolve(self, tile_revision_id: str, appearance_revision_id: str) -> PublishedTileVertexColor:
        tile = _sha256(tile_revision_id, "tile revision")
        appearance = _sha256(appearance_revision_id, "appearance revision")
        target = self.root / tile / appearance
        if not target.is_dir():
            raise FileNotFoundError("tile vertex-color preview is unavailable")
        return self._load(target, tile, appearance)

    @staticmethod
    def _load(target: Path, tile: str, appearance: str) -> PublishedTileVertexColor:
        try:
            manifest_bytes = (target / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            binary = (target / "vertex-color.bin").read_bytes()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("tile vertex-color preview is incomplete or invalid") from exc
        if manifest_bytes != _canonical_bytes(manifest):
            raise ValueError("tile vertex-color manifest is not canonical")
        if manifest.get("schema_version") != 1 or manifest.get("tile_revision_id") != tile:
            raise ValueError("tile vertex-color manifest tile identity is invalid")
        if manifest.get("appearance_revision_id") != appearance:
            raise ValueError("tile vertex-color manifest appearance identity is invalid")
        parsed = parse_tile_vertex_color_v1(
            binary,
            expected_tile_revision_id=tile,
            expected_appearance_revision_id=appearance,
        )
        reference = manifest.get("vertex_color")
        if (
            not isinstance(reference, dict)
            or reference.get("format") != "RGB8_VALIDITY8_TILE_V1"
            or reference.get("bytes") != len(binary)
            or reference.get("sha256") != hashlib.sha256(binary).hexdigest()
            or manifest.get("vertex_count") != parsed.vertex_count
        ):
            raise ValueError("tile vertex-color content verification failed")
        return PublishedTileVertexColor(
            tile, appearance, manifest, target, target / "manifest.json", target / "vertex-color.bin"
        )


def _linear_rgb8(linear: np.ndarray) -> bytes:
    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
