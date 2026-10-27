"""Validate a paired Nav2 YAML/PGM map and its USER_PRIOR provenance."""
from __future__ import annotations

import json
from pathlib import Path
import math
from typing import Any

import yaml


def _pgm_dimensions(path: Path) -> tuple[int, int]:
    body = path.read_bytes()
    if not body.startswith(b"P5"):
        raise ValueError("Nav2 map image must be binary PGM P5")
    tokens: list[bytes] = []
    index = 2
    while index < len(body) and len(tokens) < 3:
        while index < len(body) and body[index] in b" \t\r\n":
            index += 1
        if index < len(body) and body[index] == ord("#"):
            newline = body.find(b"\n", index)
            index = len(body) if newline < 0 else newline + 1
            continue
        start = index
        while index < len(body) and body[index] not in b" \t\r\n":
            index += 1
        if index > start:
            tokens.append(body[start:index])
    if len(tokens) != 3:
        raise ValueError("Nav2 PGM header is incomplete")
    try:
        width, height, max_value = (int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError("Nav2 PGM header is invalid") from exc
    if width <= 0 or height <= 0 or max_value != 255:
        raise ValueError("Nav2 PGM dimensions or max value are invalid")
    return width, height


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is invalid")
    return result


def validate_nav2_map(map_yaml_path: str | Path) -> dict[str, Any]:
    yaml_path = Path(map_yaml_path).resolve(strict=False)
    if not yaml_path.is_file():
        raise ValueError("Nav2 map YAML is unavailable")
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("Nav2 map YAML is unreadable") from exc
    if not isinstance(config, dict):
        raise ValueError("Nav2 map YAML must be an object")
    image_name = config.get("image")
    if not isinstance(image_name, str) or not image_name or Path(image_name).is_absolute():
        raise ValueError("Nav2 map image must be a relative path")
    map_root = yaml_path.parent.resolve(strict=False)
    image_path = (map_root / image_name).resolve(strict=False)
    try:
        image_path.relative_to(map_root)
    except ValueError as exc:
        raise ValueError("Nav2 map image must stay inside map directory") from exc
    if not image_path.is_file():
        raise ValueError("Nav2 map image is unavailable")
    width, height = _pgm_dimensions(image_path)

    provenance_path = map_root / "map.provenance.json"
    if not provenance_path.is_file():
        raise ValueError("Nav2 map provenance sidecar is required")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Nav2 map provenance sidecar is unreadable") from exc
    if not isinstance(provenance, dict):
        raise ValueError("Nav2 map provenance sidecar is invalid")
    if provenance.get("schema_version") != 1:
        raise ValueError("Nav2 map provenance schema is unsupported")
    if provenance.get("provenance") != "USER_PRIOR":
        raise ValueError("Nav2 map provenance must be USER_PRIOR")
    if provenance.get("geometry_authority") is not False:
        raise ValueError("Nav2 map geometry_authority must be false")
    if provenance.get("coordinate_frame") not in {"SESSION_ODOM_LOCAL", "BUILDING_LOCAL"}:
        raise ValueError("Nav2 map coordinate_frame is unsupported")
    source_revision = provenance.get("source_revision")
    if isinstance(source_revision, bool) or not isinstance(source_revision, int) or source_revision <= 0:
        raise ValueError("Nav2 map source_revision is invalid")
    yaml_resolution = _number(config.get("resolution"), "Nav2 map resolution")
    sidecar_resolution = _number(provenance.get("resolution"), "Nav2 provenance resolution")
    if yaml_resolution <= 0 or not math.isclose(yaml_resolution, sidecar_resolution, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Nav2 map resolution does not match provenance")
    if provenance.get("width") != width or provenance.get("height") != height:
        raise ValueError("Nav2 map dimensions do not match provenance")
    return {
        "map_yaml": yaml_path,
        "image": image_path,
        "provenance": provenance["provenance"],
        "provenance_path": provenance_path,
        "map_id": provenance.get("map_id"),
        "coordinate_frame": provenance["coordinate_frame"],
        "provenance_label": provenance["provenance"],
        "geometry_authority": False,
        "source_revision": source_revision,
        "resolution": yaml_resolution,
        "width": width,
        "height": height,
    }
