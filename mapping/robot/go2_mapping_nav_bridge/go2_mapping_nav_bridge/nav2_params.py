"""Prepare a frame-consistent, user-owned Nav2 parameter overlay."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


_SUPPORTED_FRAMES = {"SESSION_ODOM_LOCAL", "BUILDING_LOCAL"}


def _parameters(payload: dict[str, Any], node_name: str) -> dict[str, Any]:
    node = payload.setdefault(node_name, {})
    if not isinstance(node, dict):
        raise ValueError(f"Nav2 parameters for {node_name} must be an object")
    parameters = node.setdefault("ros__parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"Nav2 ros__parameters for {node_name} must be an object")
    return parameters


def prepare_nav2_params(
    source_file: str | Path,
    map_frame: str,
    output_dir: str | Path,
) -> Path:
    """Create a temporary Nav2 parameter file whose global frame matches the map.

    The source profile is never changed. Only parameters that define the map's
    global frame are rewritten; local costmap and behavior frames remain in
    odom, preserving Nav2's rolling-window semantics.
    """

    if map_frame not in _SUPPORTED_FRAMES:
        raise ValueError(f"unsupported Nav2 map frame: {map_frame}")
    source = Path(source_file).resolve(strict=False)
    if not source.is_file():
        raise ValueError("Nav2 parameter source is unavailable")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("Nav2 parameter source is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("Nav2 parameter source must be an object")
    payload = deepcopy(payload)

    _parameters(payload, "map_server")["frame_id"] = map_frame
    _parameters(payload, "amcl")["global_frame_id"] = map_frame
    _parameters(payload, "bt_navigator")["global_frame"] = map_frame
    global_costmap = payload.setdefault("global_costmap", {})
    if not isinstance(global_costmap, dict):
        raise ValueError("Nav2 global_costmap parameters must be an object")
    nested = global_costmap.setdefault("global_costmap", {})
    if not isinstance(nested, dict):
        raise ValueError("Nav2 global_costmap node parameters must be an object")
    nested_parameters = nested.setdefault("ros__parameters", {})
    if not isinstance(nested_parameters, dict):
        raise ValueError("Nav2 global_costmap ros__parameters must be an object")
    nested_parameters["global_frame"] = map_frame

    destination = Path(output_dir).resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="nav2-params-", suffix=".yaml.partial", dir=destination
    )
    temporary = Path(temporary_name)
    final = destination / temporary.name.removesuffix(".partial")
    try:
        body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return final


__all__ = ["prepare_nav2_params"]
