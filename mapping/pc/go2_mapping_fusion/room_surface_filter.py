"""Conservative room-surface filtering for delayed observed meshes.

The filter follows common point-cloud practice: radius/statistical support
first, then connected-component retention. It never fills holes or invents
surfaces; unsupported returns are simply excluded from meshing.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def filter_room_surface_points(
    positions: np.ndarray,
    *,
    support_radius_m: float = 0.05,
    min_neighbors: int = 3,
    min_component_size: int = 8,
    plane_residual_m: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not len(values):
        return values, np.empty(0, dtype=np.int64)
    if support_radius_m <= 0 or min_neighbors < 1 or min_component_size < 1 or plane_residual_m <= 0:
        raise ValueError("room surface filter parameters must be positive")
    tree = cKDTree(values)
    neighbors = tree.query_ball_point(values, support_radius_m)
    supported = np.asarray([len(item) >= min_neighbors for item in neighbors], dtype=bool)
    # Reject dense but non-surface spray using a local PCA plane residual. Keep
    # very small neighborhoods for object details; component support decides
    # whether those survive later.
    for index in np.flatnonzero(supported):
        local = values[neighbors[index]]
        if len(local) < 6:
            continue
        centred = local - local.mean(axis=0)
        _, singular, vectors = np.linalg.svd(centred, full_matrices=False)
        normal = vectors[-1]
        residual = float(np.percentile(np.abs(centred @ normal), 90))
        if residual > plane_residual_m and singular[-1] > singular[0] * 0.35:
            supported[index] = False
    indices = np.flatnonzero(supported)
    if not len(indices):
        return np.empty((0, 3), dtype=np.float64), indices
    # Components are built only over supported points; sparse wave tails do
    # not bridge into dense walls unless they have local support themselves.
    supported_tree = cKDTree(values[indices])
    adjacency = supported_tree.query_ball_point(values[indices], support_radius_m * 1.25)
    seen = np.zeros(len(indices), dtype=bool)
    keep_local: list[int] = []
    for start in range(len(indices)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for nxt in adjacency[current]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        if len(component) >= min_component_size:
            keep_local.extend(component)
    keep = indices[np.asarray(sorted(keep_local), dtype=np.int64)]
    return values[keep], keep


def filter_static_surface_points(
    positions: np.ndarray,
    *,
    support_radius_m: float = 0.05,
    min_neighbors: int = 3,
    min_component_size: int = 8,
    plane_residual_m: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return filtered room points plus auditable rejection counts.

    This is the offline analogue of ERASOR/Removert static-support filtering:
    only locally supported, surface-like components survive. It never fills
    free space or performs completion.
    """
    values = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    filtered, keep = filter_room_surface_points(
        values,
        support_radius_m=support_radius_m,
        min_neighbors=min_neighbors,
        min_component_size=min_component_size,
        plane_residual_m=plane_residual_m,
    )
    kept = np.zeros(len(values), dtype=bool)
    kept[keep] = True
    return filtered, keep, {
        "input_points": int(len(values)),
        "kept_points": int(len(filtered)),
        "rejected_points": int(len(values) - len(filtered)),
    }


__all__ = ["filter_room_surface_points"]
