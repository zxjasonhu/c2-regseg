"""Post-registration segmentation utilities: gap-filling and smoothing."""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def _find_nearest_label_bfs(
    mask: np.ndarray,
    z: int, y: int, x: int,
    max_radius: int = 10,
    min_neighbors: int = 8,
) -> int:
    """Find the best label near (z, y, x) via BFS with distance-weighted voting."""
    shape = mask.shape
    neighbors = []
    queue = deque([(z, y, x, 0)])
    visited = {(z, y, x)}

    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    while queue and len(neighbors) < min_neighbors:
        cz, cy, cx, dist = queue.popleft()
        if dist > max_radius:
            break

        if (cz, cy, cx) != (z, y, x):
            label = mask[cz, cy, cx]
            if label > 1:
                distance = np.sqrt((cz - z)**2 + (cy - y)**2 + (cx - x)**2)
                neighbors.append((label, distance))

        for dz, dy, dx in directions:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]
                    and (nz, ny, nx) not in visited):
                queue.append((nz, ny, nx, dist + 1))
                visited.add((nz, ny, nx))

    if not neighbors:
        return 0

    weights: dict[int, float] = {}
    for label, dist in neighbors:
        w = 1.0 / (dist**2 + 1e-6)
        weights[label] = weights.get(label, 0) + w

    return int(max(weights, key=weights.get))


def fill_mask_with_subsegments(
    binary_mask: np.ndarray,
    subsegment_mask: np.ndarray,
    max_radius: int = 10,
    min_neighbors: int = 8,
) -> np.ndarray:
    """Fill gaps in *subsegment_mask* within the *binary_mask* region via BFS.

    Voxels where ``binary_mask == 1`` but ``subsegment_mask <= 1`` are assigned
    labels from their nearest labelled neighbours.
    """
    filled = np.zeros_like(binary_mask)

    # Direct copy where labels already exist
    direct = (binary_mask == 1) & (subsegment_mask > 1)
    filled[direct] = subsegment_mask[direct]

    # BFS for unlabelled foreground voxels
    need = (binary_mask == 1) & (subsegment_mask <= 1)
    coords = np.argwhere(need)
    total = len(coords)
    logger.info("Filling %d voxels via BFS", total)

    for i, (z, y, x) in enumerate(coords):
        filled[z, y, x] = _find_nearest_label_bfs(
            subsegment_mask, z, y, x,
            max_radius=max_radius,
            min_neighbors=min_neighbors,
        )
        if (i + 1) % 5000 == 0 or i == total - 1:
            logger.debug("Filled %d / %d (%.1f%%)", i + 1, total, 100 * (i + 1) / total)

    return filled
