"""Voxel-level similarity metrics for registration quality assessment."""

import numpy as np


def dice_score(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Compute binary Dice overlap between a target and warped atlas mask."""
    a = np.asarray(mask_a) > 0.9
    b = np.asarray(mask_b) > 0.9
    denominator = int(a.sum() + b.sum())
    return float(2 * np.sum(a & b) / denominator) if denominator else 0.0
