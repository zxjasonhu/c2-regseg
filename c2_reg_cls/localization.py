"""Paper localization helpers: CAM enhancement, PGIRP, and axis alignment."""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np

from .heatmap.prediction import predict_with_levels


def enhance_cam(cam: np.ndarray) -> np.ndarray:
    """Normalize and apply the paper's quadratic CAM enhancement."""
    result = np.asarray(cam, dtype=np.float32).copy()
    result -= result.min()
    maximum = result.max()
    if maximum > 0:
        result /= maximum
    result = result**2 + result
    maximum = result.max()
    if maximum > 0:
        result /= maximum
    return result


def _ordered_unique(labels: list[int]) -> list[int]:
    return list(dict.fromkeys(int(label) for label in labels))


def build_prediction_params(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the PGIRP parameters used in the paper."""
    return {
        "center_relative_threshold": float(config.get("center_relative_threshold", 0.3825)),
        "region_relative_threshold": float(config.get("region_relative_threshold", 0.22)),
        "min_region_size": int(config.get("min_region_size", 120)),
        "max_regions": int(config.get("max_regions", 4)),
        "level_ratio": float(config.get("level_ratio", 0.25)),
        "core_frac": float(config.get("core_frac", 0.03)),
    }


def _predict(
    segment_mask: np.ndarray | None,
    cam: np.ndarray | None,
    params: dict[str, Any],
) -> list[int]:
    if segment_mask is None or cam is None or not np.any(cam):
        return []

    result = predict_with_levels(
        segment_mask=np.asarray(segment_mask),
        cam=np.asarray(cam),
        level_threshold_ratios=[params["level_ratio"]],
        core_level_min_intensity_fraction=params["core_frac"],
        region_proposal_params={
            "center_relative_threshold": params["center_relative_threshold"],
            "region_relative_threshold": params["region_relative_threshold"],
            "min_region_size": params["min_region_size"],
            "max_regions": params["max_regions"],
        },
    )
    return _ordered_unique(result["predictions"])


def predict_default_pipeline(
    segment_mask: np.ndarray | None,
    cam: np.ndarray | None,
    params: dict[str, Any],
) -> list[int]:
    """Run the paper's iterative PGIRP with non-collapsed atlas labels."""
    return _predict(segment_mask, cam, params)


def align_side_image_to_nii(
    nii_image: nib.Nifti1Image,
    side_image: np.ndarray | None,
) -> np.ndarray | None:
    """Convert a model ``(Z,Y,X)`` volume to the NIfTI mask's axis order."""
    if side_image is None:
        return None
    side_image = np.asarray(side_image)
    main_shape = tuple(int(value) for value in nii_image.shape[:3])
    if tuple(side_image.shape) == main_shape:
        return side_image
    if side_image.ndim == 3:
        legacy = side_image[::-1].transpose(2, 1, 0)
        if tuple(legacy.shape) == main_shape:
            return legacy
        transposed = side_image.transpose(2, 1, 0)
        if tuple(transposed.shape) == main_shape:
            return transposed
    raise ValueError(
        f"Side image shape {tuple(side_image.shape)} cannot align to NIfTI shape {main_shape}"
    )
