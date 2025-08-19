"""Paper PGIRP region proposal and prediction from Grad-CAM + segmentation."""

import numpy as np
from scipy import ndimage


def region_proposal_iterative(
    cam: np.ndarray,
    center_relative_threshold: float = 0.7,
    region_relative_threshold: float = 0.5,
    min_region_size: int = 750,
    max_regions: int = 20,
    segment_mask: np.ndarray | None = None,
) -> list[dict]:
    """Iteratively grow regions from highest-intensity peaks.

    This follows the legacy notebook behavior used in the original
    ``c2-super-classifier`` evaluation: the stopping threshold is fixed from the
    initial maximum, while the region-growing threshold is recomputed from the
    current hottest peak at each iteration.
    """
    if cam is None or cam.sum() == 0:
        return []

    work = cam.copy()
    center_thr = work.max() * center_relative_threshold

    structure = np.ones((3, 3, 3), dtype=bool)
    regions = []
    rid = 1

    while len(regions) < max_regions:
        current_max = work.max()
        if current_max < center_thr:
            break

        seed = np.unravel_index(np.argmax(work), work.shape)
        region_thr = current_max * region_relative_threshold
        candidate = work > region_thr
        labeled, n = ndimage.label(candidate, structure=structure)
        if n == 0:
            break

        region_label = labeled[seed]
        rmask = labeled == region_label
        work[rmask] = 0

        size = rmask.sum()
        if size >= min_region_size:
            if segment_mask is not None:
                rmask = rmask & (segment_mask > 0)
            vals = cam[rmask]
            if vals.size == 0:
                continue
            regions.append({
                "id": rid,
                "mask": rmask,
                "max_value": float(vals.max()),
                "avg_value": float(vals.mean()),
                "max_coords": seed,
                "size": int(size),
            })
            rid += 1

    return regions


# ---------------------------------------------------------------------------
# Level analysis within a proposed region
# ---------------------------------------------------------------------------

def _analyze_region_levels(
    region: dict,
    cam: np.ndarray,
    segment_mask: np.ndarray,
    level_ratios: list[float],
) -> list[dict]:
    """Decompose a region into intensity shells and analyse label overlap."""
    rmask = region["mask"]
    rmax = region["max_value"]
    ratios = sorted([r for r in level_ratios if 0 < r <= 1.0], reverse=True)

    if rmax <= 0 or not ratios:
        return []

    cam_vals = cam[rmask]
    seg_vals = segment_mask[rmask]
    total_intensity = cam_vals.sum() or 1e-9

    prev_above = np.zeros_like(cam_vals, dtype=bool)
    levels = []

    for i, ratio in enumerate(ratios):
        thr = ratio * rmax
        above = cam_vals > thr
        shell = above & ~prev_above

        shell_cam = cam_vals[shell]
        shell_seg = seg_vals[shell]
        shell_total = shell_cam.sum()

        assoc = {}
        if shell.any():
            ulabels, counts = np.unique(shell_seg[shell_seg > 0], return_counts=True)
            for lbl, cnt in zip(ulabels, counts):
                lbl_mask = shell_seg == lbl
                intensity = shell_cam[lbl_mask].sum()
                assoc[int(lbl)] = {
                    "intensity_sum": float(intensity),
                    "intensity_fraction_of_shell": float(intensity / shell_total) if shell_total > 0 else 0,
                    "intensity_mean": float(intensity / cnt),
                    "voxel_count": int(cnt),
                }

        levels.append({
            "level_index": i,
            "threshold_ratio": ratio,
            "threshold_value": thr,
            "shell_voxel_count": int(shell.sum()),
            "shell_total_intensity": float(shell_total),
            "shell_intensity_ratio_of_region": float(shell_total / total_intensity),
            "label_associations": assoc,
        })
        prev_above = above

    return levels


# ---------------------------------------------------------------------------
# Main prediction entry-point
# ---------------------------------------------------------------------------

def predict_with_levels(
    segment_mask: np.ndarray,
    cam: np.ndarray,
    level_threshold_ratios: list[float] | None = None,
    core_level_min_intensity_fraction: float = 0.1,
    region_proposal_params: dict | None = None,
) -> dict:
    """Predict labels using the paper's iterative PGIRP intensity analysis.

    Parameters
    ----------
    segment_mask : np.ndarray
        3-D registered segmentation mask.
    cam : np.ndarray
        3-D GradCAM activation volume.
    level_threshold_ratios : list[float]
        Intensity level thresholds (descending ratios of region max).
    core_level_min_intensity_fraction : float
        Minimum intensity fraction in the core shell for a label to be predicted.
    region_proposal_params : dict
        Keyword arguments forwarded to the iterative region proposal function.

    Returns
    -------
    dict with ``"predictions"`` (list[int]) and ``"details"`` (list[dict]).
    """
    if level_threshold_ratios is None:
        level_threshold_ratios = [0.4]
    if region_proposal_params is None:
        region_proposal_params = {}

    regions = region_proposal_iterative(
        cam, segment_mask=segment_mask, **region_proposal_params
    )
    if not regions:
        return {"predictions": [], "details": []}

    detailed = []
    for region in regions:
        levels = _analyze_region_levels(region, cam, segment_mask, level_threshold_ratios)
        region["level_analysis"] = levels
        region["predicted_labels_from_core"] = []

        if levels and levels[0]["shell_voxel_count"] > 0:
            core = levels[0]
            candidates = [
                (lbl, info["intensity_sum"])
                for lbl, info in core["label_associations"].items()
                if info["intensity_fraction_of_shell"] >= core_level_min_intensity_fraction
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            region["predicted_labels_from_core"] = [lbl for lbl, _ in candidates]

        detailed.append(region)

    detailed.sort(key=lambda r: r["max_value"], reverse=True)

    predictions = []
    for r in detailed:
        predictions.extend(r.get("predicted_labels_from_core", []))
    # Deduplicate while preserving order
    predictions = list(dict.fromkeys(predictions))

    return {"predictions": predictions, "details": detailed}

