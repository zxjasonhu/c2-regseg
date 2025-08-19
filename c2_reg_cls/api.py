"""Small, stable APIs for the reviewer-facing C2 localization pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from .config import LABEL_TO_ANATOMICAL_GROUP, REGION_LABELS
from .localization import (
    align_side_image_to_nii,
    build_prediction_params,
    enhance_cam,
    predict_default_pipeline,
)
from .heatmap import aggregate_heatmaps
from .pipeline_config import project_root
from .registration import (
    fill_mask_with_subsegments,
    load_reference_masks,
    normalize_mask,
    register_subsegment_mask_multi,
    restore_labels_to_original_space,
)

def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path against the release root."""
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root() / candidate).resolve()


def default_atlas_paths() -> tuple[Path, ...]:
    """Return the three de-identified, expert-annotated atlas paths."""
    atlas_dir = project_root() / "resources" / "atlases"
    paths = tuple(sorted(atlas_dir.glob("atlas_*_embedded.npz")))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected three embedded atlases in {atlas_dir}; found {len(paths)}")
    return paths


def load_prediction_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the paper-tuned PGIRP parameters."""
    config_path = resolve_project_path(
        path or "configs/paper_tuned_multiatlas_pgirp_params.json"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return dict(config.get("prediction", config))


def anatomical_groups_for_labels(labels: Sequence[int]) -> list[str]:
    """Map ranked subregion labels to unique paper groups in first-hit order."""
    groups = (
        LABEL_TO_ANATOMICAL_GROUP.get(int(label))
        for label in labels
    )
    return list(dict.fromkeys(group for group in groups if group is not None))


def aggregate_model_cams(
    cam_paths: Sequence[str | Path],
    target_shape: Sequence[int],
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Load and combine CAMs, optionally masking them to C2 before normalization."""
    paths = [str(resolve_project_path(path)) for path in cam_paths]
    shape = tuple(int(value) for value in target_shape)
    return aggregate_heatmaps(paths, shape, mask=mask)


def register_c2_subregions(
    c2_mask: str | Path | nib.Nifti1Image,
    *,
    atlas_paths: Sequence[str | Path] | None = None,
    scan: np.ndarray | None = None,
    cam: np.ndarray | None = None,
    output_shape: Sequence[int] = (96, 96, 96),
    threshold: float = 0.3,
    fusion: str = "uniform",
    transform_type: str = "SyNAggro",
    bfs_radius: int = 10,
    bfs_min_neighbors: int = 8,
) -> dict[str, np.ndarray]:
    """Register the expert atlases to one binary C2 mask.

    ``scan`` must use the NIfTI axis order ``(X,Y,Z)``. ``cam`` may be either
    NIfTI order or the model output order ``(Z,Y,X)``; it is aligned safely.
    The returned arrays all use the normalized ``(Z,Y,X)`` pipeline space.
    """
    nii_mask = nib.load(str(resolve_project_path(c2_mask))) if isinstance(c2_mask, (str, Path)) else c2_mask
    if not isinstance(nii_mask, nib.spatialimages.SpatialImage):
        raise TypeError("c2_mask must be a NIfTI path or nibabel image")

    shape = tuple(int(value) for value in output_shape)
    if len(shape) != 3:
        raise ValueError("output_shape must have three integers")
    aligned_cam = align_side_image_to_nii(nii_mask, cam)
    mask_emb, scan_emb = normalize_mask(nii_mask, output_shape=shape, side_image=scan)
    _, cam_emb = normalize_mask(nii_mask, output_shape=shape, side_image=aligned_cam)

    mask_emb = mask_emb.transpose(2, 1, 0)[:, ::-1, :]
    result: dict[str, np.ndarray] = {"mask": mask_emb}
    if scan_emb is not None:
        result["scan"] = scan_emb.transpose(2, 1, 0)[:, ::-1, :]
    if cam_emb is not None:
        result["cam"] = cam_emb.transpose(2, 1, 0)[:, ::-1, :]

    selected = atlas_paths or default_atlas_paths()
    references = load_reference_masks([str(resolve_project_path(path)) for path in selected])
    references = [reference.transpose(0, 3, 2, 1) for reference in references]
    registered = register_subsegment_mask_multi(
        references,
        mask_emb,
        threshold=float(threshold),
        fusion=fusion,
        transform_type=transform_type,
    )
    result["reg"] = fill_mask_with_subsegments(
        mask_emb,
        registered,
        max_radius=int(bfs_radius),
        min_neighbors=int(bfs_min_neighbors),
    )
    return result


def restore_c2_subregions_to_original(
    c2_mask: str | Path | nib.Nifti1Image,
    registered_subregions: np.ndarray,
    *,
    output_shape: Sequence[int] = (96, 96, 96),
) -> np.ndarray:
    """Map normalized registered labels back to the original NIfTI grid."""
    nii_mask = (
        nib.load(str(resolve_project_path(c2_mask)))
        if isinstance(c2_mask, (str, Path))
        else c2_mask
    )
    if not isinstance(nii_mask, nib.spatialimages.SpatialImage):
        raise TypeError("c2_mask must be a NIfTI path or nibabel image")
    shape = tuple(int(value) for value in output_shape)
    if len(shape) != 3:
        raise ValueError("output_shape must have three integers")
    return restore_labels_to_original_space(
        nii_mask,
        np.asarray(registered_subregions),
        output_shape=shape,
    )


def localize_fracture_regions(
    registered_subregions: np.ndarray,
    cam: np.ndarray,
    *,
    prediction_config: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Overlap PGIRP proposals with registered anatomy and rank C2 subregions."""
    if prediction_config is None or isinstance(prediction_config, (str, Path)):
        raw_config = load_prediction_config(prediction_config)
    else:
        raw_config = dict(prediction_config)
    params = build_prediction_params(raw_config)

    processed_cam = enhance_cam(np.asarray(cam))
    labels = predict_default_pipeline(
        np.asarray(registered_subregions), processed_cam, params
    )
    top1_labels = labels[:1]
    top3_labels = labels[:3]
    return {
        "labels": labels,
        "regions": [REGION_LABELS.get(label, f"Unknown label {label}") for label in labels],
        "top1_labels": top1_labels,
        "top1_regions": [
            REGION_LABELS.get(label, f"Unknown label {label}") for label in top1_labels
        ],
        "top1_groups": anatomical_groups_for_labels(top1_labels),
        "top3_labels": top3_labels,
        "top3_regions": [
            REGION_LABELS.get(label, f"Unknown label {label}") for label in top3_labels
        ],
        "top3_groups": anatomical_groups_for_labels(top3_labels),
        "parameters": params,
        "cam": processed_cam,
    }


def localize_registered_archive(
    archive_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for an ``.npz`` containing ``reg`` and ``cam``."""
    path = resolve_project_path(archive_path)
    with np.load(path, allow_pickle=False) as archive:
        if not {"reg", "cam"}.issubset(archive.files):
            raise KeyError(f"{path} must contain arrays named 'reg' and 'cam'")
        return localize_fracture_regions(archive["reg"], archive["cam"], **kwargs)
