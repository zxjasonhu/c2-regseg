"""Normalize patient masks and load the included expert atlases."""

from pathlib import Path

import nibabel as nib
import numpy as np

from ..config import DEFAULT_OUTPUT_SHAPE
from .preprocessing import embed_to_target_size, isotropic_resample


def normalize_mask(
    nii_image: nib.Nifti1Image,
    output_shape: tuple[int, int, int] = DEFAULT_OUTPUT_SHAPE,
    side_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resample and center the patient C2 mask and an optional aligned volume."""
    main, side = isotropic_resample(nii_image, side_image=side_image)
    embedded_main = embed_to_target_size(main, output_shape)
    embedded_side = embed_to_target_size(side, output_shape) if side is not None else None
    return embedded_main, embedded_side


def load_reference_masks(paths: list[str]) -> list[np.ndarray]:
    """Load the numeric two-channel atlas arrays."""
    references = []
    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as archive:
            references.append(np.asarray(archive["mask"]))
    return references
