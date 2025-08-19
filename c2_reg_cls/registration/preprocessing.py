"""Mask resampling and fixed-size embedding used before registration."""

import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components
from scipy.ndimage import zoom


def _remove_small_components(volume: np.ndarray, min_size: int = 1000) -> np.ndarray:
    labels, _ = connected_components(volume)
    counts = np.bincount(labels.ravel())
    small = np.where(counts < min_size)[0]
    return np.where(np.isin(labels, small), 0, volume)


def isotropic_resample(
    image: nib.Nifti1Image,
    side_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Binarize, crop, and resample a C2 mask and optional aligned volume to 1 mm."""
    # Keep float64 here to match nibabel's historical ``get_fdata`` path used
    # for the paper results.
    main = (image.get_fdata() > 0).astype(np.float64)
    original_shape = main.shape
    main = _remove_small_components(main)
    coordinates = np.argwhere(main > 0)
    if not len(coordinates):
        raise ValueError("C2 mask is empty after removing small components")
    lower = coordinates.min(axis=0)
    # Preserve the paper implementation's upper-exclusive crop convention.
    upper = coordinates.max(axis=0)
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    factors = np.asarray(image.header.get_zooms()[:3], dtype=float)
    main = zoom(main[crop], factors, order=0)

    side = None
    if side_image is not None:
        side_image = np.asarray(side_image)
        if side_image.shape != original_shape:
            raise ValueError(
                f"Aligned volume shape {side_image.shape} does not match mask {original_shape}"
            )
        side = zoom(side_image[crop], factors, order=0)
    return main, side


def embed_to_target_size(
    volume: np.ndarray,
    output_shape: tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    """Isotropically resize and center one cropped volume in ``output_shape``."""
    scale = output_shape[0] / max(volume.shape)
    resized = zoom(volume, scale, order=0)
    # The paper implementation embedded into NumPy's default float64 array.
    output = np.zeros(output_shape)
    offset = (np.asarray(output_shape) - np.asarray(resized.shape)) // 2
    slices = tuple(slice(int(start), int(start + size)) for start, size in zip(offset, resized.shape))
    output[slices] = resized
    return output


def _resize_nearest_exact(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a label volume with nearest-neighbor interpolation to an exact shape."""
    if tuple(volume.shape) == target_shape:
        return np.asarray(volume)
    factors = np.asarray(target_shape, dtype=float) / np.asarray(volume.shape, dtype=float)
    resized = zoom(volume, factors, order=0)
    if tuple(resized.shape) == target_shape:
        return resized
    exact = np.zeros(target_shape, dtype=resized.dtype)
    shared = tuple(slice(0, min(a, b)) for a, b in zip(resized.shape, target_shape))
    exact[shared] = resized[shared]
    return exact


def restore_labels_to_original_space(
    image: nib.Nifti1Image,
    labels_zyx: np.ndarray,
    output_shape: tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    """Invert the paper normalization and return labels on the NIfTI ``(X,Y,Z)`` grid.

    This reconstructs the same crop, 1-mm resampling, fixed-size embedding, and
    axis conversion used before registration. Nearest-neighbor interpolation
    preserves integer anatomical labels, and the result is clipped to the
    original binary C2 mask.
    """
    labels_zyx = np.asarray(labels_zyx)
    if tuple(labels_zyx.shape) != tuple(output_shape):
        raise ValueError(
            f"Normalized labels must have shape {tuple(output_shape)}; "
            f"got {tuple(labels_zyx.shape)}"
        )

    original_mask = (image.get_fdata() > 0).astype(np.float64)
    cleaned_mask = _remove_small_components(original_mask)
    coordinates = np.argwhere(cleaned_mask > 0)
    if not len(coordinates):
        raise ValueError("C2 mask is empty after removing small components")

    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    cropped_mask = cleaned_mask[crop]
    spacings = np.asarray(image.header.get_zooms()[:3], dtype=float)
    isotropic_mask = zoom(cropped_mask, spacings, order=0)

    scale = output_shape[0] / max(isotropic_mask.shape)
    resized_mask = zoom(isotropic_mask, scale, order=0)
    offset = (np.asarray(output_shape) - np.asarray(resized_mask.shape)) // 2
    embedded_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(offset, resized_mask.shape)
    )

    # Registration space is produced from embedded NIfTI space by
    # transpose(2, 1, 0) followed by a reversal of its Y axis.
    labels_xyz = labels_zyx[:, ::-1, :].transpose(2, 1, 0)
    resized_labels = labels_xyz[embedded_slices]
    isotropic_labels = _resize_nearest_exact(resized_labels, isotropic_mask.shape)
    cropped_labels = _resize_nearest_exact(isotropic_labels, cropped_mask.shape)

    restored = np.zeros(original_mask.shape, dtype=np.int32)
    restored[crop] = np.rint(cropped_labels).astype(np.int32)
    restored[original_mask <= 0] = 0
    return restored
