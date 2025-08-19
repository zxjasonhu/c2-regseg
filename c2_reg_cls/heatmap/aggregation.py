"""Load, resize, and sum the two model Grad-CAM volumes."""

from pathlib import Path

import numpy as np
from scipy import ndimage


def load_heatmap(path: str | Path) -> np.ndarray:
    """Load one ``cam`` array and normalize it to ``[0,1]``."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            cam = np.asarray(archive["cam"], dtype=np.float32)
    else:
        cam = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if cam.ndim != 3:
        raise ValueError(f"CAM must be 3-D; got {cam.shape} from {path}")
    cam -= cam.min()
    maximum = cam.max()
    if maximum > 0:
        cam /= maximum
    return cam


def _resize_cam(cam: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(cam.shape) == target_shape:
        return cam
    factors = tuple(target / source for source, target in zip(cam.shape, target_shape))
    resized = ndimage.zoom(cam, factors, order=1)
    if tuple(resized.shape) != target_shape:
        exact = np.zeros(target_shape, dtype=np.float32)
        common = tuple(slice(0, min(a, b)) for a, b in zip(resized.shape, target_shape))
        exact[common] = resized[common]
        resized = exact
    return resized


def aggregate_heatmaps(
    heatmap_paths: list[str],
    target_shape: tuple[int, int, int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize, resize, sum, mask to C2, and renormalize model CAMs."""
    if len(target_shape) != 3:
        raise ValueError("target_shape must contain three dimensions")
    result = np.zeros(target_shape, dtype=np.float32)
    loaded = 0
    for raw_path in heatmap_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        result += _resize_cam(load_heatmap(path), target_shape)
        loaded += 1
    if not loaded:
        return result

    if mask is not None:
        c2_mask = np.asarray(mask)
        if tuple(c2_mask.shape) != target_shape:
            raise ValueError(
                f"C2 mask shape {c2_mask.shape} does not match CAM shape {target_shape}"
            )
        result *= c2_mask > 0

    maximum = result.max()
    if maximum > 0:
        result /= maximum
    return result
