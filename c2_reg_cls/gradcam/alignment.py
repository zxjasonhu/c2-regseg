"""Map generated CAM volumes onto the source scan's ``(Z,Y,X)`` grid."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import zoom


def _find_nifti(path: Path) -> Path | None:
    if path.is_file() and path.name.endswith((".nii", ".nii.gz")):
        return path
    if path.is_dir():
        return next(
            (item for item in sorted(path.iterdir()) if item.name.endswith((".nii", ".nii.gz"))),
            None,
        )
    return None


def scan_shape_zyx(scan_path: str | Path) -> tuple[int, int, int]:
    """Return the original scan grid in model/CAM axis order."""
    path = Path(scan_path).expanduser().resolve()
    nifti = _find_nifti(path)
    if nifti is not None:
        x, y, z = nib.load(str(nifti)).shape[:3]
        return int(z), int(y), int(x)

    dicom_paths = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".dcm"
    )
    if not dicom_paths:
        raise FileNotFoundError(f"No NIfTI or DICOM slices found in {path}")
    first = pydicom.dcmread(str(dicom_paths[0]), stop_before_pixels=True)
    return len(dicom_paths), int(first.Rows), int(first.Columns)


def align_cam_to_scan(cam: np.ndarray, scan_path: str | Path) -> np.ndarray:
    """Linearly resize a generator CAM to the original ``(Z,Y,X)`` scan grid."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 3:
        raise ValueError(f"CAM must be 3-D; got {cam.shape}")
    target_shape = scan_shape_zyx(scan_path)
    if tuple(cam.shape) == target_shape:
        return cam
    factors = tuple(target / source for source, target in zip(cam.shape, target_shape))
    result = zoom(cam, factors, order=1)
    if tuple(result.shape) != target_shape:
        exact = np.zeros(target_shape, dtype=np.float32)
        shared = tuple(slice(0, min(a, b)) for a, b in zip(result.shape, target_shape))
        exact[shared] = result[shared]
        result = exact
    return np.asarray(result, dtype=np.float32)
