"""Generate and validate the whole-C2 mask used before atlas registration."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np


C2_MASK_FILENAME = "vertebrae_C2.nii.gz"
TOTALSEGMENTATOR_VERSION = "2.8.0"
TOTALSEGMENTATOR_CSPINE_TASK_ID = 292


def _require_paper_totalsegmentator() -> None:
    """Require the version declared by the paper's zxnnframe C-spine path."""
    try:
        installed = version("TotalSegmentator")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "TotalSegmentator is required to generate a C2 mask. Install "
            "requirements-segmentation.txt."
        ) from error
    if installed != TOTALSEGMENTATOR_VERSION:
        raise RuntimeError(
            f"Paper-compatible C2 generation requires TotalSegmentator "
            f"{TOTALSEGMENTATOR_VERSION}; found {installed}. Install "
            "requirements-segmentation.txt in a dedicated environment."
        )


def _nifti_input(path: Path) -> Path | None:
    """Return the unambiguous source NIfTI, or ``None`` for a DICOM folder."""
    if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return path
    if not path.is_dir():
        return None
    candidates = sorted(path.glob("*.nii")) + sorted(path.glob("*.nii.gz"))
    if len(candidates) > 1:
        raise ValueError(f"Multiple source NIfTI files found in {path}")
    return candidates[0] if candidates else None


def _dicom_shape_zyx(path: Path) -> tuple[int, int, int] | None:
    """Read only DICOM headers and return ``(slices, rows, columns)``."""
    if not path.is_dir():
        return None
    try:
        import pydicom
    except ImportError:
        return None

    dimensions: list[tuple[int, int]] = []
    for candidate in sorted(path.iterdir()):
        if not candidate.is_file():
            continue
        try:
            header = pydicom.dcmread(str(candidate), stop_before_pixels=True)
        except Exception:
            continue
        if hasattr(header, "Rows") and hasattr(header, "Columns"):
            dimensions.append((int(header.Rows), int(header.Columns)))
    if not dimensions:
        return None
    if len(set(dimensions)) != 1:
        raise ValueError(f"Inconsistent DICOM slice dimensions in {path}")
    rows, columns = dimensions[0]
    return len(dimensions), rows, columns


def _sorted_dicom_headers(path: Path) -> list[Any]:
    """Load one conventional CT series and sort it as dicom2nifti does."""
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required for DICOM geometry validation") from error

    headers = []
    for candidate in sorted(path.iterdir()):
        if not candidate.is_file():
            continue
        try:
            header = pydicom.dcmread(str(candidate), stop_before_pixels=True)
        except Exception:
            continue
        required = (
            "Rows",
            "Columns",
            "PixelSpacing",
            "ImageOrientationPatient",
            "ImagePositionPatient",
        )
        if all(hasattr(header, field) for field in required):
            headers.append(header)
    if not headers:
        raise FileNotFoundError(f"No spatially complete DICOM slices found in {path}")

    series_ids = {
        str(header.SeriesInstanceUID)
        for header in headers
        if hasattr(header, "SeriesInstanceUID")
    }
    if len(series_ids) > 1:
        raise ValueError(f"Expected one DICOM series in {path}; found {len(series_ids)}")
    orientations = {
        tuple(float(value) for value in header.ImageOrientationPatient)
        for header in headers
    }
    if len(orientations) != 1:
        raise ValueError("DICOM ImageOrientationPatient is inconsistent across slices")

    positions = np.asarray(
        [header.ImagePositionPatient for header in headers], dtype=float
    )
    axis = int(np.argmax(np.ptp(positions, axis=0)))
    return sorted(headers, key=lambda header: float(header.ImagePositionPatient[axis]))


def _historical_dicom_affine(headers: list[Any]) -> np.ndarray:
    """Reproduce dicom2nifti's ``reorient=False`` RAS affine construction."""
    first = headers[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    orient_1, orient_2 = orientation[:3], orientation[3:]
    row_spacing, column_spacing = (float(value) for value in first.PixelSpacing)
    first_position = np.asarray(first.ImagePositionPatient, dtype=float)
    if len(headers) == 1:
        thickness = float(getattr(first, "SliceThickness", 1.0))
        step = -np.cross(orient_1, orient_2) * thickness
    else:
        last_position = np.asarray(headers[-1].ImagePositionPatient, dtype=float)
        step = (first_position - last_position) / (1 - len(headers))
    if np.linalg.norm(step) == 0:
        raise ValueError("DICOM slices do not span a 3-D volume")
    return np.asarray(
        [
            [-orient_1[0] * column_spacing, -orient_2[0] * row_spacing, -step[0], -first_position[0]],
            [-orient_1[1] * column_spacing, -orient_2[1] * row_spacing, -step[1], -first_position[1]],
            [orient_1[2] * column_spacing, orient_2[2] * row_spacing, step[2], first_position[2]],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def validate_nifti_dicom_alignment(
    dicom_directory: str | Path,
    nifti_path: str | Path,
    *,
    affine_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Check a NIfTI against the historical ``dicom2nifti`` direction convention."""
    dicom_path = Path(dicom_directory).expanduser().resolve()
    output_path = Path(nifti_path).expanduser().resolve()
    if not dicom_path.is_dir():
        raise FileNotFoundError(f"DICOM directory does not exist: {dicom_path}")
    if not output_path.is_file():
        raise FileNotFoundError(f"Converted NIfTI does not exist: {output_path}")

    headers = _sorted_dicom_headers(dicom_path)
    rows = int(headers[0].Rows)
    columns = int(headers[0].Columns)
    expected_shape = (columns, rows, len(headers))
    expected_affine = _historical_dicom_affine(headers)
    image = nib.load(str(output_path))
    if tuple(image.shape[:3]) != expected_shape:
        raise ValueError(
            f"Converted NIfTI shape {image.shape[:3]} does not match DICOM {expected_shape}"
        )
    if not np.allclose(image.affine, expected_affine, rtol=0, atol=affine_tolerance):
        raise ValueError("Converted NIfTI affine does not match DICOM direction metadata")
    expected_orientation = nib.aff2axcodes(expected_affine)
    output_orientation = nib.aff2axcodes(image.affine)
    if output_orientation != expected_orientation:
        raise ValueError(
            f"Converted orientation {output_orientation} does not match DICOM-derived "
            f"orientation {expected_orientation}"
        )
    return {
        "nifti_path": output_path,
        "nifti_shape_xyz": expected_shape,
        "scan_shape_zyx": tuple(reversed(expected_shape)),
        "orientation": output_orientation,
        "geometry_check": "dicom2nifti_reorient_false_affine",
    }


def convert_dicom_to_nifti(
    dicom_directory: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    converter: Callable[..., Any] | None = None,
) -> Path:
    """Convert one DICOM series using the paper's ``reorient=False`` convention."""
    dicom_path = Path(dicom_directory).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not destination.name.endswith((".nii", ".nii.gz")):
        raise ValueError("Converted CT output must be a .nii or .nii.gz file")
    if destination.exists() and not overwrite:
        validate_nifti_dicom_alignment(dicom_path, destination)
        return destination
    _sorted_dicom_headers(dicom_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if converter is None:
        try:
            from dicom2nifti import dicom_series_to_nifti as converter
        except ImportError as error:
            raise RuntimeError(
                "dicom2nifti is required for the historical DICOM conversion step"
            ) from error
    converter(
        original_dicom_directory=str(dicom_path),
        output_file=str(destination),
        reorient_nifti=False,
    )
    validate_nifti_dicom_alignment(dicom_path, destination)
    return destination


def resolve_segmentation_input(scan: str | Path) -> Path:
    """Resolve one NIfTI file or retain a DICOM directory for TotalSegmentator."""
    path = Path(scan).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CT input does not exist: {path}")
    nifti = _nifti_input(path)
    if nifti is not None:
        return nifti
    if path.is_dir() and _dicom_shape_zyx(path) is not None:
        return path
    raise FileNotFoundError(f"No unambiguous NIfTI or readable DICOM series in {path}")


def validate_c2_mask_alignment(
    scan: str | Path,
    mask: str | Path,
    *,
    affine_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Validate that a binary C2 mask occupies the source CT grid.

    NIfTI inputs receive strict shape, affine, and orientation checks. For a
    DICOM directory, the saved NIfTI dimensions are checked against the series;
    TotalSegmentator is responsible for the DICOM-to-NIfTI spatial transform.
    """
    source = resolve_segmentation_input(scan)
    mask_path = Path(mask).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"C2 mask does not exist: {mask_path}")

    mask_image = nib.load(str(mask_path))
    mask_data = np.asarray(mask_image.dataobj)
    if mask_data.ndim != 3:
        raise ValueError(f"C2 mask must be 3-D; got shape {mask_data.shape}")
    foreground = mask_data > 0
    voxel_count = int(np.count_nonzero(foreground))
    if voxel_count == 0:
        raise ValueError("C2 mask is empty")
    finite_values = np.unique(mask_data[np.isfinite(mask_data)])
    if not set(finite_values.tolist()).issubset({0, 1}):
        raise ValueError(f"C2 mask must be binary; found values {finite_values.tolist()}")

    source_nifti = _nifti_input(source)
    if source_nifti is not None:
        scan_image = nib.load(str(source_nifti))
        if tuple(mask_image.shape) != tuple(scan_image.shape):
            raise ValueError(
                f"C2 mask shape {mask_image.shape} does not match CT {scan_image.shape}"
            )
        if not np.allclose(
            mask_image.affine,
            scan_image.affine,
            rtol=0,
            atol=float(affine_tolerance),
        ):
            raise ValueError("C2 mask affine does not match the source CT affine")
        scan_orientation = nib.aff2axcodes(scan_image.affine)
        mask_orientation = nib.aff2axcodes(mask_image.affine)
        if mask_orientation != scan_orientation:
            raise ValueError(
                f"C2 mask orientation {mask_orientation} does not match CT {scan_orientation}"
            )
        geometry_check = "nifti_shape_affine_orientation"
        scan_shape_zyx = tuple(reversed(scan_image.shape[:3]))
    else:
        scan_shape_zyx = _dicom_shape_zyx(source)
        if scan_shape_zyx is None:
            raise ValueError(f"Could not determine DICOM dimensions in {source}")
        mask_shape_zyx = tuple(reversed(mask_image.shape[:3]))
        if mask_shape_zyx != scan_shape_zyx:
            raise ValueError(
                f"C2 mask (Z,Y,X) shape {mask_shape_zyx} does not match DICOM {scan_shape_zyx}"
            )
        geometry_check = "dicom_dimensions"

    return {
        "mask_path": mask_path,
        "mask_shape_xyz": tuple(int(value) for value in mask_image.shape[:3]),
        "scan_shape_zyx": tuple(int(value) for value in scan_shape_zyx),
        "foreground_voxels": voxel_count,
        "geometry_check": geometry_check,
    }


def generate_c2_mask(
    scan: str | Path,
    output_path: str | Path,
    *,
    device: str = "gpu",
    force_split: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    segmenter: Callable[..., Any] | None = None,
) -> Path:
    """Generate ``vertebrae_C2.nii.gz`` with TotalSegmentator and validate it.

    The paper pipeline's zxnnframe implementation runs the v2 cervical model
    (Task 292) directly, without TotalSegmentator's rough ROI-cropping pass.
    Restricting saved output to ``vertebrae_C2`` avoids writing the other
    cervical masks. Model weights remain managed by TotalSegmentator and are
    not included in this repository.
    """
    source = resolve_segmentation_input(scan)
    if _nifti_input(source) is None:
        raise ValueError(
            "Paper-compatible C2 generation requires a source NIfTI. Convert the "
            "DICOM series with convert_dicom_to_nifti() first."
        )
    destination = Path(output_path).expanduser().resolve()
    if destination.name != C2_MASK_FILENAME:
        raise ValueError(f"C2 output filename must be {C2_MASK_FILENAME!r}")
    if destination.exists() and not overwrite:
        validate_c2_mask_alignment(source, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    if segmenter is None:
        _require_paper_totalsegmentator()
        try:
            import torch
            from totalsegmentator.python_api import setup_nnunet, setup_totalseg

            setup_nnunet()
            setup_totalseg()
            from totalsegmentator.libs import download_pretrained_weights
            from totalsegmentator.nnunet import nnUNet_predict_image as segmenter
        except ImportError as error:
            raise RuntimeError(
                "TotalSegmentator is required to generate a C2 mask. "
                "Install the release's segmentation optional dependency."
            ) from error

        download_pretrained_weights(TOTALSEGMENTATOR_CSPINE_TASK_ID)
        requested_device = str(device).lower()
        if requested_device == "gpu":
            inference_device: Any = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device.startswith("gpu:"):
            gpu_index = requested_device.split(":", 1)[1]
            inference_device = (
                torch.device(f"cuda:{gpu_index}")
                if torch.cuda.is_available()
                else "cpu"
            )
        elif requested_device in {"cpu", "mps"}:
            inference_device = requested_device
        else:
            raise ValueError("device must be 'gpu', 'gpu:N', 'cpu', or 'mps'")
    else:
        inference_device = str(device)

    segmenter(
        file_in=source,
        file_out=destination.parent,
        task_id=[TOTALSEGMENTATOR_CSPINE_TASK_ID],
        model="3d_fullres",
        folds=[0],
        trainer="nnUNetTrainerNoMirroring",
        tta=False,
        multilabel_image=False,
        resample=1.5,
        crop=None,
        crop_path=destination.parent,
        task_name="total",
        roi_subset=["vertebrae_C2"],
        force_split=bool(force_split),
        device=inference_device,
        quiet=not bool(verbose),
        verbose=bool(verbose),
    )
    if not destination.is_file():
        raise FileNotFoundError(
            f"TotalSegmentator completed without creating {destination}"
        )
    validate_c2_mask_alignment(source, destination)
    return destination
