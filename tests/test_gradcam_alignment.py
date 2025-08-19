from pathlib import Path

import nibabel as nib
import numpy as np

from c2_reg_cls.gradcam.alignment import align_cam_to_scan, scan_shape_zyx
from c2_reg_cls.gradcam.interpolation import interpolate_cam_on_voxel
from c2_reg_cls.gradcam.first_place.loader import expected_checkpoint_paths as first_paths
from c2_reg_cls.gradcam.fourth_place import expected_checkpoint_paths as fourth_paths


def test_cam_is_aligned_to_original_nifti_grid(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 10, 12)), np.eye(4)), scan)
    cam = np.zeros((6, 5, 4), dtype=np.float32)
    cam[3, 2, 2] = 1

    aligned = align_cam_to_scan(cam, scan)

    assert scan_shape_zyx(scan) == (12, 10, 8)
    assert aligned.shape == (12, 10, 8)
    assert np.isfinite(aligned).all()
    assert aligned.max() > 0


def test_reduced_and_production_checkpoint_manifests_are_distinct(tmp_path):
    first_root = Path(tmp_path / "first")
    fourth_root = Path(tmp_path / "fourth")

    assert len(first_paths(first_root)) == 31
    assert len(first_paths(first_root, reduced_ensemble=True)) == 2
    assert len(fourth_paths(fourth_root)) == 5
    assert len(fourth_paths(fourth_root, reduced_ensemble=True)) == 2


def test_empty_fourth_place_crop_is_skipped():
    result = interpolate_cam_on_voxel(
        np.zeros((4, 4, 4), dtype=np.float32),
        np.zeros((1, 2, 2, 2), dtype=np.float32),
        [{
            "type": "empty",
            "total_slice": 0,
            "current_index": 0,
            "z1": -1,
            "z2": -1,
            "x1": -1,
            "x2": -1,
            "y1": -1,
            "y2": -1,
        }],
    )
    assert not np.any(result)
