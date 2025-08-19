"""Focused tests for the released CAM aggregation path."""

import numpy as np

from c2_reg_cls.heatmap import aggregate_heatmaps, load_heatmap


def _save(path, cam):
    np.savez_compressed(path, cam=cam)


def test_load_heatmap_normalizes(tmp_path):
    cam = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "cam.npz"
    _save(path, cam)
    result = load_heatmap(path)
    assert result.shape == cam.shape
    assert result.min() == 0
    assert result.max() == 1
    assert result[-1, -1, -1] == 1


def test_aggregate_resizes_sums_and_masks(tmp_path):
    first = np.zeros((4, 8, 8), dtype=np.float32)
    second = np.zeros((8, 16, 16), dtype=np.float32)
    first[2, 4, 4] = 1
    second[4, 8, 8] = 1
    first_path, second_path = tmp_path / "first.npz", tmp_path / "second.npz"
    _save(first_path, first)
    _save(second_path, second)
    mask = np.zeros((8, 16, 16), dtype=np.uint8)
    mask[:, 4:12, 4:12] = 1
    result = aggregate_heatmaps([str(first_path), str(second_path)], mask.shape, mask=mask)
    assert result.shape == mask.shape
    assert result.max() == 1
    assert result[:, :4].sum() == 0


def test_missing_cams_return_zero_volume(tmp_path):
    result = aggregate_heatmaps([str(tmp_path / "missing.npz")], (4, 5, 6))
    assert result.shape == (4, 5, 6)
    assert not np.any(result)


def test_c2_mask_shape_must_match_cam_grid(tmp_path):
    path = tmp_path / "cam.npz"
    _save(path, np.ones((4, 5, 6), dtype=np.float32))
    with np.testing.assert_raises_regex(ValueError, "C2 mask shape"):
        aggregate_heatmaps([str(path)], (4, 5, 6), mask=np.ones((4, 5, 5)))
