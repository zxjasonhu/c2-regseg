"""Focused tests for the released PGIRP localization path."""

import numpy as np

from c2_reg_cls.localization import (
    build_prediction_params,
    enhance_cam,
    predict_default_pipeline,
)


def test_quadratic_enhancement_is_normalized(rng):
    result = enhance_cam(rng.random((8, 8, 8)))
    assert result.min() >= 0
    assert result.max() == 1


def test_pgirp_ranks_dominant_anatomy_first(synthetic_cam, synthetic_segment_mask):
    params = build_prediction_params({"min_region_size": 50, "core_frac": 0.01})
    predictions = predict_default_pipeline(
        synthetic_segment_mask, enhance_cam(synthetic_cam), params
    )
    assert predictions
    assert predictions[0] == 5


def test_pgirp_handles_empty_cam(synthetic_segment_mask):
    params = build_prediction_params({})
    assert predict_default_pipeline(
        synthetic_segment_mask, np.zeros_like(synthetic_segment_mask), params
    ) == []


def test_pgirp_preserves_noncollapsed_atlas_label():
    segment_mask = np.full((8, 8, 8), 8, dtype=np.int32)
    cam = np.ones(segment_mask.shape, dtype=np.float32)
    params = build_prediction_params({"min_region_size": 1})
    assert predict_default_pipeline(segment_mask, cam, params) == [8]
