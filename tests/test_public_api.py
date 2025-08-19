from pathlib import Path

import numpy as np

from c2_reg_cls import (
    anatomical_groups_for_labels,
    default_atlas_paths,
    load_prediction_config,
    localize_fracture_regions,
    project_root,
    validate_c2_mask_alignment,
)
from c2_reg_cls.pipeline_config import load_pipeline_config


def test_paper_group_mapping_uses_ranked_noncollapsed_labels():
    assert anatomical_groups_for_labels([6, 14, 8]) == [
        "Subdental Axis Body and Superior Lateral Mass Complex",
        "Pars-Inferior Facet Complex",
    ]


def test_release_paths_are_project_root_relative():
    root = project_root()
    config = load_pipeline_config(root / "configs" / "pipeline.example.json")
    assert config["_project_root"] == str(root)
    assert config["inputs"]["dicom_base"].startswith(str(root))
    assert all(str(path).startswith(str(root)) for path in config["inputs"]["reference_masks"])
    assert len(default_atlas_paths()) == 3


def test_release_config_exposes_only_the_paper_pgirp_parameters():
    config = load_pipeline_config(project_root() / "configs" / "pipeline.example.json")
    assert set(config["prediction"]) == {
        "center_relative_threshold",
        "region_relative_threshold",
        "min_region_size",
        "max_regions",
        "level_ratio",
        "core_frac",
    }
    assert config["c2_segmentation"] == {
        "device": "gpu",
        "force_split": False,
        "verbose": False,
    }
    assert callable(validate_c2_mask_alignment)


def test_included_atlases_are_numeric_two_channel_volumes():
    config = load_pipeline_config(project_root() / "configs" / "pipeline.example.json")
    paths = config["inputs"]["reference_masks"]
    assert [Path(path).name for path in paths] == [
        "atlas_01_embedded.npz",
        "atlas_02_embedded.npz",
        "atlas_03_embedded.npz",
    ]
    for path in default_atlas_paths():
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["mask"]
            assert archive["mask"].shape == (2, 96, 96, 96)
            assert archive["mask"].dtype.kind in "biuf"


def test_public_localization_api_handles_empty_cam():
    shape = (12, 12, 12)
    result = localize_fracture_regions(
        np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.float32)
    )
    assert result["labels"] == []
    assert result["regions"] == []
    assert result["top1_labels"] == []
    assert result["top1_groups"] == []
    assert result["top3_labels"] == []
    assert result["top3_groups"] == []
    assert result["parameters"] == load_prediction_config()
