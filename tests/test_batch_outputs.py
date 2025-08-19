import json

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.run_pipeline import _load_model_volume
from scripts.run_predictions import generate_predictions


def test_model_volume_loader_uses_historical_zyx_order(tmp_path):
    source = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(source, np.eye(4)), path)

    np.testing.assert_array_equal(
        _load_model_volume(path), source.transpose(2, 1, 0)[::-1]
    )


def test_batch_csv_includes_paper_top3_and_five_group_outputs(tmp_path):
    studies = tmp_path / "studies.csv"
    pd.DataFrame({"StudyInstanceUID": ["case"]}).to_csv(studies, index=False)

    registered = tmp_path / "registered"
    registered.mkdir()
    cam = np.zeros((8, 8, 8), dtype=np.float32)
    cam[2:6, 2:6, 2:6] = 1
    np.savez_compressed(
        registered / "case.npz",
        reg=np.full((8, 8, 8), 8, dtype=np.int32),
        cam=cam,
    )

    config = {
        "inputs": {"studies_csv": str(studies)},
        "outputs": {
            "output_root": str(tmp_path),
            "registered": "registered",
            "predictions": "predictions.csv",
        },
        "prediction": {"min_region_size": 1},
    }
    generate_predictions(config)

    row = pd.read_csv(tmp_path / "predictions.csv").iloc[0]
    assert json.loads(row["predicted_top3_labels"]) == [8]
    assert json.loads(row["predicted_top3_regions"]) == [
        "Superior Articular Facet - Right"
    ]
    assert json.loads(row["predicted_top3_groups"]) == [
        "Subdental Axis Body and Superior Lateral Mass Complex"
    ]
