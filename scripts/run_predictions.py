#!/usr/bin/env python
"""Generate reviewer-facing ranked C2 subregion predictions from registered cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from c2_reg_cls import anatomical_groups_for_labels
from c2_reg_cls.config import REGION_LABELS
from c2_reg_cls.localization import (
    build_prediction_params,
    enhance_cam,
    predict_default_pipeline,
)
from c2_reg_cls.pipeline_config import load_pipeline_config, output_paths, require_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reviewer pipeline JSON config")
    parser.add_argument("--output-csv", default=None, help="Override configured predictions path")
    return parser.parse_args()


def generate_predictions(config: dict, output_csv: str | Path | None = None) -> dict:
    """Run PGIRP for every configured study and write one row per study."""
    inputs = config["inputs"]
    require_keys(inputs, ("studies_csv",), "inputs")
    paths = output_paths(config)
    registered_dir = paths["registered"]
    output_path = Path(output_csv) if output_csv else paths["predictions"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    id_column = inputs.get("study_id_column", "StudyInstanceUID")
    studies = pd.read_csv(inputs["studies_csv"], dtype={id_column: str})
    if id_column not in studies:
        raise ValueError(f"Studies CSV is missing {id_column!r}")

    prediction_config = config.get("prediction", {})
    prediction_params = build_prediction_params(prediction_config)
    rows = []

    for _, source in tqdm(
        studies.iterrows(), total=len(studies), desc="Predicting"
    ):
        study_id = str(source[id_column])
        sample_path = registered_dir / f"{study_id}.npz"
        row = {
            id_column: study_id,
            "status": "ok",
            "predicted_labels": "[]",
            "predicted_regions": "[]",
            "predicted_top1_labels": "[]",
            "predicted_top1_regions": "[]",
            "predicted_top1_groups": "[]",
            "predicted_top3_labels": "[]",
            "predicted_top3_regions": "[]",
            "predicted_top3_groups": "[]",
        }
        if not sample_path.exists():
            row["status"] = "missing_registered_sample"
            rows.append(row)
            continue

        with np.load(sample_path, allow_pickle=False) as sample:
            if "cam" not in sample or not np.any(sample["cam"]):
                predicted = []
                row["status"] = "missing_or_empty_cam"
            else:
                cam = enhance_cam(np.asarray(sample["cam"]))
                predicted = predict_default_pipeline(
                    np.asarray(sample["reg"]), cam, prediction_params
                )

        row["predicted_labels"] = json.dumps(predicted)
        row["predicted_regions"] = json.dumps(
            [REGION_LABELS.get(label, f"Unknown label {label}") for label in predicted]
        )
        row["predicted_top1_labels"] = json.dumps(predicted[:1])
        row["predicted_top1_regions"] = json.dumps(
            [REGION_LABELS.get(label, f"Unknown label {label}") for label in predicted[:1]]
        )
        row["predicted_top1_groups"] = json.dumps(
            anatomical_groups_for_labels(predicted[:1])
        )
        row["predicted_top3_labels"] = json.dumps(predicted[:3])
        row["predicted_top3_regions"] = json.dumps(
            [REGION_LABELS.get(label, f"Unknown label {label}") for label in predicted[:3]]
        )
        row["predicted_top3_groups"] = json.dumps(
            anatomical_groups_for_labels(predicted[:3])
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    summary = {
        "num_studies": len(result),
        "num_successful": int((result["status"] == "ok").sum()),
        "predictions_csv": output_path.name,
    }
    return summary


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    summary = generate_predictions(config, args.output_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
