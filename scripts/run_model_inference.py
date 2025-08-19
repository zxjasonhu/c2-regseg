#!/usr/bin/env python
"""Batch challenge-model inference and Grad-CAM generation from release configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from c2_reg_cls import (
    convert_dicom_to_nifti,
    resolve_project_path,
    validate_nifti_dicom_alignment,
)
from c2_reg_cls.pipeline_config import load_pipeline_config, require_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline JSON")
    parser.add_argument("--weights-config", required=True, help="Local weights JSON")
    parser.add_argument(
        "--models", nargs="+", choices=("first_place", "fourth_place"),
        default=("first_place", "fourth_place"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--reduced-ensemble",
        action="store_true",
        help="Visual check only: one segmenter and one classifier per solution",
    )
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream models through the device one at a time (default: enabled)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _case_path(base: str, template: str, study_id: str) -> Path:
    rendered = Path(template.format(study_id=study_id))
    return rendered if rendered.is_absolute() else Path(base) / rendered


def _ensure_scan_nifti(inputs: dict[str, Any], study_id: str) -> tuple[Path, str]:
    """Return the configured CT NIfTI, converting raw DICOM when necessary."""
    scan_template = inputs.get("scan_relative_path", "{study_id}/ct.nii.gz")
    scan = _case_path(inputs["scan_base"], scan_template, study_id)
    dicom_base = inputs.get("dicom_base")
    dicom = None
    if dicom_base:
        dicom_template = inputs.get("dicom_relative_path", "{study_id}")
        dicom = _case_path(dicom_base, dicom_template, study_id)

    if scan.exists():
        if dicom is not None and dicom.exists():
            validate_nifti_dicom_alignment(dicom, scan)
            return scan, "existing_aligned"
        return scan, "existing_nifti"
    if dicom is None or not dicom.exists():
        raise FileNotFoundError(f"No converted CT NIfTI or raw DICOM for {study_id}")
    return convert_dicom_to_nifti(dicom, scan), "converted"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    inputs = config["inputs"]
    require_keys(inputs, ("studies_csv", "scan_base", "model_heatmap_dirs"), "inputs")
    if len(inputs["model_heatmap_dirs"]) < 2:
        raise ValueError("model_heatmap_dirs must list first-place then fourth-place output directories")

    weights_config_path = resolve_project_path(args.weights_config)
    weights_config = json.loads(weights_config_path.read_text(encoding="utf-8"))
    generators = {}
    if "first_place" in args.models:
        from c2_reg_cls.gradcam.first_place import FirstPlaceGradCAMGenerator

        generators["first_place"] = FirstPlaceGradCAMGenerator(
            resolve_project_path(weights_config["first_place"]["weights_root"]),
            args.device,
            reduced_ensemble=args.reduced_ensemble,
            low_memory=args.low_memory,
        )
    if "fourth_place" in args.models:
        from c2_reg_cls.gradcam.fourth_place import FourthPlaceGradCAMGenerator

        generators["fourth_place"] = FourthPlaceGradCAMGenerator(
            resolve_project_path(weights_config["fourth_place"]["weights_root"]),
            args.device,
            reduced_ensemble=args.reduced_ensemble,
            low_memory=args.low_memory,
        )

    output_dirs = {
        "first_place": Path(inputs["model_heatmap_dirs"][0]),
        "fourth_place": Path(inputs["model_heatmap_dirs"][1]),
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    id_column = inputs.get("study_id_column", "StudyInstanceUID")
    studies = pd.read_csv(inputs["studies_csv"], dtype={id_column: str})
    if id_column not in studies:
        raise ValueError(f"Studies CSV is missing {id_column!r}")
    counts = {name: {"complete": 0, "existing": 0, "failed": 0} for name in generators}
    conversion_counts = {"converted": 0, "existing_aligned": 0, "existing_nifti": 0, "failed": 0}

    for study_id in studies[id_column].dropna().astype(str):
        try:
            scan, conversion_status = _ensure_scan_nifti(inputs, study_id)
            conversion_counts[conversion_status] += 1
        except Exception as error:
            conversion_counts["failed"] += 1
            print(f"CT conversion failed for configured study {study_id}: {error}")
            for count in counts.values():
                count["failed"] += 1
            continue
        for name, generator in generators.items():
            output = output_dirs[name] / f"{study_id}.npz"
            if output.exists() and not args.overwrite:
                counts[name]["existing"] += 1
                continue
            try:
                result = generator.run(scan)
                cam = np.asarray(result.pop("cam"), dtype=np.float32)
                if cam.ndim != 3:
                    raise RuntimeError(f"Expected 3-D CAM, got {cam.shape}")
                np.savez_compressed(output, cam=cam)
                output.with_suffix(".json").write_text(
                    json.dumps(_json_safe(result), indent=2), encoding="utf-8"
                )
                counts[name]["complete"] += 1
            except Exception as error:
                counts[name]["failed"] += 1
                print(f"{name} failed for configured study {study_id}: {error}")
    print(json.dumps({"conversion": conversion_counts, "models": counts}, indent=2))
    if conversion_counts["failed"] or any(count["failed"] for count in counts.values()):
        raise SystemExit("One or more inference jobs failed")


if __name__ == "__main__":
    main()
