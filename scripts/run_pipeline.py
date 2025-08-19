#!/usr/bin/env python
"""Run DICOM conversion, C2 segmentation, aggregation, and localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from c2_reg_cls import (
    convert_dicom_to_nifti,
    generate_c2_mask,
    validate_c2_mask_alignment,
    validate_nifti_dicom_alignment,
)
from c2_reg_cls.localization import align_side_image_to_nii
from c2_reg_cls.heatmap import aggregate_heatmaps
from c2_reg_cls.pipeline_config import (
    PipelineConfigError,
    load_pipeline_config,
    output_paths,
    require_keys,
)
from c2_reg_cls.registration import (
    fill_mask_with_subsegments,
    load_reference_masks,
    normalize_mask,
    register_subsegment_mask_multi,
)

from scripts.run_predictions import generate_predictions


STAGES = ("convert", "segment", "aggregate", "register", "predict")


def _load_model_volume(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume in the classifiers' historical ``(Z,Y,X)`` order."""
    volume = nib.load(str(path)).get_fdata()
    return volume.transpose(2, 1, 0)[::-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reviewer pipeline JSON config")
    parser.add_argument("--from-stage", choices=STAGES, default="aggregate")
    parser.add_argument("--through-stage", choices=STAGES, default="predict")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print work only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _study_ids(config: dict[str, Any]) -> list[str]:
    inputs = config["inputs"]
    require_keys(inputs, ("studies_csv",), "inputs")
    id_column = inputs.get("study_id_column", "StudyInstanceUID")
    frame = pd.read_csv(inputs["studies_csv"], dtype={id_column: str})
    if id_column not in frame:
        raise PipelineConfigError(f"Studies CSV is missing {id_column!r}")
    ids = frame[id_column].dropna().astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise PipelineConfigError("Studies CSV contains duplicate study identifiers")
    return ids


def _case_path(base: str, template: str, study_id: str) -> Path:
    rendered = template.format(study_id=study_id)
    candidate = Path(rendered)
    return candidate if candidate.is_absolute() else Path(base) / candidate


def _convert(config: dict[str, Any], study_ids: list[str], overwrite: bool) -> dict[str, int]:
    """Convert raw DICOM series with the historical non-reoriented convention."""
    inputs = config["inputs"]
    require_keys(inputs, ("dicom_base", "scan_base"), "inputs")
    dicom_template = inputs.get("dicom_relative_path", "{study_id}")
    scan_template = inputs.get("scan_relative_path", "{study_id}/ct.nii.gz")
    counts = {
        "complete": 0,
        "existing_aligned": 0,
        "missing_dicom": 0,
        "failed": 0,
    }
    for study_id in tqdm(study_ids, desc="Converting DICOM to NIfTI"):
        dicom_path = _case_path(inputs["dicom_base"], dicom_template, study_id)
        scan_path = _case_path(inputs["scan_base"], scan_template, study_id)
        if not dicom_path.exists():
            counts["missing_dicom"] += 1
            continue
        try:
            if scan_path.exists() and not overwrite:
                validate_nifti_dicom_alignment(dicom_path, scan_path)
                counts["existing_aligned"] += 1
                continue
            convert_dicom_to_nifti(
                dicom_path, scan_path, overwrite=overwrite
            )
            counts["complete"] += 1
        except Exception as error:
            counts["failed"] += 1
            print(f"DICOM conversion failed for {study_id}: {error}", file=sys.stderr)
    return counts


def _segment(config: dict[str, Any], study_ids: list[str], overwrite: bool) -> dict[str, int]:
    """Generate or validate the TotalSegmentator C2 mask for every study."""
    inputs = config["inputs"]
    require_keys(inputs, ("scan_base", "mask_base"), "inputs")
    options = config.get("c2_segmentation", {})
    scan_template = inputs.get("scan_relative_path", "{study_id}")
    mask_template = inputs.get("mask_relative_path", "{study_id}/vertebrae_C2.nii.gz")
    counts = {
        "complete": 0,
        "existing_aligned": 0,
        "missing_scan": 0,
        "failed": 0,
    }
    for study_id in tqdm(study_ids, desc="Generating C2 masks"):
        scan_path = _case_path(inputs["scan_base"], scan_template, study_id)
        mask_path = _case_path(inputs["mask_base"], mask_template, study_id)
        if not scan_path.exists():
            counts["missing_scan"] += 1
            continue
        try:
            if mask_path.exists() and not overwrite:
                validate_c2_mask_alignment(scan_path, mask_path)
                counts["existing_aligned"] += 1
                continue
            generate_c2_mask(
                scan_path,
                mask_path,
                device=str(options.get("device", "gpu")),
                force_split=bool(options.get("force_split", False)),
                overwrite=overwrite,
                verbose=bool(options.get("verbose", False)),
            )
            counts["complete"] += 1
        except Exception as error:
            counts["failed"] += 1
            print(f"C2 segmentation failed for {study_id}: {error}", file=sys.stderr)
    return counts


def _aggregate(config: dict[str, Any], study_ids: list[str], overwrite: bool) -> dict[str, int]:
    inputs = config["inputs"]
    require_keys(inputs, ("scan_base", "mask_base", "model_heatmap_dirs"), "inputs")
    paths = output_paths(config)
    destination = paths["aggregated_heatmaps"]
    destination.mkdir(parents=True, exist_ok=True)

    counts = {
        "complete": 0,
        "missing_scan": 0,
        "missing_mask": 0,
        "missing_all_cams": 0,
        "existing": 0,
    }
    scan_template = inputs.get("scan_relative_path", "{study_id}")
    mask_template = inputs.get(
        "mask_relative_path", "{study_id}/segmentations/vertebrae_C2.nii.gz"
    )
    for study_id in tqdm(study_ids, desc="Aggregating CAMs"):
        output = destination / f"{study_id}.npz"
        if output.exists() and not overwrite:
            counts["existing"] += 1
            continue
        scan_path = _case_path(inputs["scan_base"], scan_template, study_id)
        mask_path = _case_path(inputs["mask_base"], mask_template, study_id)
        if not scan_path.exists():
            counts["missing_scan"] += 1
            continue
        if not mask_path.exists():
            counts["missing_mask"] += 1
            continue
        cam_paths = [str(Path(directory) / f"{study_id}.npz") for directory in inputs["model_heatmap_dirs"]]
        if not any(Path(path).exists() for path in cam_paths):
            counts["missing_all_cams"] += 1
            continue
        validate_c2_mask_alignment(scan_path, mask_path)
        target_shape = _load_model_volume(scan_path).shape
        c2_mask = _load_model_volume(mask_path) > 0
        cam = aggregate_heatmaps(cam_paths, target_shape, mask=c2_mask)
        np.savez_compressed(output, cam=cam)
        counts["complete"] += 1
    return counts


def _register(config: dict[str, Any], study_ids: list[str], overwrite: bool) -> dict[str, int]:
    inputs = config["inputs"]
    require_keys(
        inputs,
        ("scan_base", "mask_base", "reference_masks"),
        "inputs",
    )
    paths = output_paths(config)
    destination = paths["registered"]
    destination.mkdir(parents=True, exist_ok=True)
    options = config.get("registration", {})
    output_shape = tuple(int(v) for v in options.get("output_shape", [96, 96, 96]))
    if len(output_shape) != 3:
        raise PipelineConfigError("registration.output_shape must have three integers")
    references = load_reference_masks(inputs["reference_masks"])
    references = [mask.transpose(0, 3, 2, 1) for mask in references]
    scan_template = inputs.get("scan_relative_path", "{study_id}")
    mask_template = inputs.get(
        "mask_relative_path", "{study_id}/segmentations/vertebrae_C2.nii.gz"
    )

    counts = {
        "complete": 0,
        "existing": 0,
        "missing_scan": 0,
        "missing_mask": 0,
        "missing_cam": 0,
        "failed": 0,
    }
    for study_id in tqdm(study_ids, desc="Registering atlases"):
        output = destination / f"{study_id}.npz"
        if output.exists() and not overwrite:
            counts["existing"] += 1
            continue
        scan_path = _case_path(inputs["scan_base"], scan_template, study_id)
        mask_path = _case_path(inputs["mask_base"], mask_template, study_id)
        cam_path = paths["aggregated_heatmaps"] / f"{study_id}.npz"
        if not scan_path.exists():
            counts["missing_scan"] += 1
            continue
        if not mask_path.exists():
            counts["missing_mask"] += 1
            continue
        if not cam_path.exists():
            counts["missing_cam"] += 1
            continue
        try:
            validate_c2_mask_alignment(scan_path, mask_path)
            nii_mask = nib.load(str(mask_path))
            scan = nib.load(str(scan_path)).get_fdata()
            with np.load(cam_path, allow_pickle=False) as archive:
                cam = align_side_image_to_nii(nii_mask, np.asarray(archive["cam"]))
            mask_emb, scan_emb = normalize_mask(
                nii_mask, output_shape=output_shape, side_image=scan
            )
            _, cam_emb = normalize_mask(
                nii_mask, output_shape=output_shape, side_image=cam
            )
            mask_emb = mask_emb.transpose(2, 1, 0)[:, ::-1, :]
            scan_emb = scan_emb.transpose(2, 1, 0)[:, ::-1, :]
            cam_emb = cam_emb.transpose(2, 1, 0)[:, ::-1, :]
            registered = register_subsegment_mask_multi(
                references,
                mask_emb,
                threshold=float(options.get("threshold", 0.3)),
                fusion=str(options.get("fusion", "uniform")),
                transform_type=str(options.get("transform_type", "SyNAggro")),
            )
            registered = fill_mask_with_subsegments(
                mask_emb,
                registered,
                max_radius=int(options.get("bfs_radius", 10)),
                min_neighbors=int(options.get("bfs_min_neighbors", 8)),
            )
            np.savez_compressed(
                output, mask=mask_emb, scan=scan_emb, reg=registered, cam=cam_emb
            )
            counts["complete"] += 1
        except Exception as error:
            counts["failed"] += 1
            print(f"Registration failed for {study_id}: {error}", file=sys.stderr)
    return counts


def _selected_stages(first: str, last: str) -> tuple[str, ...]:
    start = STAGES.index(first)
    stop = STAGES.index(last)
    if start > stop:
        raise PipelineConfigError("--from-stage must not come after --through-stage")
    return STAGES[start : stop + 1]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incomplete_messages(summary: dict[str, Any]) -> list[str]:
    messages = []
    stages = summary.get("stages", {})
    conversion = stages.get("convert", {})
    for key in ("missing_dicom", "failed"):
        if conversion.get(key, 0):
            messages.append(f"convert.{key}={conversion[key]}")
    segmentation = stages.get("segment", {})
    for key in ("missing_scan", "failed"):
        if segmentation.get(key, 0):
            messages.append(f"segment.{key}={segmentation[key]}")
    aggregate = stages.get("aggregate", {})
    for key in ("missing_scan", "missing_mask", "missing_all_cams"):
        if aggregate.get(key, 0):
            messages.append(f"aggregate.{key}={aggregate[key]}")
    registration = stages.get("register", {})
    for key in ("missing_scan", "missing_mask", "missing_cam", "failed"):
        if registration.get(key, 0):
            messages.append(f"register.{key}={registration[key]}")
    prediction = stages.get("predict", {})
    expected = int(summary.get("num_studies", 0))
    if prediction and prediction.get("num_successful", 0) != expected:
        messages.append(
            f"predict.num_successful={prediction.get('num_successful', 0)}/{expected}"
        )
    return messages


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    study_ids = _study_ids(config)
    paths = output_paths(config)
    stages = _selected_stages(args.from_stage, args.through_stage)
    print(f"Validated {len(study_ids)} unique studies; stages: {', '.join(stages)}")
    if args.dry_run:
        return

    paths["root"].mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": Path(config["_config_path"]).name,
        "num_studies": len(study_ids),
        "parameters": {
            "dicom_conversion": {"reorient": False},
            "c2_segmentation": config.get("c2_segmentation", {}),
            "registration": config.get("registration", {}),
            "prediction": config.get("prediction", {}),
        },
        "atlas_sha256": [
            _sha256(path) for path in config["inputs"].get("reference_masks", [])
        ],
        "stages": {},
    }
    if "convert" in stages:
        summary["stages"]["convert"] = _convert(config, study_ids, args.overwrite)
    if "segment" in stages:
        summary["stages"]["segment"] = _segment(config, study_ids, args.overwrite)
    if "aggregate" in stages:
        summary["stages"]["aggregate"] = _aggregate(config, study_ids, args.overwrite)
    if "register" in stages:
        summary["stages"]["register"] = _register(config, study_ids, args.overwrite)
    if "predict" in stages:
        summary["stages"]["predict"] = generate_predictions(config)
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    incomplete = _incomplete_messages(summary)
    if incomplete:
        summary["incomplete"] = incomplete
    paths["manifest"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if incomplete:
        raise SystemExit("Pipeline incomplete: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
