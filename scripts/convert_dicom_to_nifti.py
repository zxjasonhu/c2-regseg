#!/usr/bin/env python
"""Convert one DICOM CT series with the historical non-reoriented convention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2_reg_cls import convert_dicom_to_nifti, validate_nifti_dicom_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", required=True, help="Directory containing one CT series")
    parser.add_argument("--output", required=True, help="Output CT .nii or .nii.gz path")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = convert_dicom_to_nifti(
        args.dicom_dir, args.output, overwrite=args.overwrite
    )
    report = validate_nifti_dicom_alignment(args.dicom_dir, output)
    report["nifti_path"] = str(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
