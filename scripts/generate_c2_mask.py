#!/usr/bin/env python
"""Generate one aligned binary C2 mask with TotalSegmentator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2_reg_cls import generate_c2_mask, validate_c2_mask_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Converted source CT NIfTI")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path ending in segmentations/vertebrae_C2.nii.gz",
    )
    parser.add_argument("--device", default="gpu", help="gpu, gpu:N, cpu, or mps")
    parser.add_argument("--force-split", action="store_true", help="Process a large CT in chunks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_path = generate_c2_mask(
        args.input,
        args.output,
        device=args.device,
        force_split=args.force_split,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    report = validate_c2_mask_alignment(args.input, mask_path)
    report["mask_path"] = str(mask_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
