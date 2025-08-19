#!/usr/bin/env python
"""Run one released challenge model and save its prediction and Grad-CAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from c2_reg_cls import project_root, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("first_place", "fourth_place"))
    parser.add_argument("--scan", required=True, help="Study directory (DICOM or a directory containing NIfTI)")
    parser.add_argument("--weights-root", required=True, help="Downloaded checkpoint root")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu")
    parser.add_argument(
        "--reduced-ensemble",
        action="store_true",
        help="Visual check only: load one segmenter and one classifier",
    )
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream models through the device one at a time (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan = resolve_project_path(args.scan)
    weights = resolve_project_path(args.weights_root)
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "first_place":
        from c2_reg_cls.gradcam.first_place import FirstPlaceGradCAMGenerator

        generator = FirstPlaceGradCAMGenerator(
            weights,
            device=args.device,
            reduced_ensemble=args.reduced_ensemble,
            low_memory=args.low_memory,
        )
    else:
        from c2_reg_cls.gradcam.fourth_place import FourthPlaceGradCAMGenerator

        generator = FourthPlaceGradCAMGenerator(
            weights,
            device=args.device,
            reduced_ensemble=args.reduced_ensemble,
            low_memory=args.low_memory,
        )

    result = generator.run(scan)
    raw_cam = result.pop("cam")
    if raw_cam is None:
        raise RuntimeError("The model returned no Grad-CAM volume")
    cam = np.asarray(raw_cam, dtype=np.float32)
    if cam.ndim != 3:
        raise RuntimeError(f"Expected a 3-D Grad-CAM volume; got shape {cam.shape}")
    serializable = {
        key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }
    np.savez_compressed(output, cam=cam)
    output.with_suffix(".json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    try:
        display_output = str(output.relative_to(project_root()))
    except ValueError:
        display_output = str(output)
    print(json.dumps({"cam": display_output, **serializable}, indent=2))


if __name__ == "__main__":
    main()
