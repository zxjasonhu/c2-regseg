"""Load and validate the reviewer-facing JSON pipeline configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PipelineConfigError(ValueError):
    """Raised when a pipeline configuration is incomplete or inconsistent."""


PATH_FIELDS = {
    "studies_csv",
    "dicom_base",
    "scan_base",
    "mask_base",
    "output_root",
}


def project_root(config_path: Path | None = None) -> Path:
    """Return the standalone release root without relying on the caller's CWD."""
    if config_path is not None:
        for candidate in (config_path.parent, *config_path.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "c2_reg_cls"
            ).is_dir():
                return candidate.resolve()
    return Path(__file__).resolve().parent.parent


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config and resolve every declared path relative to it."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if config.get("schema_version") != 1:
        raise PipelineConfigError("schema_version must be 1")

    inputs = config.setdefault("inputs", {})
    outputs = config.setdefault("outputs", {})
    config_dir = config_path.parent
    root = project_root(config_path)
    path_base = config.get("path_base", "project_root")
    if path_base not in {"project_root", "config_dir"}:
        raise PipelineConfigError("path_base must be 'project_root' or 'config_dir'")
    base_dir = root if path_base == "project_root" else config_dir

    for key in PATH_FIELDS - {"output_root"}:
        if inputs.get(key):
            inputs[key] = _resolve_path(inputs[key], base_dir)
    if outputs.get("output_root"):
        outputs["output_root"] = _resolve_path(outputs["output_root"], base_dir)

    for key in ("reference_masks", "model_heatmap_dirs"):
        if inputs.get(key):
            inputs[key] = [_resolve_path(item, base_dir) for item in inputs[key]]

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_dir)
    config["_project_root"] = str(root)
    return config


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], section: str) -> None:
    """Raise a readable error when required configuration keys are absent."""
    missing = [key for key in keys if mapping.get(key) in (None, "", [])]
    if missing:
        raise PipelineConfigError(
            f"Missing required {section} setting(s): {', '.join(missing)}"
        )


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Return the standard pipeline output directories."""
    outputs = config["outputs"]
    require_keys(outputs, ("output_root",), "outputs")
    root = Path(outputs["output_root"])
    return {
        "root": root,
        "aggregated_heatmaps": root
        / outputs.get("aggregated_heatmaps", "aggregated_heatmaps"),
        "registered": root / outputs.get("registered", "registered"),
        "predictions": root / outputs.get("predictions", "predictions.csv"),
        "manifest": root / outputs.get("manifest", "run_manifest.json"),
    }
