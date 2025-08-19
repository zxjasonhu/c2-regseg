"""Public interface for the standalone C2 registration/localization release."""

from .c2_mask import (
    convert_dicom_to_nifti,
    generate_c2_mask,
    validate_c2_mask_alignment,
    validate_nifti_dicom_alignment,
)

from .api import (
    aggregate_model_cams,
    anatomical_groups_for_labels,
    default_atlas_paths,
    load_prediction_config,
    localize_fracture_regions,
    localize_registered_archive,
    project_root,
    register_c2_subregions,
    resolve_project_path,
    restore_c2_subregions_to_original,
)

__version__ = "0.2.0"

__all__ = [
    "aggregate_model_cams",
    "anatomical_groups_for_labels",
    "default_atlas_paths",
    "convert_dicom_to_nifti",
    "generate_c2_mask",
    "load_prediction_config",
    "localize_fracture_regions",
    "localize_registered_archive",
    "project_root",
    "register_c2_subregions",
    "resolve_project_path",
    "restore_c2_subregions_to_original",
    "validate_c2_mask_alignment",
    "validate_nifti_dicom_alignment",
]
