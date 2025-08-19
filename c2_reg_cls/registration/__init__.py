"""Public registration operations used by the reviewer pipeline."""

from .core import register_subsegment_mask_multi
from .pipeline import load_reference_masks, normalize_mask
from .preprocessing import restore_labels_to_original_space
from .segmentation import fill_mask_with_subsegments

__all__ = [
    "fill_mask_with_subsegments",
    "load_reference_masks",
    "normalize_mask",
    "register_subsegment_mask_multi",
    "restore_labels_to_original_space",
]
