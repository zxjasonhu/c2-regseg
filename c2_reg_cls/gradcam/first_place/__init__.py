"""RSNA 2022 first-place inference and Grad-CAM support."""

__all__ = ["FirstPlaceGradCAMGenerator", "expected_checkpoint_paths"]


def __getattr__(name):
    if name == "FirstPlaceGradCAMGenerator":
        from .generator import FirstPlaceGradCAMGenerator

        return FirstPlaceGradCAMGenerator
    if name == "expected_checkpoint_paths":
        from .loader import expected_checkpoint_paths

        return expected_checkpoint_paths
    raise AttributeError(name)
