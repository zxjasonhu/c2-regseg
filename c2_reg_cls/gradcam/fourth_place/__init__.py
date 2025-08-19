"""RSNA 2022 fourth-place inference and Grad-CAM support."""

from .checkpoints import expected_checkpoint_paths

__all__ = ["FourthPlaceGradCAMGenerator", "expected_checkpoint_paths"]


def __getattr__(name):
    if name == "FourthPlaceGradCAMGenerator":
        from .generator import FourthPlaceGradCAMGenerator

        return FourthPlaceGradCAMGenerator
    raise AttributeError(name)
