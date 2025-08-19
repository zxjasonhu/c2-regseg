"""Shared fixtures for c2-reg-cls tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_cam(rng):
    """96³ synthetic GradCAM with two hot-spots at known anatomical locations."""
    cam = np.zeros((96, 96, 96), dtype=np.float32)

    # Hot-spot 1: centred at (30, 48, 48), high intensity
    z, y, x = np.ogrid[:96, :96, :96]
    d1 = np.sqrt((z - 30) ** 2 + (y - 48) ** 2 + (x - 48) ** 2)
    cam += np.exp(-d1 ** 2 / (2 * 8 ** 2)).astype(np.float32)

    # Hot-spot 2: centred at (60, 48, 60), medium intensity
    d2 = np.sqrt((z - 60) ** 2 + (y - 48) ** 2 + (x - 60) ** 2)
    cam += 0.6 * np.exp(-d2 ** 2 / (2 * 6 ** 2)).astype(np.float32)

    cam /= cam.max()
    return cam


@pytest.fixture
def synthetic_segment_mask():
    """96³ anatomical label map with 3 distinct regions."""
    mask = np.zeros((96, 96, 96), dtype=np.int32)

    # Label 5 (Odontoid Body): z=20-40, covers hot-spot 1
    mask[20:40, 35:65, 35:65] = 5

    # Label 6 (Lateral Mass R): z=50-70, covers hot-spot 2
    mask[50:70, 35:65, 45:75] = 6

    # Label 14 (Pars R): z=70-85, no significant activation
    mask[70:85, 40:60, 40:60] = 14

    return mask


@pytest.fixture
def synthetic_binary_mask(synthetic_segment_mask):
    """Binary foreground mask derived from the segment mask."""
    return (synthetic_segment_mask > 0).astype(np.uint8)
