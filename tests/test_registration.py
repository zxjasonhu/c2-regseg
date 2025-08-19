import numpy as np
import nibabel as nib

from c2_reg_cls.registration import core
from c2_reg_cls.registration import normalize_mask, restore_labels_to_original_space


class _ArrayImage:
    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array


def test_uniform_fusion_is_default_and_dice_weighting_remains_available(monkeypatch):
    target = np.ones((1, 1, 4), dtype=np.float32)
    references = []
    for index in range(3):
        binary = np.ones_like(target) if index == 0 else np.array([[[1, 0, 0, 0]]])
        anatomy = np.full_like(target, 3 if index == 0 else 4)
        references.append(np.stack([binary, anatomy]))

    monkeypatch.setattr(
        core,
        "_register",
        lambda reference, _target, _transform: {"warpedmovout": _ArrayImage(reference)},
    )
    monkeypatch.setattr(
        core,
        "_warp",
        lambda mask, _registration, _target: np.asarray(mask, dtype=np.float32),
    )

    uniform = core.register_subsegment_mask_multi(references, target, labels=[3, 4])
    weighted = core.register_subsegment_mask_multi(
        references, target, labels=[3, 4], fusion="dice_weighted"
    )

    assert np.all(uniform == 4)
    assert np.all(weighted == 3)


def test_registered_labels_restore_to_original_ct_grid():
    mask = np.zeros((28, 30, 32), dtype=np.uint8)
    mask[5:20, 7:22, 9:24] = 1
    image = nib.Nifti1Image(mask, np.diag([1.2, 0.8, 1.5, 1.0]))

    normalized, _ = normalize_mask(image)
    normalized_zyx = normalized.transpose(2, 1, 0)[:, ::-1, :]
    registered = np.where(normalized_zyx > 0, 8, 0)
    restored = restore_labels_to_original_space(image, registered)

    assert restored.shape == mask.shape
    assert set(np.unique(restored)) <= {0, 8}
    assert not np.any(restored[mask == 0])
    assert np.count_nonzero(restored[mask > 0]) / np.count_nonzero(mask) > 0.8
