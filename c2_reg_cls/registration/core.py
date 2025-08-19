"""Multi-atlas registration with uniform or Dice-weighted label fusion."""

import numpy as np

from ..config import ALL_LABELS
from .metrics import dice_score


def _ants():
    import ants

    return ants


def _binarize(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.float32)


def _register(reference: np.ndarray, target: np.ndarray, transform_type: str) -> dict:
    ants = _ants()
    return ants.registration(
        fixed=ants.from_numpy(target.astype(np.float32)),
        moving=ants.from_numpy(reference.astype(np.float32)),
        type_of_transform=transform_type,
    )


def _warp(mask: np.ndarray, registration: dict, target: np.ndarray) -> np.ndarray:
    ants = _ants()
    return ants.apply_transforms(
        fixed=ants.from_numpy(target.astype(np.float32)),
        moving=ants.from_numpy(mask.astype(np.float32)),
        transformlist=registration["fwdtransforms"],
    ).numpy()


def register_subsegment_mask_multi(
    references: list[np.ndarray],
    target: np.ndarray,
    labels: list[int] | None = None,
    threshold: float = 0.3,
    fusion: str = "uniform",
    transform_type: str = "SyNAggro",
) -> np.ndarray:
    """Register all atlases and fuse their anatomical label probabilities.

    ``uniform`` (the default) gives every atlas equal weight, i.e. plain
    averaging. ``dice_weighted`` weights each atlas by its binary registration
    Dice score before averaging.
    """
    if fusion not in {"uniform", "dice_weighted"}:
        raise ValueError("fusion must be 'uniform' or 'dice_weighted'")
    if not references:
        raise ValueError("At least one atlas is required")
    labels = list(labels or ALL_LABELS)
    binary_target = _binarize(target)

    registrations, anatomies, scores = [], [], []
    for reference in references:
        if not isinstance(reference, np.ndarray) or reference.ndim != 4 or reference.shape[0] != 2:
            raise ValueError("Each atlas must have shape (2,Z,Y,X)")
        binary_reference, anatomy = reference
        registration = _register(_binarize(binary_reference), binary_target, transform_type)
        registrations.append(registration)
        anatomies.append(anatomy)
        scores.append(dice_score(binary_target, registration["warpedmovout"].numpy()))

    if fusion == "uniform":
        weights = np.full(len(references), 1 / len(references), dtype=np.float64)
    else:
        weights = np.asarray(scores, dtype=np.float64)
        total = float(weights.sum())
        weights = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))

    votes = np.zeros((len(labels) + 1, *binary_target.shape), dtype=np.float64)
    for weight, registration, anatomy in zip(weights, registrations, anatomies):
        for index, label in enumerate(labels, start=1):
            votes[index] += weight * _warp(anatomy == label, registration, binary_target)
    votes[votes < threshold] = 0
    indices = np.argmax(votes, axis=0)
    result = np.zeros(binary_target.shape, dtype=np.int32)
    for index, label in enumerate(labels, start=1):
        result[indices == index] = label
    return result
