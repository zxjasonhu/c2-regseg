"""Checkpoint layout for the RSNA 2022 fourth-place model."""

from pathlib import Path


def expected_checkpoint_paths(
    weights_root: str | Path,
    reduced_ensemble: bool = False,
) -> list[Path]:
    """Return checkpoints required for the full or reduced visual ensemble."""
    root = Path(weights_root).expanduser().resolve() / "checkpoints"
    paths = [
        root / "swa_3_best_ClassifierResNet3dCSN2P1D_r152ir_0.pth",
        root / "swa_5_best_ClassifierResNet3dCSN2P1D_r152ir_1.pth",
        root / "swa_5_best_ClassifierResNet3dCSN2P1D_r152ir_2.pth",
        root / "swa_5_best_ClassifierResNet3dCSN2P1D_r152ir_3.pth",
        root / "256_ResNet3dCSN2P1D_r50ir_1_dice",
    ]
    return [paths[0], paths[-1]] if reduced_ensemble else paths
