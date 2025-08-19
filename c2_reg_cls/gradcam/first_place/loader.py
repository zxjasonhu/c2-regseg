"""Checkpoint loaders for the RSNA 2022 first-place ensemble."""

from __future__ import annotations

from pathlib import Path

import torch


SEGMENTATION_SPECS = (
    (
        "seg-v2s-0911",
        "timm3d_v2s_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_mixup1_lr1e3_20x50ep",
        "tf_efficientnetv2_s_in21ft1k",
        range(5),
    ),
    (
        "segres18d0920",
        "timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep",
        "resnet18d",
        range(5),
    ),
)

CLASSIFIER_SPECS = (
    (
        "cls-convnn512-1011",
        "0920_2d_lstmv22headv2_convnn_512_15_6ch_lossv3_augv2_mixupv3p5_dpr3_drl3_rov1p2_rov3p2_bs4_lr6e6_eta6e6_lw151_50ep_ddp",
        "convnext_nano",
        512,
        range(5),
    ),
    (
        "clsconvnpc5121023",
        "0920_2d_lstmv22headv2_convnpc_512_15_6ch_lossv3_augv2_mixupv3p5_dpr3_drl3_rov1p2_rov3p2_bs4_lr6e6_eta6e6_lw151_50ep_ddp",
        "convnext_pico_ols",
        512,
        (3, 4),
    ),
    (
        "clsconvnt1019",
        "0920_2d_lstmv22headv2_convnt_384_15_6ch_lossv3_augv2_mixupv3p5_dpr3_drl3_rov1p2_rov3p2_bs4_lr10e6_eta10e6_lw151_50ep_ddp",
        "convnext_tiny_in22ft1k",
        384,
        (5, 6),
    ),
    (
        "clsnfl03841026",
        "0920_2d_lstmv22headv2_nfl0_384_15_6ch_lossv3_augv2_mixupv3p5_dpr3_drl3_rov1p2_rov3p2_bs4_lr15e6_eta15e6_lw151_50ep_ddp",
        "eca_nfnet_l0",
        384,
        (1, 9),
    ),
)

BONE_SPECS = (
    (
        "cls-1bone-v2b3-512-1017",
        "0920_1bonev2_effv2s_512_15_6ch_augv2_mixupp6_dpr2_drl3_rov1p2_bs7_lr20e6_eta20e6_75ep",
        "tf_efficientnetv2_s_in21ft1k",
        512,
        range(5),
    ),
    (
        "cls-1bone-convnt-384-1021",
        "0920_1bonev2_convnt_384_15_6ch_augv2_mixupp5_dpr3_drl3_rov1p2_bs21_lr14e6_eta14e6_50ep_dp",
        "convnext_tiny_in22ft1k",
        384,
        (0, 2, 5, 8, 9),
    ),
)


def _checkpoint(root: Path, directory: str, kernel: str, fold: int) -> Path:
    path = root / directory / f"{kernel}_fold{fold}_best.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Missing first-place checkpoint: {path}")
    return path


def _load_state(model, path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_segmentation_models(
    weights_root: str | Path,
    device: torch.device,
    reduced_ensemble: bool = False,
):
    from .models import TimmSegModel, convert_3d

    root = Path(weights_root).expanduser().resolve()
    models = []
    specs = SEGMENTATION_SPECS[:1] if reduced_ensemble else SEGMENTATION_SPECS
    for directory, kernel, backbone, folds in specs:
        selected_folds = tuple(folds)[:1] if reduced_ensemble else folds
        for fold in selected_folds:
            model = convert_3d(TimmSegModel(backbone, pretrained=False))
            models.append(_load_state(model, _checkpoint(root, directory, kernel, fold), device))
    return models


def load_classifier_models(
    weights_root: str | Path,
    device: torch.device,
    reduced_ensemble: bool = False,
):
    from .models import TimmModel

    root = Path(weights_root).expanduser().resolve()
    models = []
    ensemble_index = 0
    specs = CLASSIFIER_SPECS[:1] if reduced_ensemble else CLASSIFIER_SPECS
    for directory, kernel, backbone, image_size, folds in specs:
        selected_folds = tuple(folds)[:1] if reduced_ensemble else folds
        for fold in selected_folds:
            model = TimmModel(backbone, image_size=image_size, pretrained=False)
            model._ensemble_transform_index = ensemble_index
            model._backbone_name = backbone
            models.append(_load_state(model, _checkpoint(root, directory, kernel, fold), device))
            ensemble_index += 1
    return models


def load_bone_models(
    weights_root: str | Path,
    device: torch.device,
    reduced_ensemble: bool = False,
):
    from .models import Timm1BoneModel

    if reduced_ensemble:
        return []
    root = Path(weights_root).expanduser().resolve()
    models = []
    for directory, kernel, backbone, image_size, folds in BONE_SPECS:
        for fold in folds:
            model = Timm1BoneModel(backbone, image_size=image_size, pretrained=False)
            model._ensemble_transform_index = len(models)
            models.append(_load_state(model, _checkpoint(root, directory, kernel, fold), device))
    return models


def expected_checkpoint_paths(
    weights_root: str | Path,
    reduced_ensemble: bool = False,
) -> list[Path]:
    """Return checkpoints required for the full or reduced visual ensemble."""
    root = Path(weights_root).expanduser().resolve()
    if reduced_ensemble:
        selected = (SEGMENTATION_SPECS[0], CLASSIFIER_SPECS[0])
        return [
            root / directory / f"{kernel}_fold{tuple(folds)[0]}_best.pth"
            for directory, kernel, *_middle, folds in selected
        ]
    specs = (*SEGMENTATION_SPECS, *CLASSIFIER_SPECS, *BONE_SPECS)
    return [
        root / directory / f"{kernel}_fold{fold}_best.pth"
        for directory, kernel, *_middle, folds in specs
        for fold in folds
    ]
