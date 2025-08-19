"""Inference and Grad-CAM generation for the RSNA 2022 first-place ensemble."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM

from ..interpolation import cam_to_intermediate_cam
from ..alignment import align_cam_to_scan
from .loader import load_bone_models, load_classifier_models, load_segmentation_models
from .preprocessing import get_cam_trans, get_trans, load_cropped_images, load_dicom_line_par, load_nifti


class _SliceLogitModel(nn.Module):
    """Expose only the slice logits while preserving hooks on the base model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError(f"Grad-CAM wrapper expects (slices,C,H,W); got {inputs.shape}")
        slice_logits, _ = self.model(inputs.unsqueeze(0))
        # A one-item list makes Grad-CAM apply one scalar target to the complete
        # slice vector instead of pairing that target with only its first row.
        return [slice_logits]


class _SumPositiveLogits:
    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        return output.sum()


class FirstPlaceGradCAMGenerator:
    """Exact first-place preprocessing, inference ensemble, and Grad-CAM mapping."""

    def __init__(
        self,
        weights_root: str | Path,
        device: str | torch.device | None = None,
        reduced_ensemble: bool = False,
        low_memory: bool = True,
    ):
        self.weights_root = Path(weights_root).expanduser().resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reduced_ensemble = reduced_ensemble
        self.low_memory = low_memory
        load_device = torch.device("cpu") if low_memory else self.device
        self.segmentation_models = load_segmentation_models(
            self.weights_root, load_device, reduced_ensemble
        )
        self.classifier_models = load_classifier_models(
            self.weights_root, load_device, reduced_ensemble
        )
        self.bone_models = load_bone_models(
            self.weights_root, load_device, reduced_ensemble
        )

    def _activate(self, model: nn.Module) -> nn.Module:
        return model.to(self.device).eval()

    def _amp_context(self):
        if self.low_memory and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _release(self, model: nn.Module) -> None:
        if self.low_memory and self.device.type != "cpu":
            model.to("cpu")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _load_segmentation_input(scan_path: str | Path) -> torch.Tensor:
        path = Path(scan_path)
        if path.is_file() and path.name.endswith((".nii", ".nii.gz")):
            image = load_nifti(str(path))
        elif path.is_dir():
            nifti = next((item for item in path.iterdir() if item.name.endswith((".nii", ".nii.gz"))), None)
            image = load_nifti(str(nifti)) if nifti else load_dicom_line_par(str(path))
        else:
            raise FileNotFoundError(path)
        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32).repeat(3, 0) / 255.0
        return torch.from_numpy(image).unsqueeze(0).float()

    def _segment_and_crop(self, scan_path: str | Path):
        images = self._load_segmentation_input(scan_path)
        mask_sum = None
        with torch.no_grad():
            for model in self.segmentation_models:
                model = self._activate(model)
                with self._amp_context():
                    prediction = model(images.to(self.device)).sigmoid().cpu()
                mask_sum = prediction if mask_sum is None else mask_sum + prediction
                self._release(model)
        mask_np = (mask_sum / len(self.segmentation_models)).numpy()
        cropped, crop_info = load_cropped_images(mask_np[0], str(scan_path))
        classifier_input = cropped.permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
        if self.reduced_ensemble:
            classifier_input = F.interpolate(
                classifier_input.reshape(-1, 6, 512, 512),
                size=256,
                mode="bilinear",
            ).view(1, 105, 6, 256, 256)
            for model in self.classifier_models:
                model.image_size = 256
        return classifier_input, crop_info

    def _predict(self, classifier_input: torch.Tensor) -> dict:
        pred_slice, pred_patient = [], []
        source_size = classifier_input.shape[-1]
        needs_384 = any(model.image_size == 384 for model in self.classifier_models)
        needs_384 = needs_384 or any(model.image_size == 384 for model in self.bone_models)
        bone_input = classifier_input.view(7, 15, 6, source_size, source_size).contiguous()
        classifier_384 = None
        if needs_384:
            classifier_384 = torch.stack(
                [F.interpolate(bone_input[index], size=384, mode="bilinear") for index in range(7)]
            ).view(-1, 105, 6, 384, 384).contiguous()
        with torch.no_grad():
            for model in self.classifier_models:
                model = self._activate(model)
                index = model._ensemble_transform_index
                model_input = classifier_input if model.image_size == source_size else classifier_384
                with self._amp_context():
                    logits, patient = model(get_trans(model_input, index).to(self.device))
                pred_slice.append(logits.sigmoid().view(-1, 7, 15).cpu())
                pred_patient.append(patient.sigmoid().cpu())
                self._release(model)

            bone_384 = classifier_384.view(7, 15, 6, 384, 384) if classifier_384 is not None else None
            for model in self.bone_models:
                model = self._activate(model)
                index = model._ensemble_transform_index
                model_input = bone_input if model.image_size == source_size else bone_384
                with self._amp_context():
                    logits = model(get_trans(model_input, index).to(self.device))
                pred_slice.append(logits.sigmoid().view(-1, 7, 15).cpu())
                self._release(model)

        pred_slice = torch.stack(pred_slice).mean(0)
        pred_patient = torch.stack(pred_patient).mean(0)
        vertebra_probabilities = pred_slice.sort(-1).values[:, :, -5:].mean(-1).clamp(0.0001, 0.9999)
        any_fracture = 1 - torch.prod(1 - pred_slice.sort(-1).values[:, :, -1], 1)
        max_fracture = pred_slice.sort(-1).values[:, :, -1].max(1).values
        patient_probability = (any_fracture * 0.4 + max_fracture * 0.01 + pred_patient.view(-1) * 0.6).clamp(0.0001, 0.9999)
        return {
            "vertebra_probabilities": vertebra_probabilities[0].detach().cpu().numpy(),
            "c2_probability": float(vertebra_probabilities[0, 1]),
            "patient_probability": float(patient_probability[0]),
        }

    def _model_cam(self, model, target_layer, classifier_input, transform_index):
        targets = [_SumPositiveLogits()]
        gradcam_model = _SliceLogitModel(model)
        transformed = get_trans(classifier_input, transform_index)
        cam_input = transformed.reshape(-1, *transformed.shape[2:]).to(self.device)
        with GradCAM(model=gradcam_model, target_layers=[target_layer]) as gradcam:
            with self._amp_context():
                cam = gradcam(input_tensor=cam_input, targets=targets)
        cam = get_cam_trans(torch.as_tensor(cam), transform_index)
        if cam.shape[-1] != 512:
            cam = F.interpolate(cam.unsqueeze(0), size=512, mode="bilinear").squeeze(0)
        return cam.cpu().numpy()

    def _generate_cam(self, classifier_input: torch.Tensor, crop_info: list) -> np.ndarray:
        cudnn_was_enabled = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = False
            grayscale_cam = np.zeros((105, 512, 512), dtype=np.float32)
            source_size = classifier_input.shape[-1]
            classifier_384 = None
            if any(model.image_size == 384 for model in self.classifier_models):
                bone_input = classifier_input.view(7, 15, 6, source_size, source_size)
                classifier_384 = torch.stack(
                    [F.interpolate(bone_input[index], size=384, mode="bilinear") for index in range(7)]
                ).view(-1, 105, 6, 384, 384).contiguous()
            for model in self.classifier_models:
                model = self._activate(model)
                index = model._ensemble_transform_index
                model_input = classifier_input if model.image_size == source_size else classifier_384
                backbone = model._backbone_name
                target_layer = (
                    model.encoder.final_act
                    if "nfnet" in backbone
                    else model.encoder.stages[-1].blocks[-1]
                )
                grayscale_cam += self._model_cam(
                    model, target_layer, model_input, index
                )
                self._release(model)
            grayscale_cam /= len(self.classifier_models)
        finally:
            torch.backends.cudnn.enabled = cudnn_was_enabled

        scan_count = next((info[-1] for info in crop_info if info is not None), None)
        if scan_count is None:
            raise RuntimeError("No vertebral crop was produced; Grad-CAM cannot be mapped")
        full_cam = np.zeros((scan_count, 512, 512), dtype=np.float32)
        for vertebra_index, info in enumerate(crop_info):
            if info is None:
                continue
            x1, x2, y1, y2, z1, z2, _ = info
            full_cam[z1:z2, x1:x2, y1:y2] += cam_to_intermediate_cam(
                grayscale_cam[vertebra_index * 15 : (vertebra_index + 1) * 15],
                z2 - z1,
                z2 - z1,
                x2 - x1,
                y2 - y1,
            )
        return full_cam

    def run(self, scan_path: str | Path) -> dict:
        """Run classifier inference and return probabilities plus a `(Z,Y,X)` CAM."""
        classifier_input, crop_info = self._segment_and_crop(scan_path)
        prediction = self._predict(classifier_input)
        prediction["cam"] = align_cam_to_scan(
            self._generate_cam(classifier_input, crop_info), scan_path
        )
        prediction["ensemble_mode"] = "reduced" if self.reduced_ensemble else "full"
        prediction["low_memory"] = self.low_memory
        prediction["cam_axis_order"] = "ZYX"
        prediction["cam_shape"] = list(prediction["cam"].shape)
        return prediction
