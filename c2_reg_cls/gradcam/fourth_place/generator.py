import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import albumentations
import cv2
import nibabel as nib
import numpy as np
import pydicom
import torch
from albumentations import ReplayCompose

from ..interpolation import interpolate_cam_on_voxel
from ..alignment import align_cam_to_scan

from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from skimage import measure
from torch import Tensor

from .zoo import ClassifierResNet3dCSN2P1D, ResNet3dCSN2P1D
from .checkpoints import expected_checkpoint_paths

from pytorch_grad_cam import HiResCAM

import gc


configs = {
    "classifier_csn_ir152": {
        "network": ClassifierResNet3dCSN2P1D,
        "encoder_params": {"encoder": "r152ir", "num_classes": 8, "pool": "max"},
    },
    "segmentor_csn_ir152": {
        "network": ResNet3dCSN2P1D,
        "encoder_params": {"encoder": "r50ir"},
    },
}


@dataclass
class BatchSlice:
    i_from: int
    i_to: int
    i_start: int


def get_slices(
    batch: Tensor, dim=1, window: int = 16, overlap: int = 8
) -> List[BatchSlice]:
    num_imgs = batch.size(dim)
    if num_imgs <= window:
        return [BatchSlice(0, num_imgs, 0)]
    stride = window - overlap
    result = []
    current_idx = 0
    while True:
        next_idx = current_idx + window

        if next_idx >= num_imgs:
            current_idx = num_imgs - window
            offset = overlap // 2 if current_idx > 0 else 0
            next_idx = num_imgs
            result.append(BatchSlice(current_idx, next_idx, offset))
            break
        else:
            offset = overlap // 2 if current_idx > 0 else 0
            result.append(BatchSlice(current_idx, next_idx, offset))
        current_idx += stride
    return result


def load_checkpoint(model, checkpoint_path, strict=False, verbose=True):
    if verbose:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
        orig_state_dict = model.state_dict()
        mismatched_keys = []
        for k, v in state_dict.items():
            ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
            if v.size() != ori_size:
                if verbose:
                    print(
                        "SKIPPING!!! Shape of {} changed from {} to {}".format(
                            k, v.size(), ori_size
                        )
                    )
                mismatched_keys.append(k)
        for k in mismatched_keys:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
        del orig_state_dict
        if verbose:
            epoch = checkpoint.get("epoch", "not recorded")
            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
    else:
        model.load_state_dict(checkpoint)
    del checkpoint


def load_model(conf: Dict, checkpoint: str):
    model = conf["network"](**conf["encoder_params"])
    load_checkpoint(model, checkpoint)
    return model.eval()


crop_augs = albumentations.ReplayCompose(
    [
        albumentations.LongestMaxSize(256),
        albumentations.PadIfNeeded(
            256, 256, border_mode=cv2.BORDER_CONSTANT, value=0
        ),
    ]
)


class FourthPlaceGradCAMGenerator:
    """Original fourth-place inference and HiResCAM pipeline with configurable weights."""

    def __init__(
        self,
        weights_root: str | Path,
        device: str | torch.device | None = None,
        reduced_ensemble: bool = False,
        low_memory: bool = True,
    ):
        root_path = Path(weights_root).expanduser().resolve()
        checkpoint_root = root_path / "checkpoints"
        required = expected_checkpoint_paths(root_path, reduced_ensemble)
        missing = [path for path in required if not path.is_file()]
        if missing:
            listing = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing fourth-place checkpoints:\n{listing}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.reduced_ensemble = reduced_ensemble
        self.low_memory = low_memory

        classifier_paths = expected_checkpoint_paths(root_path)[:-1]
        if reduced_ensemble:
            classifier_paths = classifier_paths[:1]
        cls_models = [
            load_model(configs["classifier_csn_ir152"], str(path))
            for path in classifier_paths
        ]
        load_device = torch.device("cpu") if low_memory else self.device
        self.cls_models = [model.to(load_device) for model in cls_models]
        self.seg_model = load_model(
            configs["segmentor_csn_ir152"],
            os.path.join(
                checkpoint_root,
                "256_ResNet3dCSN2P1D_r50ir_1_dice",
            ),
        ).to(load_device)

    def _activate(self, model):
        return model.to(self.device).eval()

    def _amp_context(self):
        if self.low_memory and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _release(self, model):
        if self.low_memory and self.device.type != "cpu":
            model.to("cpu")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _combine_scan(self, data, size=512, fix_monochrome: bool = True) -> np.ndarray:
        images = []
        first = None
        last = None
        dicoms = []

        scan_dir = Path(data["scan_dir"])
        nifti_path = (
            scan_dir
            if scan_dir.is_file() and scan_dir.name.endswith((".nii", ".nii.gz"))
            else next(
                (
                    item
                    for item in sorted(scan_dir.iterdir())
                    if item.name.endswith((".nii", ".nii.gz"))
                ),
                None,
            )
        )
        if nifti_path is not None:
            volume = nib.load(str(nifti_path)).get_fdata().transpose(2, 1, 0)[::-1]
            return np.asarray(
                [cv2.resize(image, (size, size)) for image in volume[::2]]
            )

        for file in os.listdir(scan_dir):
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(scan_dir, file))
                dicoms.append(ds)

        dicoms.sort(key=lambda x: int(x.InstanceNumber))

        #  ::2 is important,
        #  as models were trained with stride 2, as augmentation one can use two predicts with 1::2 and  ::2 for an ensemble
        for ds in dicoms[::2]:
            if not first:
                first = ds
            last = ds

            data = ds.pixel_array
            data = cv2.resize(data, (size, size))
            if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data
            images.append(data)

        if first and last:
            if last.ImagePositionPatient[2] > first.ImagePositionPatient[2]:
                images = images[::-1]
        return np.array(images)

    def _get_image_cube_segmentation(self, data) -> Dict:
        image_cube = self._combine_scan(data, size=256)
        image_mean = image_cube.mean()
        image_std = image_cube.std()
        h = image_cube.shape[0]

        images = image_cube
        if h % 32 > 0:
            tmp = np.zeros(((h // 32 + 1) * 32, 256, 256))
            tmp[:h] = images
            images = tmp
        images = (images - image_mean) / image_std
        images = np.expand_dims(images, 0)
        sample = {}
        sample["image"] = torch.from_numpy(images).float()
        sample["h"] = h
        return sample

    def _get_image_cubes_classification(
        self, data, mask_cube: np.ndarray
    ) -> torch.Tensor:
        image_cube = self._combine_scan(data, size=512)
        boxes = {}
        for rprop in measure.regionprops(mask_cube):
            boxes[rprop.label] = rprop.bbox, rprop.area

        image_mean = image_cube.mean()
        image_std = image_cube.std()
        slice_size = 40
        all_images = []
        for li in range(1, 8):
            if li not in boxes:
                all_images.append(np.zeros((3, slice_size, 256, 256)))
            else:
                bbox, area = boxes[li]
                z1, z2 = bbox[0], bbox[3]
                y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
                x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
                if z2 - z1 < slice_size:
                    diff = (slice_size - z2 + z1) // 2
                    z1 = max(0, z1 - diff)
                    z2 = z1 + slice_size
                images = image_cube[z1:z2, y1 * 2 : y2 * 2, x1 * 2 : x2 * 2].copy()
                masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()

                replay = None
                image_crops = []
                mask_crops = []
                for i in range(images.shape[0]):
                    image = images[i]
                    mask = masks[i]
                    (
                        h,
                        w,
                    ) = mask.shape
                    mask = cv2.resize(
                        mask, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST
                    )
                    if replay is None:
                        sample = crop_augs(image=image, mask=mask)
                        replay = sample["replay"]
                    else:
                        sample = ReplayCompose.replay(replay, image=image, mask=mask)
                    image_ = sample["image"]
                    image_crops.append(image_)
                    mask_crops.append(sample["mask"])
                images = np.array(image_crops).astype(np.float32)
                masks = np.array(mask_crops).astype(np.float32)
                images = np.expand_dims(images, -1)
                masks = np.expand_dims(masks, -1)
                images = (images - image_mean) / image_std

                images = np.concatenate([images, images, masks], axis=-1)
                h = images.shape[0]
                if h > slice_size:
                    images = images[:slice_size]
                    all_images.append(np.moveaxis(images, -1, 0))
                    images = images[-slice_size:]
                    all_images.append(np.moveaxis(images, -1, 0))
                else:
                    if h != slice_size:
                        print("h != slice_size", h, slice_size)
                        tmp = np.zeros((slice_size, *images.shape[1:]))
                        tmp[:h] = images
                        images = tmp
                    all_images.append(np.moveaxis(images, -1, 0))

        sample = {}
        sample["image"] = torch.from_numpy(np.array(all_images)).float()
        return sample

    def _predict_seg_mask(self, data) -> np.ndarray:
        sample = self._get_image_cube_segmentation(data)

        image = sample["image"]
        h = int(sample["h"])
        imgs = image.cpu().float().unsqueeze(0)

        case_preds = np.zeros((imgs.shape[2], 256, 256), dtype=np.float32)

        self.seg_model = self._activate(self.seg_model)
        with torch.no_grad():
            window, overlap = (64, 32) if self.low_memory else (256, 128)
            slices = get_slices(imgs, dim=2, window=window, overlap=overlap)
            for slice in slices:
                batch = imgs[:, :, slice.i_from : slice.i_to].to(self.device).float()
                with self._amp_context():
                    preds = torch.softmax(self.seg_model(batch)["mask"], dim=1)[0]
                preds = torch.argmax(preds, dim=0)
                preds = preds.cpu().numpy()

                for pred_idx in range(slice.i_start, preds.shape[0]):
                    idx = slice.i_from + pred_idx
                    y_pred = preds[pred_idx]
                    case_preds[idx] = y_pred[:, :]
        self._release(self.seg_model)
        case_preds = np.array(case_preds)[:h]
        case_preds = case_preds.astype(np.uint8)
        return case_preds

    def _get_image_cubes_classification_explain(
        self, data, mask_cube: np.ndarray
    ) -> torch.Tensor:
        image_cube = self._combine_scan(data, size=512)
        boxes = {}
        for rprop in measure.regionprops(mask_cube):
            boxes[rprop.label] = rprop.bbox, rprop.area

        image_mean = image_cube.mean()
        image_std = image_cube.std()
        slice_size = 40
        all_images = []
        all_images_info = []
        for li in range(1, 8):
            if li not in boxes:
                all_images_info.append(
                    {
                        "type": "empty",
                        "total_slice": 0,
                        "current_index": len(all_images),
                        "z1": -1,
                        "z2": -1,
                        "x1": -1,
                        "x2": -1,
                        "y1": -1,
                        "y2": -1,
                    }
                )
                all_images.append(np.zeros((3, slice_size, 256, 256)))
            else:
                bbox, area = boxes[li]
                z1, z2 = bbox[0], bbox[3]
                y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
                x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
                if z2 - z1 < slice_size:
                    diff = (slice_size - z2 + z1) // 2
                    z1 = max(0, z1 - diff)
                    z2 = z1 + slice_size

                images = image_cube[z1:z2, y1 * 2 : y2 * 2, x1 * 2 : x2 * 2].copy()
                masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()
                slice_size = slice_size

                replay = None
                image_crops = []
                mask_crops = []
                for i in range(images.shape[0]):
                    image = images[i]
                    mask = masks[i]
                    (
                        h,
                        w,
                    ) = mask.shape
                    mask = cv2.resize(
                        mask, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST
                    )
                    if replay is None:
                        sample = crop_augs(image=image, mask=mask)
                        replay = sample["replay"]
                    else:
                        sample = ReplayCompose.replay(replay, image=image, mask=mask)
                    image_ = sample["image"]
                    image_crops.append(image_)
                    mask_crops.append(sample["mask"])
                images = np.array(image_crops).astype(np.float32)
                masks = np.array(mask_crops).astype(np.float32)
                images = np.expand_dims(images, -1)
                masks = np.expand_dims(masks, -1)
                images = (images - image_mean) / image_std

                images = np.concatenate([images, images, masks], axis=-1)
                h = images.shape[0]
                if h > slice_size:
                    all_images_info.append(
                        {
                            "type": "overlap",
                            "total_slice": slice_size,
                            "current_index": len(all_images),
                            "z1": z1,
                            "z2": z1 + slice_size,
                            "x1": x1 * 2,
                            "x2": x2 * 2,
                            "y1": y1 * 2,
                            "y2": y2 * 2,
                        }
                    )
                    _images = images[:slice_size]
                    all_images.append(np.moveaxis(_images, -1, 0))
                    all_images_info.append(
                        {
                            "type": "overlap",
                            "total_slice": slice_size,
                            "current_index": len(all_images),
                            "z1": z2 - slice_size,
                            "z2": z2,
                            "x1": x1 * 2,
                            "x2": x2 * 2,
                            "y1": y1 * 2,
                            "y2": y2 * 2,
                        }
                    )
                    _images = images[-slice_size:]
                    all_images.append(np.moveaxis(_images, -1, 0))
                else:
                    if h != slice_size:
                        all_images_info.append(
                            {
                                "type": "padding",
                                "total_slice": h,
                                "current_index": len(all_images),
                                "z1": z1,
                                "z2": z1 + h,
                                "x1": x1 * 2,
                                "x2": x2 * 2,
                                "y1": y1 * 2,
                                "y2": y2 * 2,
                            }
                        )
                        print("h != slice_size", h, slice_size)
                        tmp = np.zeros((slice_size, *images.shape[1:]))
                        tmp[:h] = images
                        images = tmp
                    else:
                        all_images_info.append(
                            {
                                "type": "normal",
                                "total_slice": h,
                                "current_index": len(all_images),
                                "z1": z1,
                                "z2": z2,
                                "x1": x1 * 2,
                                "x2": x2 * 2,
                                "y1": y1 * 2,
                                "y2": y2 * 2,
                            }
                        )
                    all_images.append(np.moveaxis(images, -1, 0))
        sample = {}
        sample["image"] = torch.from_numpy(np.array(all_images)).float()
        sample["info"] = all_images_info
        sample["image_cube"] = image_cube
        return sample

    def _predict_cls(self, data, mask_cube):
        sample = self._get_image_cubes_classification(data, mask_cube)
        image = sample["image"]
        imgs = image.cpu().float()

        def predict_model(model):
            preds = []
            with torch.no_grad():
                for i in range(len(imgs)):
                    with self._amp_context():
                        output = model(imgs[i : i + 1].to(self.device))
                    pred_slice = (
                        torch.sigmoid(output.float()).cpu().numpy().astype(np.float32)
                    )
                    with self._amp_context():
                        output = model(
                            torch.flip(imgs[i : i + 1].to(self.device), dims=(-1,))
                        )
                    pred_slice += (
                        torch.sigmoid(output.float()).cpu().numpy().astype(np.float32)
                    )
                    pred_slice /= 2
                    preds.append(pred_slice)
            preds = np.max(np.array(preds), axis=0)
            preds[np.isnan(preds)] = 0.01
            return preds

        with torch.no_grad():
            preds = []
            for model in self.cls_models:
                model = self._activate(model)
                preds.append(predict_model(model))
                self._release(model)
            preds = np.average(np.array(preds), axis=0)
            preds = np.clip(preds, 0.01, 0.99)
        return preds

    def explain_gradcam(self, data, mask_cube=None):
        if isinstance(data, (str, Path)):
            data = {"scan_dir": str(data)}
        sample = self._get_image_cubes_classification_explain(
            data, mask_cube if mask_cube is not None else self._predict_seg_mask(data)
        )
        input_tensor = sample["image"]
        image_cube = sample["image_cube"]
        input_tensor_info = sample["info"]

        targets = [BinaryClassifierOutputTarget(1)]

        full_mask = None
        for model in self.cls_models:
            model = self._activate(model)
            target_layers = [model.backbone.layer4[-1]]
            with HiResCAM(model=model, target_layers=target_layers) as cam:
                # One crop at a time keeps 3-D activation tensors below the
                # memory required by the original all-crops batch.
                crop_cams = []
                for index in range(len(input_tensor)):
                    with self._amp_context():
                        crop_cam = cam(
                            input_tensor=input_tensor[index : index + 1].to(self.device),
                            targets=targets,
                        )
                    crop_cams.append(crop_cam[0])
                grayscale_cam = np.stack(crop_cams)
                if full_mask is None:
                    full_mask = interpolate_cam_on_voxel(
                        image_cube, grayscale_cam, input_tensor_info
                    )
                else:
                    full_mask += interpolate_cam_on_voxel(
                        image_cube, grayscale_cam, input_tensor_info
                    )

            del cam
            del grayscale_cam
            del target_layers
            self._release(model)

        if full_mask is not None:
            return full_mask / len(self.cls_models)

        return None

    def run(self, scan_dir: str | Path) -> dict:
        """Return the eight classifier probabilities and the full-volume CAM."""
        data = {"scan_dir": str(Path(scan_dir).expanduser().resolve())}
        mask_cube = self._predict_seg_mask(data)
        prediction = self._predict_cls(data, mask_cube)
        cam = self.explain_gradcam(data, mask_cube=mask_cube)
        cam = align_cam_to_scan(cam, scan_dir) if cam is not None else None
        return {
            "prediction": prediction,
            "cam": cam,
            "ensemble_mode": "reduced" if self.reduced_ensemble else "full",
            "low_memory": self.low_memory,
            "cam_axis_order": "ZYX",
            "cam_shape": list(cam.shape) if cam is not None else None,
        }
