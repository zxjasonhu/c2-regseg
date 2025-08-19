import os
import threading
from glob import glob

import cv2
import nibabel as nib
import numpy as np
import pydicom
import torch

image_size_seg = (128, 128, 128)
msk_size = image_size_seg[0]
image_size_cls = 512
n_slice_per_c = 15
n_ch = 5
in_chans = 6


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(
        data, (image_size_seg[0], image_size_seg[1]), interpolation=cv2.INTER_AREA
    )
    return data


def load_dicom_line_par(path):
    t_paths = sorted(
        glob(os.path.join(path, "*")),
        key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[-1]),
    )

    n_scans = len(t_paths)
    indices = (
        np.quantile(list(range(n_scans)), np.linspace(0.0, 1.0, image_size_seg[2]))
        .round()
        .astype(int)
    )
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images


def load_nifti(path):
    nifti = nib.load(path)
    data = nifti.get_fdata()

    # Flip along the z-axis to match DICOM orientation
    # Since NIFTI is bottom-to-top and DICOM is top-to-bottom
    data = np.flip(data, axis=2)

    # Resize to target dimensions (128, 128, 128)
    resized_data = np.zeros(image_size_seg, dtype=np.float32)

    # Resample along all dimensions if needed
    if data.shape != image_size_seg:
        # Resample along z-axis
        z_indices = np.linspace(0, data.shape[2] - 1, image_size_seg[2]).astype(int)

        for i, z_idx in enumerate(z_indices):
            slice_img = data[:, :, z_idx]
            # Handle multi-channel slices if they exist
            if slice_img.ndim > 2:
                slice_img = slice_img[:, :, 0]  # Take first channel

            # Resize each 2D slice to target dimensions
            resized_slice = cv2.resize(
                slice_img,
                (image_size_seg[0], image_size_seg[1]),
                interpolation=cv2.INTER_AREA,
            )
            resized_data[:, :, i] = resized_slice
    else:
        resized_data = data

    # Normalize to 0-255 range, matching the DICOM processing
    data = resized_data - np.min(resized_data)
    data = data / (np.max(data) + 1e-4)
    data = (data * 255).astype(np.uint8)

    return data


def load_bone(msk, cid, t_paths, cropped_images, crop_info):
    nii_image = None
    if isinstance(t_paths, str):
        nii_image = load_nifti(t_paths)
        _length = nii_image.shape[2]
    else:
        _length = len(t_paths)

    bone = []
    try:
        msk_b = msk[cid] > 0.2
        msk_c = msk[cid] > 0.05

        x = np.where(msk_b.sum(1).sum(1) > 0)[0]
        y = np.where(msk_b.sum(0).sum(1) > 0)[0]
        z = np.where(msk_b.sum(0).sum(0) > 0)[0]

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            x = np.where(msk_c.sum(1).sum(1) > 0)[0]
            y = np.where(msk_c.sum(0).sum(1) > 0)[0]
            z = np.where(msk_c.sum(0).sum(0) > 0)[0]

        x1, x2 = max(0, x[0] - 1), min(msk.shape[1], x[-1] + 1)
        y1, y2 = max(0, y[0] - 1), min(msk.shape[2], y[-1] + 1)
        z1, z2 = max(0, z[0] - 1), min(msk.shape[3], z[-1] + 1)
        zz1, zz2 = int(z1 / msk_size * _length), int(z2 / msk_size * _length)

        inds = np.linspace(zz1, zz2 - 1, n_slice_per_c).astype(int)
        inds_ = np.linspace(z1, z2 - 1, n_slice_per_c).astype(int)
        for sid, (ind, ind_) in enumerate(zip(inds, inds_)):
            msk_this = msk[cid, :, :, ind_]

            images = []
            for i in range(-n_ch // 2 + 1, n_ch // 2 + 1):
                try:
                    if nii_image is not None:
                        images.append(nii_image[:, :, ind + i])
                    else:
                        dicom = pydicom.read_file(t_paths[ind + i])
                        images.append(dicom.pixel_array)
                except:
                    images.append(np.zeros((512, 512)))

            data = np.stack(images, -1)
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-4)
            data = (data * 255).astype(np.uint8)
            msk_this = msk_this[x1:x2, y1:y2]
            xx1 = int(x1 / msk_size * data.shape[0])
            xx2 = int(x2 / msk_size * data.shape[0])
            yy1 = int(y1 / msk_size * data.shape[1])
            yy2 = int(y2 / msk_size * data.shape[1])
            data = data[xx1:xx2, yy1:yy2]
            data = np.stack(
                [
                    cv2.resize(
                        data[:, :, i],
                        (image_size_cls, image_size_cls),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for i in range(n_ch)
                ],
                -1,
            )
            msk_this = (msk_this * 255).astype(np.uint8)
            msk_this = cv2.resize(
                msk_this,
                (image_size_cls, image_size_cls),
                interpolation=cv2.INTER_LINEAR,
            )

            data = np.concatenate([data, msk_this[:, :, np.newaxis]], -1)

            bone.append(torch.tensor(data))
        crop_info[cid] = [xx1, xx2, yy1, yy2, zz1, zz2, _length]

    except Exception as e:
        print(f"Error in loading bone for cid {cid}: {e}")
        import traceback
        traceback.print_exc()
        for sid in range(n_slice_per_c):
            bone.append(torch.ones((image_size_cls, image_size_cls, n_ch + 1)).int())

    cropped_images[cid] = torch.stack(bone, 0)


def load_cropped_images(msk, image_folder):
    # list files in folder
    # Assuming the folder contains DICOM files
    files = os.listdir(image_folder)
    for f in files:
        if f.endswith(".nii.gz") or f.endswith(".nii"):
            image_folder = os.path.join(image_folder, f)
            break

    if image_folder.endswith(".nii.gz") or image_folder.endswith(".nii"):
        t_paths = image_folder
    else:
        t_paths = sorted(
            glob(os.path.join(image_folder, "*")),
            key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[-1]),
        )

    threads = [None] * 7
    cropped_images = [None] * 7
    crop_info = [None] * 7

    for cid in range(7):
        threads[cid] = threading.Thread(
            target=load_bone, args=(msk, cid, t_paths, cropped_images, crop_info)
        )
        threads[cid].start()
    for cid in range(7):
        threads[cid].join()

    return torch.cat(cropped_images, 0), crop_info


def get_trans(img, I):
    I = I % 8
    if I >= 4:
        img = img.transpose(3, 4)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(3)
    elif I % 4 == 2:
        return img.flip(4)
    elif I % 4 == 3:
        return img.flip(3).flip(4)


def get_cam_trans(cam, I):
    I = I % 8
    if I >= 4:
        cam = cam.transpose(1, 2)
    if I % 4 == 0:
        return cam
    elif I % 4 == 1:
        return cam.flip(1)
    elif I % 4 == 2:
        return cam.flip(2)
    elif I % 4 == 3:
        return cam.flip(1).flip(2)
