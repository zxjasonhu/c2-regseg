from typing import List, Dict

import numpy as np

from scipy.ndimage import zoom

import cv2


def cam_to_intermediate_cam(
    cam: np.ndarray,
    interval: int,
    length: int = 40,
    h: int = 512,
    w: int = 512,
) -> np.ndarray:
    cam_length = cam.shape[0]

    grayscale_cam_resized = zoom(
        cam, (length / cam_length, h / cam.shape[1], w / cam.shape[2]), order=1
    )

    return grayscale_cam_resized[:interval, :, :]


def naive_overlap_cams(
    cam: np.ndarray, target_length, h: int = 512, w: int = 512
) -> np.ndarray:
    assert cam.shape[0] == 2, f"cam.shape[0]={cam.shape[0]} != 2"

    _cam = np.zeros((target_length, h, w))
    # fill the _cam by cam[0] and cam[1] and max() for the overlapping part
    cam_length = cam.shape[1]

    # Place the first cam at the beginning
    _cam[:cam_length] = cam[0]

    # Place the second cam at the end, taking the maximum for any overlapping regions
    second_start = target_length - cam_length
    _cam[second_start:] = np.maximum(_cam[second_start:], cam[1])

    return _cam


def interpolate_cam_on_voxel(
    voxel: np.ndarray, cam: np.ndarray, conversion_info: List[Dict]
) -> np.ndarray:
    assert (
        len(conversion_info) == cam.shape[0]
    ), f"len(conversion_info)={len(conversion_info)} != cam.shape[0]={cam.shape[0]}"

    final_mask = np.zeros(voxel.shape)

    # parse conversion info
    i = 0
    while i < len(conversion_info):
        info = conversion_info[i]
        _type = info["type"]
        _total_slice = info["total_slice"]
        _current_index = info["current_index"]
        _z1 = info["z1"]
        _z2 = info["z2"]
        _x1 = info["x1"]
        _x2 = info["x2"]
        _y1 = info["y1"]
        _y2 = info["y2"]
        real_length = _z2 - _z1
        real_height = _y2 - _y1
        real_width = _x2 - _x1

        if _type == "overlap":
            former = cam_to_intermediate_cam(cam[i], 40, 40, real_height, real_width)
            latter = cam_to_intermediate_cam(
                cam[i + 1], 40, 40, real_height, real_width
            )
            current_cam = naive_overlap_cams(
                np.stack([former, latter]), _total_slice, real_height, real_width
            )
            current_cam = cam_to_intermediate_cam(
                current_cam, real_length, real_length, real_height, real_width
            )
            i += 1
        elif _type == "padding":
            current_cam = cam_to_intermediate_cam(
                cam[i],
                int(40 * real_length / _total_slice),
                real_length,
                real_height,
                real_width,
            )
        elif _type == "normal":
            current_cam = cam_to_intermediate_cam(
                cam[i], real_length, real_length, real_height, real_width
            )
        else:
            i += 1
            continue

        # DEBUG use
        # if np.max(current_cam) < 0.5:
        #     current_cam = np.ones_like(current_cam)
        # fill the final mask with max() for the overlapping part
        final_mask[_z1:_z2, _y1:_y2, _x1:_x2] = np.maximum(
            final_mask[_z1:_z2, _y1:_y2, _x1:_x2], current_cam
        )

        i += 1

    # Normalize the final mask
    return final_mask


def convert_to_rgb(image):
    return np.stack((image,) * 3, axis=-1)
