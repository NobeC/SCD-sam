import random

import numpy as np


VALID_FLIPS = ("none", "vertical", "horizontal", "both")


def rand_rot90_flip_SCD_masks(
        img1, img2, label1, label2, masks, rotate=None, flip=None):
    """Apply one shared 90-degree rotation and flip to all spatial inputs."""
    masks = np.asarray(masks, dtype=np.bool_)
    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(label1.shape):
        raise ValueError("masks must have shape [N, H, W] matching the labels")
    if (label1.shape != label2.shape or img1.shape[:2] != label1.shape
            or img2.shape[:2] != label1.shape):
        raise ValueError("images, labels, and masks must share one spatial shape")

    if rotate is None:
        rotate = random.random() >= 0.5
    if not isinstance(rotate, (bool, np.bool_)):
        raise ValueError("rotate must be a boolean or None")
    if flip is None:
        flip = VALID_FLIPS[min(int(random.random() * 4), 3)]
    if flip not in VALID_FLIPS:
        raise ValueError("flip must be one of {}".format(VALID_FLIPS))

    if rotate:
        img1 = np.rot90(img1).copy()
        img2 = np.rot90(img2).copy()
        label1 = np.rot90(label1).copy()
        label2 = np.rot90(label2).copy()
        masks = np.rot90(masks, axes=(-2, -1)).copy()

    if flip == "vertical":
        image_axis, mask_axis = 0, -2
    elif flip == "horizontal":
        image_axis, mask_axis = 1, -1
    elif flip == "both":
        image_axis, mask_axis = (0, 1), (-2, -1)
    else:
        return img1, img2, label1, label2, masks

    return (
        np.flip(img1, axis=image_axis).copy(),
        np.flip(img2, axis=image_axis).copy(),
        np.flip(label1, axis=image_axis).copy(),
        np.flip(label2, axis=image_axis).copy(),
        np.flip(masks, axis=mask_axis).copy(),
    )
