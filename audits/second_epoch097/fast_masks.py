"""Exact packed-bit implementation of the audit mask deduplication rule."""
import numpy as np
import torch
from utils.sam_mask_cache import _validate_masks, _normalize_image_shape


def prepare_shared_masks(masks_a, masks_b, image_shape, config):
    shape = _normalize_image_shape(image_shape)
    combined = np.concatenate([_validate_masks("masks_A", masks_a, shape),
                               _validate_masks("masks_B", masks_b, shape)])
    if not len(combined):
        return torch.empty((0, *shape), dtype=torch.bool)
    areas = combined.reshape(len(combined), -1).sum(1)
    minimum = max(config.min_mask_pixels, int(np.ceil(config.min_area_ratio * np.prod(shape))))
    maximum = int(np.floor(config.max_area_ratio * np.prod(shape)))
    order = sorted([i for i, a in enumerate(areas) if 0 < a and minimum <= a <= maximum],
                   key=lambda i: (int(areas[i]), i))
    packed = np.packbits(combined.reshape(len(combined), -1), axis=1)
    bits = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    boxes = {}
    for i in order:
        y, x = np.nonzero(combined[i])
        boxes[i] = np.array([x.min(), y.min(), x.max() + 1, y.max() + 1])
    kept = []
    for i in order:
        duplicate = False
        for j in kept:
            a, b = float(areas[i]), float(areas[j])
            lower = np.maximum(boxes[i][:2], boxes[j][:2])
            upper = np.minimum(boxes[i][2:], boxes[j][2:])
            bound = min(a, b, float(np.maximum(upper - lower, 0).prod()))
            required = min(config.dedup_iou_threshold * (a + b) / (1 + config.dedup_iou_threshold),
                           config.dedup_containment_threshold * min(a, b))
            if bound < required:
                continue
            intersection = int(bits[np.bitwise_and(packed[i], packed[j])].sum())
            if intersection and (intersection / max(a + b - intersection, 1) >= config.dedup_iou_threshold
                                 or intersection / max(min(a, b), 1) >= config.dedup_containment_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return torch.from_numpy(combined[kept].copy())
