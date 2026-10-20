from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import torch

from utils.sam_cache_format import CACHE_VERSION, unpack_masks


PathLike = Union[str, Path]


class SamMaskCacheError(ValueError):
    """Raised when a SAM mask cache entry violates the cache contract."""


@dataclass(frozen=True)
class SamMaskCacheConfig:
    format_version: int = 1
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.05
    min_mask_pixels: int = 1
    dedup_iou_threshold: float = 0.95
    dedup_containment_threshold: float = 0.98

    def __post_init__(self):
        values = (
            self.min_area_ratio,
            self.max_area_ratio,
            self.dedup_iou_threshold,
            self.dedup_containment_threshold,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("SAM mask cache thresholds must be finite")
        if not 0.0 <= self.min_area_ratio <= self.max_area_ratio <= 1.0:
            raise ValueError("SAM mask area ratios must satisfy 0 <= min <= max <= 1")
        if self.min_mask_pixels < 0:
            raise ValueError("min_mask_pixels must be non-negative")
        if not 0.0 <= self.dedup_iou_threshold <= 1.0:
            raise ValueError("dedup_iou_threshold must be in [0, 1]")
        if not 0.0 <= self.dedup_containment_threshold <= 1.0:
            raise ValueError("dedup_containment_threshold must be in [0, 1]")


def cache_path_for_sample(
    cache_root: PathLike, dataset_name: str, split: str, sample_id: str
) -> Path:
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError("sample_id must be a non-empty filename component")
    return Path(cache_root) / str(dataset_name) / str(split) / (sample_id + ".npz")


def _normalize_image_shape(image_shape: Sequence[int]) -> Tuple[int, int]:
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain H and W")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")
    return height, width


def _load_shared_masks_v2(path, sample_id, image_shape, config):
    """Load only the cached shared masks needed by the graph dataset.

    Version 2 caches already contain the deduplicated shared-mask indices.
    Avoid unpacking all raw A/B masks and running the legacy deduplication pass
    on every sample access.
    """
    height, width = _normalize_image_shape(image_shape)
    with np.load(str(path), allow_pickle=False) as payload:
        cached_shape = tuple(
            int(value) for value in np.asarray(payload["image_shape"]).reshape(-1)
        )
        if cached_shape != (height, width):
            raise ValueError(
                "image_shape {}, expected {}".format(cached_shape, (height, width))
            )
        version = int(np.asarray(payload["format_version"]).reshape(-1)[0])
        if version != CACHE_VERSION:
            raise ValueError("not a SAM cache v2 file")
        cached_sample_id = str(np.asarray(payload["sample_id"]).reshape(-1)[0])
        if cached_sample_id != sample_id:
            raise ValueError(
                "contains sample_id {!r}".format(cached_sample_id)
            )

        source = np.asarray(payload["shared_source"], dtype=np.uint8)
        indices = np.asarray(payload["shared_index"], dtype=np.int64)
        if source.ndim != 1 or indices.shape != source.shape:
            raise ValueError("shared mask index arrays have incompatible shapes")
        if source.size == 0:
            return torch.empty((0, height, width), dtype=torch.bool)
        if not np.logical_or(source == 0, source == 1).all():
            raise ValueError("shared_source contains an invalid source")

        areas_a = np.asarray(payload["area_A"], dtype=np.int64)
        areas_b = np.asarray(payload["area_B"], dtype=np.int64)
        if (source == 0).any() and int(indices[source == 0].max()) >= len(areas_a):
            raise ValueError("shared_index exceeds area_A")
        if (source == 1).any() and int(indices[source == 1].max()) >= len(areas_b):
            raise ValueError("shared_index exceeds area_B")

        areas = np.empty_like(indices, dtype=np.int64)
        source_a = source == 0
        areas[source_a] = areas_a[indices[source_a]]
        areas[~source_a] = areas_b[indices[~source_a]]
        image_area = height * width
        minimum = max(
            config.min_mask_pixels,
            int(np.ceil(config.min_area_ratio * image_area)),
        )
        maximum = int(np.floor(config.max_area_ratio * image_area))
        valid = (areas > 0) & (areas >= minimum) & (areas <= maximum)
        if not valid.any():
            return torch.empty((0, height, width), dtype=torch.bool)

        selected_source = source[valid]
        selected_indices = indices[valid]
        packed_width = (width + 7) // 8
        selected = np.empty(
            (selected_source.size, height, packed_width), dtype=np.uint8
        )
        if (selected_source == 0).any():
            packed_a = np.asarray(payload["packed_A"])
            if tuple(packed_a.shape[1:]) != (height, packed_width):
                raise ValueError("packed_A has an invalid shape")
            selected[selected_source == 0] = packed_a[
                selected_indices[selected_source == 0]
            ]
        if (selected_source == 1).any():
            packed_b = np.asarray(payload["packed_B"])
            if tuple(packed_b.shape[1:]) != (height, packed_width):
                raise ValueError("packed_B has an invalid shape")
            selected[selected_source == 1] = packed_b[
                selected_indices[selected_source == 1]
            ]
        masks = unpack_masks(selected, width)
    return prepare_shared_masks(
        masks,
        np.empty((0, height, width), dtype=np.bool_),
        (height, width),
        config,
    )
def _validate_masks(name: str, masks, image_shape: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(masks)
    expected_shape = tuple(image_shape)
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_shape:
        raise ValueError(
            "{} must have shape [N, {}, {}], got {}".format(
                name, expected_shape[0], expected_shape[1], tuple(array.shape)
            )
        )
    if array.dtype == np.bool_:
        return array.copy()
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("{} must contain binary integer or boolean values".format(name))
    if array.size and not np.logical_or(array == 0, array == 1).all():
        raise ValueError("{} must contain binary values".format(name))
    return array.astype(np.bool_, copy=False)


def _is_duplicate(mask: np.ndarray, kept: np.ndarray, config: SamMaskCacheConfig) -> bool:
    intersection = int(np.logical_and(mask, kept).sum())
    if intersection == 0:
        return False
    mask_area = int(mask.sum())
    kept_area = int(kept.sum())
    union = mask_area + kept_area - intersection
    iou = intersection / float(max(union, 1))
    containment = intersection / float(max(min(mask_area, kept_area), 1))
    return (
        iou >= config.dedup_iou_threshold
        or containment >= config.dedup_containment_threshold
    )


def prepare_shared_masks(
    masks_a,
    masks_b,
    image_shape: Sequence[int],
    config: SamMaskCacheConfig = SamMaskCacheConfig(),
) -> torch.Tensor:
    """Validate, union, filter, and deterministically deduplicate A/B SAM masks."""
    height, width = _normalize_image_shape(image_shape)
    arrays = (
        _validate_masks("masks_A", masks_a, (height, width)),
        _validate_masks("masks_B", masks_b, (height, width)),
    )
    combined = np.concatenate(arrays, axis=0)
    if combined.shape[0] == 0:
        return torch.empty((0, height, width), dtype=torch.bool)

    areas = combined.reshape(combined.shape[0], -1).sum(axis=1)
    image_area = height * width
    minimum = max(config.min_mask_pixels, int(np.ceil(config.min_area_ratio * image_area)))
    maximum = int(np.floor(config.max_area_ratio * image_area))
    valid_indices = [
        index for index, area in enumerate(areas.tolist())
        if area > 0 and minimum <= area <= maximum
    ]
    valid_indices.sort(key=lambda index: (int(areas[index]), index))

    kept = []
    for index in valid_indices:
        candidate = combined[index]
        if any(_is_duplicate(candidate, existing, config) for existing in kept):
            continue
        kept.append(candidate.copy())

    if not kept:
        return torch.empty((0, height, width), dtype=torch.bool)
    return torch.from_numpy(np.stack(kept, axis=0)).to(dtype=torch.bool)


def load_shared_sam_masks(
    cache_root: PathLike,
    dataset_name: str,
    split: str,
    sample_id: str,
    image_shape: Sequence[int],
    config: SamMaskCacheConfig = SamMaskCacheConfig(),
) -> torch.Tensor:
    """Load one safe NPZ entry and return its prepared shared masks."""
    path = cache_path_for_sample(cache_root, dataset_name, split, sample_id)
    context = "SAM mask cache for sample {!r} at {}".format(sample_id, path)
    if not path.is_file():
        raise SamMaskCacheError("{} does not exist".format(context))
    try:
        expected_shape = _normalize_image_shape(image_shape)
        with np.load(str(path), allow_pickle=False) as payload:
            required = {"image_shape", "format_version"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise SamMaskCacheError("{} is missing keys {}".format(context, missing))
            cached_shape = tuple(int(value) for value in np.asarray(payload["image_shape"]).reshape(-1))
            if cached_shape != expected_shape:
                raise SamMaskCacheError(
                    "{} has image_shape {}, expected {}".format(context, cached_shape, expected_shape)
                )
            version_values = np.asarray(payload["format_version"]).reshape(-1)
            if version_values.size != 1:
                raise SamMaskCacheError(
                    "{} has unsupported format_version".format(context)
                )
            version = int(version_values[0])
            if version == 1:
                legacy_required = {"masks_A", "masks_B"}
                missing = sorted(legacy_required.difference(payload.files))
                if missing:
                    raise SamMaskCacheError("{} is missing keys {}".format(context, missing))
                masks_a = np.asarray(payload["masks_A"])
                masks_b = np.asarray(payload["masks_B"])
            elif version != CACHE_VERSION:
                raise SamMaskCacheError("{} has unsupported format_version {}".format(context, version))
        if version == CACHE_VERSION:
            return _load_shared_masks_v2(path, sample_id, expected_shape, config)
    except SamMaskCacheError:
        raise
    except Exception as error:
        raise SamMaskCacheError("{} could not be read: {}".format(context, error)) from error

    try:
        return prepare_shared_masks(masks_a, masks_b, expected_shape, config)
    except ValueError as error:
        raise SamMaskCacheError("{} is invalid: {}".format(context, error)) from error
