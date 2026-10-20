import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.sam_mask_cache import (
    SamMaskCacheConfig,
    load_shared_sam_masks,
    prepare_shared_masks,
)


class SamMaskCacheTest(unittest.TestCase):
    def test_union_deduplicates_and_filters_mid_masks(self):
        masks_a = np.zeros((2, 10, 10), dtype=np.uint8)
        masks_b = np.zeros((2, 10, 10), dtype=np.uint8)
        masks_a[0, 0:2, 0:2] = 1
        masks_a[1, 0, 0] = 1
        masks_b[0, 0:2, 0:2] = 1
        masks_b[1, 5:8, 5:8] = 1
        config = SamMaskCacheConfig(
            min_area_ratio=0.02,
            max_area_ratio=0.10,
            min_mask_pixels=2,
            dedup_iou_threshold=0.95,
            dedup_containment_threshold=0.95,
        )

        shared = prepare_shared_masks(masks_a, masks_b, (10, 10), config)

        self.assertEqual(shared.dtype, torch.bool)
        self.assertEqual(tuple(shared.shape), (2, 10, 10))
        self.assertEqual(sorted(shared.flatten(1).sum(1).tolist()), [4, 9])

    def test_containment_keeps_smaller_mask(self):
        masks_a = np.zeros((1, 10, 10), dtype=np.uint8)
        masks_b = np.zeros((1, 10, 10), dtype=np.uint8)
        masks_a[0, 1:5, 1:5] = 1
        masks_b[0, 2:4, 2:4] = 1
        config = SamMaskCacheConfig(
            min_area_ratio=0.01,
            max_area_ratio=1.0,
            min_mask_pixels=1,
            dedup_iou_threshold=0.99,
            dedup_containment_threshold=0.99,
        )

        shared = prepare_shared_masks(masks_a, masks_b, (10, 10), config)

        self.assertEqual(tuple(shared.shape), (1, 10, 10))
        self.assertEqual(int(shared.sum()), 4)

    def test_loads_safe_npz_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            cache_path = cache_root / "Landsat" / "train" / "sample.npz"
            cache_path.parent.mkdir(parents=True)
            masks = np.zeros((1, 10, 10), dtype=np.uint8)
            masks[0, 1:3, 1:3] = 1
            np.savez_compressed(
                cache_path,
                masks_A=masks,
                masks_B=np.empty((0, 10, 10), dtype=np.uint8),
                image_shape=np.asarray([10, 10], dtype=np.int64),
                format_version=np.asarray(1, dtype=np.int64),
            )

            shared = load_shared_sam_masks(
                cache_root, "Landsat", "train", "sample", (10, 10),
                SamMaskCacheConfig(min_area_ratio=0.01, max_area_ratio=1.0),
            )

            self.assertEqual(tuple(shared.shape), (1, 10, 10))

    def test_loads_compressed_v2_cache_and_filters_shared_masks(self):
        from utils.sam_cache_format import save_sam_cache_v2

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            cache_path = cache_root / "Landsat" / "train" / "sample.npz"
            small = np.zeros((10, 10), dtype=np.bool_)
            small[1:3, 1:3] = True
            large = np.zeros((10, 10), dtype=np.bool_)
            large[1:9, 1:9] = True
            records = [
                {"segmentation": small, "predicted_iou": .9, "stability_score": .9},
                {"segmentation": large, "predicted_iou": .9, "stability_score": .9},
            ]
            save_sam_cache_v2(cache_path, "sample", (10, 10), records, [], {})

            shared = load_shared_sam_masks(
                cache_root, "Landsat", "train", "sample", (10, 10),
                SamMaskCacheConfig(min_area_ratio=0.02, max_area_ratio=0.10),
            )

            self.assertEqual(tuple(shared.shape), (1, 10, 10))
            self.assertEqual(int(shared.sum()), 4)

    def test_invalid_cache_reports_sample_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "sample.*sample.npz"):
                load_shared_sam_masks(
                    tmp, "Landsat", "train", "sample", (10, 10),
                    SamMaskCacheConfig(),
                )

    def test_non_binary_masks_are_rejected(self):
        masks = np.zeros((1, 10, 10), dtype=np.uint8)
        masks[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            prepare_shared_masks(masks, np.empty((0, 10, 10)), (10, 10), SamMaskCacheConfig())

    def test_empty_masks_keep_spatial_shape(self):
        shared = prepare_shared_masks(
            np.empty((0, 7, 9), dtype=np.uint8),
            np.empty((0, 7, 9), dtype=np.uint8),
            (7, 9),
            SamMaskCacheConfig(min_area_ratio=0.0, max_area_ratio=1.0),
        )
        self.assertEqual(tuple(shared.shape), (0, 7, 9))


if __name__ == "__main__":
    unittest.main()
