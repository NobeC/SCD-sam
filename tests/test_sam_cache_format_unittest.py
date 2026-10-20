import tempfile
import unittest
from pathlib import Path

import numpy as np


class SamCacheFormatTest(unittest.TestCase):
    def test_pack_round_trip_preserves_non_byte_aligned_width(self):
        from utils.sam_cache_format import pack_masks, unpack_masks
        masks = np.zeros((2, 3, 10), dtype=np.bool_)
        masks[0, :, 1:4] = True
        masks[1, 1:, 8:] = True
        self.assertTrue(np.array_equal(masks, unpack_masks(pack_masks(masks), 10)))

    def test_dedup_prefers_quality_and_keeps_different_scale_containment(self):
        from utils.sam_cache_format import deduplicate_records
        base = np.zeros((10, 10), dtype=np.bool_); base[2:8, 2:8] = True
        duplicate = base.copy()
        large = np.zeros_like(base); large[1:9, 1:9] = True
        records = [
            {"segmentation": base, "predicted_iou": .8, "stability_score": .8},
            {"segmentation": duplicate, "predicted_iou": .9, "stability_score": .9},
            {"segmentation": large, "predicted_iou": .7, "stability_score": .9},
        ]
        keep = deduplicate_records(records, duplicate_iou_thresh=.9,
                                   containment_thresh=.95, containment_area_ratio=.8)
        self.assertEqual(keep, [1, 2])

    def test_v2_round_trip_preserves_raw_and_shared_masks(self):
        from utils.sam_cache_format import load_sam_cache_v2, save_sam_cache_v2
        a = np.zeros((4, 5), dtype=np.bool_); a[1:3, 1:3] = True
        b = np.zeros((4, 5), dtype=np.bool_); b[2:, 2:] = True
        record_a = {"segmentation": a, "area": 4, "bbox": [1, 1, 2, 2],
                    "predicted_iou": .9, "stability_score": .95}
        record_b = {"segmentation": b, "area": 4, "bbox": [2, 2, 2, 2],
                    "predicted_iou": .85, "stability_score": .9}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            save_sam_cache_v2(path, "sample", (4, 5), [record_a], [record_b], {"seed": 1})
            loaded = load_sam_cache_v2(path)
        self.assertEqual(loaded["sample_id"], "sample")
        self.assertTrue(np.array_equal(loaded["masks_A"][0], a))
        self.assertEqual(loaded["shared_masks"].shape, (2, 4, 5))


if __name__ == "__main__":
    unittest.main()
