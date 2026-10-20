import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.RS_ST_graph import SamGraphDataset, collate_sam_graph_batch
from utils.sam_mask_cache import SamMaskCacheConfig


class _FakeBaseDataset:
    def __init__(self):
        values = torch.arange(16).reshape(4, 4)
        self.image = values.float().unsqueeze(0).repeat(3, 1, 1)
        self.label = values.long() % 3

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.image, self.image + 20, self.label, self.label

    def get_mask_name(self, index):
        return "tile_001.png"


class SamGraphDatasetTest(unittest.TestCase):
    def _write_cache(self, root):
        path = Path(root) / "Landsat" / "train" / "tile_001.npz"
        path.parent.mkdir(parents=True)
        mask = np.zeros((1, 4, 4), dtype=np.uint8)
        mask[0, 0:2, 0:2] = 1
        np.savez_compressed(
            path,
            masks_A=mask,
            masks_B=np.empty((0, 4, 4), dtype=np.uint8),
            image_shape=np.asarray([4, 4]),
            format_version=np.asarray(1),
        )

    def test_wrapper_returns_five_items_with_stable_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_cache(tmp)
            dataset = SamGraphDataset(
                _FakeBaseDataset(), "Landsat", "train", tmp,
                SamMaskCacheConfig(min_area_ratio=0.0, max_area_ratio=1.0),
            )

            sample = dataset[0]

            self.assertEqual(len(sample), 5)
            self.assertEqual(sample[4]["sample_id"], "tile_001")
            self.assertEqual(tuple(sample[4]["shared_masks"].shape), (1, 4, 4))

    def test_graph_collate_keeps_variable_metadata_as_list(self):
        graph_a = {"sample_id": "a", "shared_masks": torch.ones((1, 4, 4), dtype=torch.bool)}
        graph_b = {"sample_id": "b", "shared_masks": torch.ones((2, 4, 4), dtype=torch.bool)}
        dense = (torch.zeros(3, 4, 4), torch.zeros(3, 4, 4),
                 torch.zeros(4, 4, dtype=torch.long), torch.zeros(4, 4, dtype=torch.long))

        batch = collate_sam_graph_batch([dense + (graph_a,), dense + (graph_b,)])

        self.assertEqual(tuple(batch[0].shape), (2, 3, 4, 4))
        self.assertEqual([item["sample_id"] for item in batch[4]], ["a", "b"])
        self.assertEqual([item["shared_masks"].shape[0] for item in batch[4]], [1, 2])

    def test_base_dataset_behavior_is_not_modified(self):
        sample = _FakeBaseDataset()[0]
        self.assertEqual(len(sample), 4)


if __name__ == "__main__":
    unittest.main()
