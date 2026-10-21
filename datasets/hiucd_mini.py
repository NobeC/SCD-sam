"""Hi-UCD mini training pairs as aligned 512px crops; official val/test are never loaded."""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
QUADRANTS = ((0, 0), (0, 512), (512, 0), (512, 512))


def aligned_transform(arrays, geometry):
    k, horizontal, vertical = geometry
    result = []
    for array in arrays:
        array = np.rot90(array, k)
        if horizontal:
            array = np.flip(array, axis=1)
        if vertical:
            array = np.flip(array, axis=0)
        result.append(np.ascontiguousarray(array))
    return result


class HiUCDMini(Dataset):
    def __init__(self, root, split_json, split, augment=False):
        if split not in ("train", "val"):
            raise ValueError("Only internal train/val subsets are allowed")
        if augment and split != "train":
            raise ValueError("Augmentation is restricted to train")
        self.root = Path(root)
        self.manifest = json.loads(Path(split_json).read_text())
        self.split = split
        self.augment = augment
        self.items = self.manifest["patches"][split]
        if not self.items:
            raise ValueError("Empty Hi-UCD subset")
        self.source_rows = {row["name"]: row for row in self.manifest["source_pairs"]}

    def __len__(self):
        return len(self.items)

    def sample_name(self, index):
        return self.items[index]["sample_id"]

    def read_patch(self, item):
        row = self.source_rows[item["source_name"]]
        image_a = cv2.imread(str(self.root / row["A"]), cv2.IMREAD_UNCHANGED)
        image_b = cv2.imread(str(self.root / row["B"]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(self.root / row["mask_merge"]), cv2.IMREAD_UNCHANGED)
        if image_a is None or image_b is None or mask is None:
            raise OSError("Missing image or merged mask for " + item["source_name"])
        arrays = (image_a, image_b, mask)
        if any(x.dtype != np.uint8 or x.shape != (1024, 1024, 3) for x in arrays):
            raise ValueError("Expected three 1024x1024 uint8 images: " + item["source_name"])
        y, x = QUADRANTS[int(item["quadrant"])]
        a = cv2.cvtColor(image_a[y:y + 512, x:x + 512], cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(image_b[y:y + 512, x:x + 512], cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(mask[y:y + 512, x:x + 512], cv2.COLOR_BGR2RGB)
        label_a, label_b, transition = (target[:, :, i] for i in range(3))
        if int(label_a.max()) > 9 or int(label_b.max()) > 9 or int(transition.max()) > 2:
            raise ValueError("Unexpected Hi-UCD label values in " + item["source_name"])
        return [a, b, label_a, label_b, (transition == 2).astype(np.uint8), (transition != 0).astype(np.uint8)]

    def __getitem__(self, index):
        item = self.items[index]
        arrays = self.read_patch(item)
        geometry = (random.randrange(4), random.random() < .5, random.random() < .5) if self.augment else (0, False, False)
        a, b, label_a, label_b, change, change_valid = aligned_transform(arrays, geometry)
        a = ((a.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
        b = ((b.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
        return {
            "name": item["sample_id"],
            "geometry": torch.tensor(geometry, dtype=torch.long),
            "A": torch.from_numpy(np.ascontiguousarray(a)),
            "B": torch.from_numpy(np.ascontiguousarray(b)),
            "label_A": torch.from_numpy(label_a.astype(np.int64)),
            "label_B": torch.from_numpy(label_b.astype(np.int64)),
            "change": torch.from_numpy(change.astype(np.int64)),
            "change_valid": torch.from_numpy(change_valid.astype(np.bool_)),
        }
