"""Hi-UCDmini adapter: same public interface as datasets/RS_ST.py.

train.py imports this module as `RS` when --dataset hiucd, so every symbol
used by train.py (num_classes, Index2Color, tensor2int, Data) must exist here.
"""
import numpy as np
import torch

from configs.datasets import get_dataset_config
from datasets.hiucd_mini import HiUCDMini

_CONFIG = get_dataset_config("hiucd")
num_classes = _CONFIG.num_classes          # 10
ST_COLORMAP = [list(c) for c in _CONFIG.colormap]
ST_CLASSES = list(_CONFIG.classes)

# Configured at runtime by train.py before any Data() is instantiated.
_root = None
_split_json = None


def configure(root, split_json):
    """Call this once, before creating Data(), to set image root + split manifest."""
    global _root, _split_json
    _root = str(root)
    _split_json = str(split_json)


# ── colour helpers ────────────────────────────────────────────────────────────

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype="uint8")
    x = np.asarray(pred, dtype="int32").clip(0, num_classes - 1)
    return colormap[x, :]


def Color2Index(color_label):
    colormap2label = np.zeros(256 ** 3, dtype=np.int64)
    for i, cm in enumerate(ST_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
    data = color_label.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return colormap2label[idx].clip(0, num_classes - 1)


# Hi-UCDmini ImageNet-style normalisation constants (from hiucd_mini.py)
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)


def tensor2int(im, time="A"):
    """Invert normalisation for visualisation (ignores time, uses shared stats)."""
    if torch.is_tensor(im):
        im = im.detach().cpu().numpy()
    im = np.asarray(im, dtype=np.float32)
    if im.ndim == 3 and im.shape[0] == 3:   # CHW → HWC
        im = im.transpose(1, 2, 0)
    return (im * _STD + _MEAN).clip(0, 255).astype(np.uint8)


# ── dataset class ─────────────────────────────────────────────────────────────

class Data(torch.utils.data.Dataset):
    """Wraps HiUCDMini and returns the same (img_A, img_B, label_A, label_B)
    tuple that RS_ST.Data returns, so train.py can use it without changes."""

    def __init__(self, mode, random_flip=False):
        if _root is None or _split_json is None:
            raise RuntimeError(
                "Call datasets.RS_HiUCD.configure(root, split_json) "
                "before creating Data()."
            )
        split = "train" if mode == "train" else "val"
        augment = bool(random_flip) and split == "train"
        self._ds = HiUCDMini(_root, _split_json, split, augment=augment)

    # ── SamGraphDataset compatibility ─────────────────────────────────────────
    def get_mask_name(self, idx):
        """Return the sample_id used as the SAM mask cache filename."""
        return self._ds.items[idx]["sample_id"]

    # ── standard Dataset interface ────────────────────────────────────────────
    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]
        # Return tuple matching RS_ST.Data: (img_A, img_B, label_A, label_B)
        return sample["A"], sample["B"], sample["label_A"], sample["label_B"]
