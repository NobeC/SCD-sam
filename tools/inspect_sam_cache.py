#!/usr/bin/env python3
"""Render deterministic visual panels for MobileSAM v2 cache inspection."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sam_cache_format import SCALE_MID, load_sam_cache_v2  # noqa: E402


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: {}".format(path))
    return image


def image_map(directory):
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return {path.stem: path for path in Path(directory).iterdir()
            if path.is_file() and path.suffix.lower() in allowed}


def overlay_masks(image, masks, seed):
    output = image.astype(np.float32).copy()
    rng = np.random.RandomState(seed)
    for mask in masks:
        color = rng.randint(30, 256, size=3).astype(np.float32)
        output[mask] = output[mask] * 0.48 + color * 0.52
    return np.clip(output, 0, 255).astype(np.uint8)


def labelled_panel(image, label):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 220), 30), (0, 0, 0), -1)
    cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _shared_mid_masks(cache):
    result = []
    for mask, side, index in zip(cache["shared_masks"], cache["shared_source"], cache["shared_index"]):
        scales = cache["scale_A"] if int(side) == 0 else cache["scale_B"]
        if int(scales[int(index)]) == SCALE_MID:
            result.append(mask)
    height, width = map(int, cache["image_shape"])
    return np.stack(result) if result else np.empty((0, height, width), dtype=np.bool_)


def render_sample(cache_path, path_a, path_b, output_path, seed=0):
    cache = load_sam_cache_v2(cache_path)
    image_a, image_b = read_bgr(path_a), read_bgr(path_b)
    expected = tuple(int(value) for value in cache["image_shape"])
    if image_a.shape[:2] != expected or image_b.shape[:2] != expected:
        raise ValueError("image/cache shape mismatch for {}".format(cache["sample_id"]))
    panels = [
        labelled_panel(image_a, "A: original"),
        labelled_panel(image_b, "B: original"),
        labelled_panel(overlay_masks(image_a, cache["masks_A"], seed + 1), "A: raw masks"),
        labelled_panel(overlay_masks(image_b, cache["masks_B"], seed + 2), "B: raw masks"),
        labelled_panel(overlay_masks(image_a, cache["shared_masks"], seed + 3), "A/B: shared"),
        labelled_panel(overlay_masks(image_a, _shared_mid_masks(cache), seed + 4), "shared: mid scale"),
    ]
    canvas = np.concatenate((np.concatenate(panels[:3], axis=1),
                             np.concatenate(panels[3:], axis=1)), axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError("failed to write {}".format(output_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-a", type=Path, required=True)
    parser.add_argument("--dir-b", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    maps = image_map(args.dir_a), image_map(args.dir_b)
    cache_paths = sorted(args.cache_dir.glob("*.npz"))
    rng = np.random.RandomState(args.seed)
    if len(cache_paths) > args.num_samples:
        selected = sorted(rng.choice(len(cache_paths), size=args.num_samples, replace=False))
        cache_paths = [cache_paths[index] for index in selected]
    failures = 0
    for index, cache_path in enumerate(cache_paths):
        sample_id = cache_path.stem
        try:
            render_sample(cache_path, maps[0][sample_id], maps[1][sample_id],
                          args.output / (sample_id + ".png"), args.seed + index * 10)
            print("wrote {}".format(args.output / (sample_id + ".png")))
        except Exception as error:
            failures += 1
            print("FAILED {}: {}".format(sample_id, error), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
