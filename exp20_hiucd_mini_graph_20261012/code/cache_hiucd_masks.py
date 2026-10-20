#!/usr/bin/env python3
"""Offline-only MobileSAM cache builder for Hi-UCD internal training patches."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.hiucd_mini import HiUCDMini
from utils.sam_cache_format import save_sam_cache_v2, load_sam_cache_v2


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for offline MobileSAM cache generation")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    split_path = Path(args.split_json)
    manifest = json.loads(split_path.read_text())
    items = manifest["patches"]["train"]
    if args.limit is not None:
        items = items[:args.limit]
    cache_root = Path(args.cache_root)
    cache_dir = cache_root / "HiUCDMini" / "train"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "vit_t", "checkpoint_sha256": sha256(checkpoint),
              "points_per_side": 32, "pred_iou_thresh": 0.75,
              "stability_score_thresh": 0.80, "crop_n_layers": 0,
              "min_mask_region_area": 16, "min_area_ratio": 0.001,
              "max_area_ratio": 0.05, "duplicate_iou_thresh": 0.90,
              "containment_thresh": 0.95, "containment_area_ratio": 0.80}
    report_path = cache_root / "HiUCDMini" / "cache_manifest.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("split_sha256") != manifest["split_sha256"] or previous.get("generator_config") != config:
            raise RuntimeError("Existing SAM cache manifest does not match this split/config")
        entries = previous.get("entries", {})
    else:
        entries = {}
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).to(args.device).eval()
    generator = SamAutomaticMaskGenerator(model, points_per_side=32,
        pred_iou_thresh=0.75, stability_score_thresh=0.80, crop_n_layers=0,
        min_mask_region_area=16)
    dataset = HiUCDMini(args.root, split_path, "train", augment=False)
    all_train_ids = {item["sample_id"] for item in manifest["patches"]["train"]}
    if any(item["sample_id"] not in all_train_ids for item in items):
        raise RuntimeError("Cache request contains non-training sample")
    started = time.time()
    for index, item in enumerate(items, 1):
        sample_id = item["sample_id"]
        dest = cache_dir / (sample_id + ".npz")
        if dest.exists():
            cache = load_sam_cache_v2(dest)
            if cache["sample_id"] != sample_id or tuple(cache["image_shape"]) != (512, 512):
                raise ValueError("Invalid existing cache entry: " + sample_id)
            entries[sample_id] = {"path": str(dest), "bytes": dest.stat().st_size,
                                  "sha256": sha256(dest), "resumed": True}
            continue
        image_a, image_b = dataset.read_patch(item)[:2]
        records_a = generator.generate(image_a)
        records_b = generator.generate(image_b)
        save_sam_cache_v2(dest, sample_id, (512, 512), records_a, records_b,
            config, duplicate_iou_thresh=0.90, containment_thresh=0.95,
            containment_area_ratio=0.80, fine_threshold=0.01, mid_threshold=0.05)
        cache = load_sam_cache_v2(dest)
        if cache["sample_id"] != sample_id or tuple(cache["image_shape"]) != (512, 512):
            raise ValueError("Cache round-trip failed: " + sample_id)
        entries[sample_id] = {"path": str(dest), "bytes": dest.stat().st_size,
                              "sha256": sha256(dest), "masks_a": len(records_a),
                              "masks_b": len(records_b), "shared": len(cache["shared_masks"])}
        if index % 10 == 0 or index == len(items):
            report = {"format_version": 2, "dataset": "HiUCDMini", "split_sha256": manifest["split_sha256"],
                      "generator_config": config, "entries": entries, "complete": len(entries) == len(all_train_ids)}
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print("CACHE", index, "/", len(items), sample_id, "elapsed_s", round(time.time() - started, 1), flush=True)
    complete = len(entries) == len(all_train_ids)
    report = {"format_version": 2, "dataset": "HiUCDMini", "split_sha256": manifest["split_sha256"],
              "generator_config": config, "entries": entries, "complete": complete}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not complete:
        print("CACHE_PARTIAL", len(entries), "/", len(all_train_ids), "(limit used)", flush=True)
    else:
        print("CACHE_COMPLETE", len(entries), "elapsed_s", round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    main()
