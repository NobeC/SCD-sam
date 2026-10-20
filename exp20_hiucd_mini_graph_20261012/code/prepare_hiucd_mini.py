#!/usr/bin/env python3
"""Audit official train pairs only and create a deterministic spatial holdout."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    rows = [r for r in csv.DictReader((root / "split_manifest.csv").open()) if r["split"] == "train"]
    if len(rows) != 300 or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Expected 300 unique official training pairs only")
    groups = sorted({r["name"].split("_")[0] for r in rows})
    rng = random.Random(args.seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    val_groups, val_count = [], 0
    target = round(.2 * len(rows))
    for group in shuffled:
        group_count = sum(r["name"].split("_")[0] == group for r in rows)
        if len(val_groups) < max(1, round(.2 * len(groups))) and abs(val_count + group_count - target) <= abs(val_count - target):
            val_groups.append(group)
            val_count += group_count
    if not val_groups or len(val_groups) == len(groups):
        raise RuntimeError("Could not create a non-empty grouped holdout")
    patches = {"train": [], "val": []}
    sem_a = np.zeros(10, dtype=np.int64)
    sem_b = np.zeros(10, dtype=np.int64)
    transition_counts = np.zeros(3, dtype=np.int64)
    totals = {"pixels": 0, "unchanged_label_mismatch": 0, "changed_label_same": 0}
    for row in rows:
        images = []
        for key in ("A", "B", "mask_merge"):
            path = root / row[key]
            if not path.is_file():
                raise FileNotFoundError(str(path))
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint8 or image.shape != (1024, 1024, 3):
                raise ValueError("Invalid source PNG: " + str(path))
            images.append(image)
        mask = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
        label_a, label_b, transition = (mask[:, :, i] for i in range(3))
        if int(label_a.max()) > 9 or int(label_b.max()) > 9 or int(transition.max()) > 2:
            raise ValueError("Unexpected label encoding in " + row["name"])
        sem_a += np.bincount(label_a.ravel(), minlength=10)
        sem_b += np.bincount(label_b.ravel(), minlength=10)
        transition_counts += np.bincount(transition.ravel(), minlength=3)
        totals["pixels"] += int(label_a.size)
        totals["unchanged_label_mismatch"] += int(((transition == 1) & (label_a > 0) & (label_b > 0) & (label_a != label_b)).sum())
        totals["changed_label_same"] += int(((transition == 2) & (label_a > 0) & (label_b > 0) & (label_a == label_b)).sum())
        split = "val" if row["name"].split("_")[0] in val_groups else "train"
        for quadrant in range(4):
            patches[split].append({"sample_id": Path(row["name"]).stem + "_q" + str(quadrant),
                                   "source_name": row["name"], "quadrant": quadrant})
    if not patches["train"] or not patches["val"]:
        raise RuntimeError("Empty train or validation subset")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "Hi-UCD mini",
        "protocol": "Official train pairs only; deterministic spatial-prefix holdout; four aligned 512x512 quadrants per 1024x1024 pair; official val/test are not read.",
        "seed": args.seed,
        "label_decode": "PNG RGB channels: semantic_A, semantic_B, transition; 0 is ignore/unknown, transition 1 unchanged, 2 changed.",
        "source_pairs": rows,
        "spatial_groups": {"train": sorted(set(groups) - set(val_groups)), "val": sorted(val_groups)},
        "patches": patches,
        "audit": {"source_pairs": len(rows), "patches": {k: len(v) for k, v in patches.items()},
                  "semantic_A_value_counts": sem_a.tolist(), "semantic_B_value_counts": sem_b.tolist(),
                  "transition_value_counts": transition_counts.tolist(), **totals},
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["split_sha256"] = digest
    (output / "split.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split_sha256": digest, "groups": manifest["spatial_groups"],
                      "audit": manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()
