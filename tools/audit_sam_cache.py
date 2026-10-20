#!/usr/bin/env python3
"""Audit MobileSAM v2 cache integrity and report aggregate statistics."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sam_cache_format import (  # noqa: E402
    SCALE_COARSE,
    SCALE_FINE,
    SCALE_MID,
    load_sam_cache_v2,
)


def audit_cache(cache_dir):
    cache_dir = Path(cache_dir)
    paths = sorted(cache_dir.glob("*.npz"))
    totals = Counter(files=len(paths))
    errors = []
    sizes = []
    for path in paths:
        try:
            cache = load_sam_cache_v2(path)
            height, width = map(int, cache["image_shape"])
            if cache["sample_id"] != path.stem:
                raise ValueError("sample_id does not match filename")
            for key in ("masks_A", "masks_B", "shared_masks"):
                if tuple(cache[key].shape[1:]) != (height, width):
                    raise ValueError("{} has invalid spatial shape".format(key))
            source = cache["shared_source"].astype(np.int64)
            indices = cache["shared_index"].astype(np.int64)
            for side, index in zip(source, indices):
                if side == 0:
                    scale = int(cache["scale_A"][index])
                elif side == 1:
                    scale = int(cache["scale_B"][index])
                else:
                    raise ValueError("invalid shared_source value {}".format(side))
                totals[{SCALE_FINE: "fine", SCALE_MID: "mid", SCALE_COARSE: "coarse"}[scale]] += 1
            totals["valid"] += 1
            totals["masks_a"] += len(cache["masks_A"])
            totals["masks_b"] += len(cache["masks_B"])
            totals["shared"] += len(cache["shared_masks"])
            sizes.append(path.stat().st_size)
        except Exception as error:
            totals["invalid"] += 1
            errors.append({"path": str(path), "error": "{}: {}".format(type(error).__name__, error)})
    valid = max(totals["valid"], 1)
    report = dict(totals)
    report.update({
        "cache_dir": str(cache_dir.resolve()),
        "cache_bytes": int(sum(sizes)),
        "mean_cache_bytes": float(np.mean(sizes)) if sizes else 0.0,
        "mean_masks_a": totals["masks_a"] / float(valid),
        "mean_masks_b": totals["masks_b"] / float(valid),
        "mean_shared": totals["shared"] / float(valid),
        "errors": errors,
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_cache(args.cache_dir)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 1 if report.get("invalid", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
