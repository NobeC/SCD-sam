#!/usr/bin/env python3
"""Prepare leakage-free HRSCD splits and aligned 512x512 PNG patches."""

import argparse
import csv
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ARCHIVES = {
    "A": "images_2006.zip", "B": "images_2012.zip",
    "label_A": "labels_land_cover_2006.zip",
    "label_B": "labels_land_cover_2012.zip", "change": "labels_change.zip",
}
SPLIT_ORDER = ("train", "val", "test")
MODALITIES = ("A", "B", "label_A", "label_B", "change")


@dataclass(frozen=True)
class SourcePair:
    index: int
    members: dict


def make_split_indices(total, train, val, test, seed):
    if min(total, train, val, test) < 0:
        raise ValueError("split sizes must be non-negative")
    if train + val + test != total:
        raise ValueError("train + val + test must equal total")
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    trainval = indices[:train + val]
    test_indices = indices[train + val:]
    rng.shuffle(trainval)
    return {"train": sorted(trainval[:train]),
            "val": sorted(trainval[train:]), "test": sorted(test_indices)}


def tile_starts(length, tile_size):
    if tile_size <= 0:
        raise ValueError("tile size must be positive")
    if length < tile_size:
        raise ValueError("image dimension is smaller than tile size")
    starts = list(range(0, length - tile_size + 1, tile_size))
    edge = length - tile_size
    if starts[-1] != edge:
        starts.append(edge)
    return starts


def _tif_members(archive):
    return sorted(name for name in archive.namelist()
                  if name.lower().endswith((".tif", ".tiff")))


def load_source_pairs(archives):
    members = {name: _tif_members(archives[name]) for name in MODALITIES}
    counts = {name: len(paths) for name, paths in members.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("archive TIFF counts differ: {}".format(counts))
    total = next(iter(counts.values()))
    if total != 291:
        raise ValueError("expected 291 source pairs, found {}".format(total))
    return [SourcePair(i, {name: members[name][i] for name in MODALITIES})
            for i in range(total)]


def _decode(archive, member):
    encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
    array = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError("OpenCV could not decode {}".format(member))
    return array


def _validate_arrays(arrays, pair):
    shapes = {name: value.shape[:2] for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError("shape mismatch at source {}: {}".format(pair.index, shapes))
    for name in ("A", "B"):
        if arrays[name].ndim != 3 or arrays[name].shape[2] != 3:
            raise ValueError("{} must be HxWx3 at source {}".format(name, pair.index))
    for name in ("label_A", "label_B", "change"):
        if arrays[name].ndim != 2:
            raise ValueError("{} must be HxW at source {}".format(name, pair.index))
        maximum = int(arrays[name].max())
        if maximum > 255:
            raise ValueError("{} value {} exceeds uint8 at source {}".format(
                name, maximum, pair.index))
        arrays[name] = arrays[name].astype(np.uint8, copy=False)
    return next(iter(shapes.values()))


def _write_png(path, array, label):
    params = [cv2.IMWRITE_PNG_COMPRESSION, 9 if label else 3]
    if not cv2.imwrite(str(path), array, params):
        raise OSError("failed to write {}".format(path))


def prepare_dataset(raw_root, output_root, seed=42, tile_size=512, workers=8):
    raw_root, output_root = Path(raw_root), Path(output_root)
    archive_paths = {name: raw_root / filename for name, filename in ARCHIVES.items()}
    missing = [str(path) for path in archive_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing archives: {}".format(missing))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".incomplete").write_text("generation in progress\n")
    for split in SPLIT_ORDER:
        for modality in MODALITIES:
            (output_root / split / modality).mkdir(parents=True, exist_ok=True)

    archives = {name: zipfile.ZipFile(path) for name, path in archive_paths.items()}
    try:
        pairs = load_source_pairs(archives)
        splits = make_split_indices(len(pairs), 117, 29, 145, seed)
        index_to_split = {i: split for split, values in splits.items() for i in values}
        with (output_root / "original_pairs.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_index", "split"] + list(MODALITIES))
            for pair in pairs:
                writer.writerow([pair.index, index_to_split[pair.index]] +
                                [pair.members[name] for name in MODALITIES])

        patch_counts = {split: 0 for split in SPLIT_ORDER}
        label_values = {name: set() for name in ("label_A", "label_B", "change")}
        with (output_root / "patches.csv").open("w", newline="") as handle, \
                ThreadPoolExecutor(max_workers=workers) as executor:
            writer = csv.writer(handle)
            writer.writerow(["split", "patch_name", "source_index", "y", "x"])
            for completed, pair in enumerate(pairs, 1):
                split = index_to_split[pair.index]
                arrays = {name: _decode(archives[name], pair.members[name])
                          for name in MODALITIES}
                height, width = _validate_arrays(arrays, pair)
                for name in label_values:
                    label_values[name].update(map(int, np.unique(arrays[name])))
                writes = []
                for y in tile_starts(height, tile_size):
                    for x in tile_starts(width, tile_size):
                        patch_counts[split] += 1
                        patch_name = "{:06d}.png".format(patch_counts[split])
                        for name in MODALITIES:
                            patch = arrays[name][y:y + tile_size, x:x + tile_size]
                            writes.append(executor.submit(
                                _write_png,
                                output_root / split / name / patch_name,
                                patch,
                                name not in ("A", "B")))
                        writer.writerow([split, patch_name, pair.index, y, x])
                for write in writes:
                    write.result()
                handle.flush()
                print("[{}/291] source={} split={} patches={}".format(
                    completed, pair.index, split, patch_counts[split]), flush=True)

        info = {
            "dataset": "HRSCD", "seed": seed, "tile_size": tile_size,
            "png_writer_workers": workers,
            "split_protocol": "project-defined original-pair split; release has no membership manifest",
            "edge_policy": "non-overlap plus final edge-anchored tile",
            "source_pair_counts": {k: len(v) for k, v in splits.items()},
            "patch_counts": patch_counts,
            "label_values": {k: sorted(v) for k, v in label_values.items()},
            "archive_files": {k: str(v) for k, v in archive_paths.items()},
        }
        (output_root / "dataset_info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n")
        (output_root / ".incomplete").unlink()
        return patch_counts
    finally:
        for archive in archives.values():
            archive.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("datasets/HRSCD_raw/HRSCD"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/HRSCD"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print("completed: {}".format(prepare_dataset(
        args.raw_root, args.output_root, args.seed, args.tile_size, args.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
