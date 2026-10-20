#!/usr/bin/env python3
"""Audit GT/U/Ignore usability of cached shared SAM masks without training."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.datasets import get_dataset_config  # noqa: E402
from utils.sam_cache_format import load_sam_cache_v2  # noqa: E402
from utils.sam_node_audit import GT, IGNORE, UNLABELED  # noqa: E402


AREA_BIN_NAMES = (
    "<0.1%", "0.1%-0.25%", "0.25%-0.5%", "0.5%-1%",
    "1%-2%", "2%-5%", "5%-10%", ">10%",
)
AREA_BIN_EDGES = np.asarray([0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10])
RANGES = (
    ("A_0.1-5", 0.001, 0.05),
    ("B_0.25-5", 0.0025, 0.05),
    ("C_0.5-5", 0.005, 0.05),
    ("D_1-5", 0.01, 0.05),
    ("E_0.1-10", 0.001, 0.10),
    ("F_0.25-10", 0.0025, 0.10),
    ("G_0.5-10", 0.005, 0.10),
)
STATE_NAMES = {IGNORE: "Ignore", GT: "GT", UNLABELED: "U"}


def assign_area_bins(ratios):
    """Assign required bins; exactly 10% remains in the 5%-10% bin."""
    ratios = np.asarray(ratios, dtype=np.float64)
    result = np.searchsorted(AREA_BIN_EDGES, ratios, side="right")
    result[ratios == AREA_BIN_EDGES[-1]] = len(AREA_BIN_EDGES) - 1
    return result.astype(np.int64)


def pair_counts(classes):
    classes = np.asarray(classes, dtype=np.int64)
    classes = classes[classes > 0]
    total = len(classes) * (len(classes) - 1) // 2
    if not len(classes):
        return 0, 0
    _, counts = np.unique(classes, return_counts=True)
    positive = int(sum(int(count) * (int(count) - 1) // 2 for count in counts))
    return positive, int(total - positive)


def component_coverage(changed, covered, small_max, medium_max, cover_threshold=0.5):
    """Return [component count, covered count] for three data-scale groups."""
    changed = np.asarray(changed, dtype=np.bool_)
    covered = np.asarray(covered, dtype=np.bool_)
    if changed.shape != covered.shape or changed.ndim != 2:
        raise ValueError("changed and covered must have matching [H,W] shapes")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8), connectivity=8)
    result = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        group = "small" if area <= small_max else "medium" if area <= medium_max else "large"
        hit = int(np.logical_and(labels == component_id, covered).sum())
        result[group][0] += 1
        result[group][1] += int(hit / float(max(area, 1)) >= cover_threshold)
    return result


def _read_color_label(path, config):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot read label {}".format(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    encoded = (
        rgb[:, :, 0].astype(np.int32) * 65536
        + rgb[:, :, 1].astype(np.int32) * 256
        + rgb[:, :, 2].astype(np.int32)
    )
    label = np.full(encoded.shape, -1, dtype=np.int16)
    for class_id, color in enumerate(config.colormap):
        code = int(color[0]) * 65536 + int(color[1]) * 256 + int(color[2])
        label[encoded == code] = class_id
    if (label < 0).any():
        unknown = np.unique(rgb[label < 0].reshape(-1, 3), axis=0)[:10].tolist()
        raise ValueError("{} contains unknown colors {}".format(path, unknown))
    return label


def _temporal_assignments(masks, labels, changed_pixels, change_ratio,
                          mask_areas, num_classes, args):
    count = len(masks)
    states = np.full(count, IGNORE, dtype=np.int8)
    classes = np.zeros(count, dtype=np.int16)
    purity = np.zeros(count, dtype=np.float32)
    best_classes = np.zeros(count, dtype=np.int16)
    if not count:
        return states, classes, purity, best_classes
    class_counts = np.stack(
        [np.logical_and(masks, labels[None] == class_id).reshape(count, -1).sum(1)
         for class_id in range(1, num_classes)], axis=1)
    best_offsets = class_counts.argmax(1)
    best_counts = class_counts[np.arange(count), best_offsets]
    best_classes = (best_offsets + 1).astype(np.int16)
    best_classes[best_counts == 0] = 0
    purity = (best_counts / np.maximum(changed_pixels, 1)).astype(np.float32)
    enough_mask = mask_areas >= args.min_mask_pixels
    states[enough_mask & ((1.0 - change_ratio) > args.tau_u)] = UNLABELED
    # Match the current training implementation: ratio/purity/changed thresholds are strict >.
    reliable = (
        enough_mask
        & (change_ratio > args.tau_gt)
        & (changed_pixels > args.min_changed_pixels)
        & (purity > args.tau_purity)
        & (best_classes > 0)
    )
    states[reliable] = GT
    classes[reliable] = best_classes[reliable]
    return states, classes, purity, best_classes


def analyze_masks(masks, labels_a, labels_b, num_classes, args):
    masks = np.asarray(masks, dtype=np.bool_)
    changed = (labels_a != 0) | (labels_b != 0)
    areas = masks.reshape(len(masks), -1).sum(1).astype(np.int64)
    changed_pixels = np.logical_and(masks, changed[None]).reshape(len(masks), -1).sum(1).astype(np.int64)
    ratios = areas.astype(np.float64) / float(changed.size)
    change_ratio = changed_pixels / np.maximum(areas, 1)
    states_a, classes_a, purity_a, best_a = _temporal_assignments(
        masks, labels_a, changed_pixels, change_ratio, areas, num_classes, args)
    states_b, classes_b, purity_b, best_b = _temporal_assignments(
        masks, labels_b, changed_pixels, change_ratio, areas, num_classes, args)
    return {
        "changed": changed, "areas": areas, "area_ratios": ratios,
        "changed_pixels": changed_pixels, "change_ratio": change_ratio,
        "states_a": states_a, "states_b": states_b,
        "classes_a": classes_a, "classes_b": classes_b,
        "purity_a": purity_a, "purity_b": purity_b,
        "best_a": best_a, "best_b": best_b,
    }


def _new_accumulator(name, minimum, maximum, num_classes):
    return {
        "name": name, "minimum": minimum, "maximum": maximum,
        "nodes_per_image": [], "gt_per_image": [], "positive_per_image": [],
        "negative_per_image": [], "gt_classes_per_image": [],
        "one_class_per_image": [], "gt_purities": [],
        "total_nodes": 0, "gt": 0, "u": 0, "ignore": 0,
        "changed_total": 0, "covered_changed": 0,
        "covered_changed_a": 0, "covered_changed_b": 0,
        "class_gt": np.zeros(num_classes, dtype=np.int64),
        "class_gt_a": np.zeros(num_classes, dtype=np.int64),
        "class_gt_b": np.zeros(num_classes, dtype=np.int64),
        "class_images": np.zeros(num_classes, dtype=np.int64),
        "class_presence": [[] for _ in range(num_classes)],
        "class_purities": [[] for _ in range(num_classes)],
        "class_pixels": np.zeros(num_classes, dtype=np.int64),
        "class_covered": np.zeros(num_classes, dtype=np.int64),
        "component_hits": [],
    }


def _update_accumulator(acc, masks, analysis, selected, labels_a, labels_b, num_classes,
                        component_labels, component_areas, component_cover_threshold):
    selected_masks = masks[selected]
    states_a, states_b = analysis["states_a"][selected], analysis["states_b"][selected]
    classes_a, classes_b = analysis["classes_a"][selected], analysis["classes_b"][selected]
    purities_a, purities_b = analysis["purity_a"][selected], analysis["purity_b"][selected]
    node_count = int(selected.sum())
    states = np.concatenate((states_a, states_b))
    classes = np.concatenate((classes_a, classes_b))
    gt_classes = classes[states == GT]
    positive, negative = pair_counts(gt_classes)
    acc["nodes_per_image"].append(node_count)
    acc["gt_per_image"].append(len(gt_classes))
    acc["positive_per_image"].append(positive)
    acc["negative_per_image"].append(negative)
    acc["gt_classes_per_image"].append(gt_classes.copy())
    acc["one_class_per_image"].append(int(len(np.unique(gt_classes)) == 1 and len(gt_classes) > 0))
    acc["total_nodes"] += node_count
    acc["gt"] += int((states == GT).sum())
    acc["u"] += int((states == UNLABELED).sum())
    acc["ignore"] += int((states == IGNORE).sum())
    acc["changed_total"] += int(analysis["changed"].sum())

    union_valid = selected_masks.any(0) if node_count else np.zeros_like(analysis["changed"])
    hits = np.bincount(
        component_labels.reshape(-1), weights=union_valid.reshape(-1).astype(np.int64),
        minlength=len(component_areas) + 1)[1:]
    acc["component_hits"].extend(
        (hits / np.maximum(component_areas, 1) >= component_cover_threshold).tolist())

    gt_a = states_a == GT
    gt_b = states_b == GT
    union_a = selected_masks[gt_a].any(0) if gt_a.any() else np.zeros_like(analysis["changed"])
    union_b = selected_masks[gt_b].any(0) if gt_b.any() else np.zeros_like(analysis["changed"])
    union_any = union_a | union_b
    acc["covered_changed"] += int((union_any & analysis["changed"]).sum())
    acc["covered_changed_a"] += int((union_a & analysis["changed"]).sum())
    acc["covered_changed_b"] += int((union_b & analysis["changed"]).sum())

    present = set()
    for class_id in range(1, num_classes):
        class_a = gt_a & (classes_a == class_id)
        class_b = gt_b & (classes_b == class_id)
        count_a, count_b = int(class_a.sum()), int(class_b.sum())
        acc["class_gt_a"][class_id] += count_a
        acc["class_gt_b"][class_id] += count_b
        acc["class_gt"][class_id] += count_a + count_b
        if count_a + count_b:
            present.add(class_id)
            acc["class_images"][class_id] += 1
            acc["class_purities"][class_id].extend(purities_a[class_a].tolist())
            acc["class_purities"][class_id].extend(purities_b[class_b].tolist())
        pixels_a, pixels_b = labels_a == class_id, labels_b == class_id
        acc["class_pixels"][class_id] += int(pixels_a.sum() + pixels_b.sum())
        cover_a = selected_masks[class_a].any(0) if class_a.any() else np.zeros_like(pixels_a)
        cover_b = selected_masks[class_b].any(0) if class_b.any() else np.zeros_like(pixels_b)
        acc["class_covered"][class_id] += int((cover_a & pixels_a).sum() + (cover_b & pixels_b).sum())
        acc["class_presence"][class_id].append(class_id in present)
    acc["gt_purities"].extend(purities_a[gt_a].tolist())
    acc["gt_purities"].extend(purities_b[gt_b].tolist())


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def _longest_false_run(values):
    best = current = 0
    for value in values:
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def _batch_statistics(classes_per_image, batch_size, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(classes_per_image))
    gt_counts, positives, negatives = [], [], []
    for start in range(0, len(order), batch_size):
        pieces = [classes_per_image[index] for index in order[start:start + batch_size]]
        classes = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        positive, negative = pair_counts(classes)
        gt_counts.append(len(classes)); positives.append(positive); negatives.append(negative)
    return {
        "batches": len(gt_counts),
        "mean_gt_nodes_per_batch": float(np.mean(gt_counts)) if gt_counts else 0.0,
        "mean_positive_pairs_per_batch": float(np.mean(positives)) if positives else 0.0,
        "mean_negative_pairs_per_batch": float(np.mean(negatives)) if negatives else 0.0,
        "zero_positive_batch_ratio": _safe_ratio(sum(value == 0 for value in positives), len(positives)),
        "zero_negative_batch_ratio": _safe_ratio(sum(value == 0 for value in negatives), len(negatives)),
    }


def _range_summary(acc, component_sizes, small_max, medium_max, args):
    images = len(acc["nodes_per_image"])
    temporal = 2 * acc["total_nodes"]
    gt_per_image = np.asarray(acc["gt_per_image"])
    positives = np.asarray(acc["positive_per_image"])
    negatives = np.asarray(acc["negative_per_image"])
    component_sizes = np.asarray(component_sizes)
    component_hits = np.asarray(acc["component_hits"], dtype=np.bool_)
    object_coverage = {}
    for group, selector in (
        ("small", component_sizes <= small_max),
        ("medium", (component_sizes > small_max) & (component_sizes <= medium_max)),
        ("large", component_sizes > medium_max),
    ):
        object_coverage[group] = {
            "components": int(selector.sum()),
            "covered": int(component_hits[selector].sum()),
            "coverage": _safe_ratio(component_hits[selector].sum(), selector.sum()),
        }
    summary = {
        "range_name": acc["name"], "area_min": acc["minimum"], "area_max": acc["maximum"],
        "total_nodes": acc["total_nodes"], "temporal_nodes": temporal,
        "mean_nodes_per_image": _safe_ratio(acc["total_nodes"], images),
        "gt_nodes": acc["gt"], "mean_gt_nodes_per_image": _safe_ratio(acc["gt"], images),
        "u_nodes": acc["u"], "ignore_nodes": acc["ignore"],
        "gt_ratio": _safe_ratio(acc["gt"], temporal),
        "u_ratio": _safe_ratio(acc["u"], temporal),
        "ignore_ratio": _safe_ratio(acc["ignore"], temporal),
        "changed_coverage": _safe_ratio(acc["covered_changed"], acc["changed_total"]),
        "coverage_a": _safe_ratio(acc["covered_changed_a"], acc["changed_total"]),
        "coverage_b": _safe_ratio(acc["covered_changed_b"], acc["changed_total"]),
        "gt_purity_mean": float(np.mean(acc["gt_purities"])) if acc["gt_purities"] else 0.0,
        "gt_purity_median": float(np.median(acc["gt_purities"])) if acc["gt_purities"] else 0.0,
        "positive_pairs": int(positives.sum()), "negative_pairs": int(negatives.sum()),
        "mean_positive_pairs_per_image": float(positives.mean()) if len(positives) else 0.0,
        "mean_negative_pairs_per_image": float(negatives.mean()) if len(negatives) else 0.0,
        "zero_gt_image_ratio": _safe_ratio((gt_per_image == 0).sum(), images),
        "one_gt_image_ratio": _safe_ratio((gt_per_image == 1).sum(), images),
        "one_gt_class_image_ratio": _safe_ratio(sum(acc["one_class_per_image"]), images),
        "zero_positive_ratio": _safe_ratio((positives == 0).sum(), images),
        "zero_negative_ratio": _safe_ratio((negatives == 0).sum(), images),
        "small_object_coverage": object_coverage["small"]["coverage"],
        "medium_object_coverage": object_coverage["medium"]["coverage"],
        "large_object_coverage": object_coverage["large"]["coverage"],
        "object_coverage": object_coverage,
    }
    summary.update(_batch_statistics(acc["gt_classes_per_image"], args.batch_size, args.seed))
    return summary


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _create_plots(output_dir, area_ratios, accumulators, config, plot_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = next(acc for acc in accumulators if acc["name"] == plot_range)
    positive_ratios = np.asarray(area_ratios)[np.asarray(area_ratios) > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(positive_ratios * 100.0, bins=np.logspace(-3, 2, 70), color="steelblue")
    ax.set_xscale("log"); ax.set_xlabel("Mask area ratio (%)"); ax.set_ylabel("Mask count")
    fig.tight_layout(); fig.savefig(str(output_dir / "mask_area_ratio_hist.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(selected["gt_per_image"], bins=range(0, max(selected["gt_per_image"] + [0]) + 2), color="seagreen")
    ax.set_xlabel("GT temporal nodes per image ({})".format(plot_range)); ax.set_ylabel("Images")
    fig.tight_layout(); fig.savefig(str(output_dir / "gt_nodes_per_image_hist.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(selected["gt_purities"], bins=30, range=(0, 1), color="darkorange")
    ax.set_xlabel("GT node purity ({})".format(plot_range)); ax.set_ylabel("GT nodes")
    fig.tight_layout(); fig.savefig(str(output_dir / "gt_purity_hist.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ids = np.arange(1, config.num_classes)
    ax.bar([config.classes[index] for index in ids], selected["class_gt"][1:], color="mediumpurple")
    ax.set_ylabel("GT temporal nodes ({})".format(plot_range)); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(str(output_dir / "gt_nodes_by_class.png"), dpi=160); plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("landsat", "second"), default="landsat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tau-gt", type=float, default=0.8)
    parser.add_argument("--tau-purity", type=float, default=0.9)
    parser.add_argument("--tau-u", type=float, default=0.95)
    parser.add_argument("--min-mask-pixels", type=int, default=4)
    parser.add_argument("--min-changed-pixels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--component-cover-threshold", type=float, default=0.5)
    parser.add_argument("--small-component-max", type=int)
    parser.add_argument("--medium-component-max", type=int)
    parser.add_argument("--plot-range", choices=tuple(item[0] for item in RANGES), default="B_0.25-5")
    parser.add_argument("--limit", type=int)
    return parser


def _validate_args(args):
    for name in ("tau_gt", "tau_purity", "tau_u", "component_cover_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("--{} must be finite and in [0,1]".format(name.replace("_", "-")))
    if args.min_mask_pixels < 0 or args.min_changed_pixels < 0:
        raise ValueError("minimum pixel counts must be non-negative")
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("batch-size and limit must be positive")


def run_audit(args):
    _validate_args(args)
    config = get_dataset_config(args.dataset)
    data_root = args.data_root or Path(config.root)
    cache_dir = args.cache_dir or Path("cache/sam") / config.name / args.split
    output_dir = args.output_dir or Path("results/sam_node_audit") / config.name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    label_a_dir, label_b_dir = data_root / args.split / "label1", data_root / args.split / "label2"
    cache_paths = sorted(cache_dir.glob("*.npz"))
    if args.limit is not None:
        cache_paths = cache_paths[:args.limit]
    if not cache_paths:
        raise ValueError("no cache files found in {}".format(cache_dir))
    label_a = {path.stem: path for path in label_a_dir.glob("*.png")}
    label_b = {path.stem: path for path in label_b_dir.glob("*.png")}
    if args.limit is None:
        cache_ids = {path.stem for path in cache_paths}
        if cache_ids != set(label_a) or cache_ids != set(label_b):
            raise ValueError(
                "cache/label sample sets differ: cache={} label_A={} label_B={}".format(
                    len(cache_ids), len(label_a), len(label_b)))
    missing = [path.stem for path in cache_paths if path.stem not in label_a or path.stem not in label_b]
    if missing:
        raise ValueError("missing A/B labels for {}".format(missing[:20]))

    accumulators = [_new_accumulator(*item, config.num_classes) for item in RANGES]
    area_counts_per_image = []
    area_ratios = []
    component_sizes = []
    errors = []
    details_path = output_dir / "sam_mask_gt_records.csv"
    detail_fields = [
        "sample_id", "mask_index", "mask_area", "area_ratio", "changed_pixels",
        "change_ratio", "best_class_A", "purity_A", "state_A",
        "best_class_B", "purity_B", "state_B",
    ]
    with details_path.open("w", newline="", encoding="utf-8") as detail_stream:
        detail_writer = csv.DictWriter(detail_stream, fieldnames=detail_fields)
        detail_writer.writeheader()
        for position, cache_path in enumerate(cache_paths, start=1):
            sample_id = cache_path.stem
            try:
                cache = load_sam_cache_v2(cache_path)
                labels_a = _read_color_label(label_a[sample_id], config)
                labels_b = _read_color_label(label_b[sample_id], config)
                masks = cache["shared_masks"]
                if cache["sample_id"] != sample_id or tuple(masks.shape[1:]) != labels_a.shape or labels_b.shape != labels_a.shape:
                    raise ValueError("cache/label identity or shape mismatch")
                analysis = analyze_masks(masks, labels_a, labels_b, config.num_classes, args)
                ratios = analysis["area_ratios"]
                area_ratios.extend(ratios.tolist())
                area_counts_per_image.append(np.bincount(assign_area_bins(ratios), minlength=len(AREA_BIN_NAMES)))

                component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                    analysis["changed"].astype(np.uint8), connectivity=8)
                areas = component_stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
                component_sizes.extend(areas.tolist())
                for acc in accumulators:
                    selected = (ratios >= acc["minimum"]) & (ratios <= acc["maximum"])
                    _update_accumulator(
                        acc, masks, analysis, selected, labels_a, labels_b, config.num_classes,
                        component_labels, areas, args.component_cover_threshold)
                for index in range(len(masks)):
                    detail_writer.writerow({
                        "sample_id": sample_id, "mask_index": index,
                        "mask_area": int(analysis["areas"][index]),
                        "area_ratio": float(ratios[index]),
                        "changed_pixels": int(analysis["changed_pixels"][index]),
                        "change_ratio": float(analysis["change_ratio"][index]),
                        "best_class_A": int(analysis["best_a"][index]),
                        "purity_A": float(analysis["purity_a"][index]),
                        "state_A": STATE_NAMES[int(analysis["states_a"][index])],
                        "best_class_B": int(analysis["best_b"][index]),
                        "purity_B": float(analysis["purity_b"][index]),
                        "state_B": STATE_NAMES[int(analysis["states_b"][index])],
                    })
                if position == 1 or position % 100 == 0 or position == len(cache_paths):
                    print("audited {}/{} {} shared_masks={}".format(
                        position, len(cache_paths), sample_id, len(masks)), flush=True)
            except Exception as error:
                errors.append({"sample_id": sample_id, "error": "{}: {}".format(type(error).__name__, error)})
                print("FAILED {}: {}".format(sample_id, error), flush=True)

    if errors:
        raise RuntimeError("{} samples failed; first errors: {}".format(len(errors), errors[:5]))
    component_sizes_array = np.asarray(component_sizes, dtype=np.int64)
    derived_small = max(1, int(np.percentile(component_sizes_array, 50))) if len(component_sizes_array) else 1
    derived_medium = max(derived_small, int(np.percentile(component_sizes_array, 90))) if len(component_sizes_array) else 1
    small_max = args.small_component_max if args.small_component_max is not None else derived_small
    medium_max = args.medium_component_max if args.medium_component_max is not None else derived_medium
    if small_max > medium_max:
        raise ValueError("small-component-max must not exceed medium-component-max")

    area_matrix = np.stack(area_counts_per_image)
    total_masks = int(area_matrix.sum())
    area_rows = []
    for index, name in enumerate(AREA_BIN_NAMES):
        values = area_matrix[:, index]
        area_rows.append({
            "area_bin": name, "mask_count": int(values.sum()),
            "ratio": _safe_ratio(values.sum(), total_masks),
            "mean_per_image": float(values.mean()), "median_per_image": float(np.median(values)),
            "p25": _percentile(values, 25), "p75": _percentile(values, 75),
            "p90": _percentile(values, 90), "min_per_image": int(values.min()),
            "max_per_image": int(values.max()),
        })
    area_csv = output_dir / "sam_area_distribution.csv"
    _write_csv(area_csv, area_rows, list(area_rows[0]))

    range_summaries = [
        _range_summary(acc, component_sizes, small_max, medium_max, args)
        for acc in accumulators
    ]
    range_csv = output_dir / "sam_node_range_comparison.csv"
    range_fields = [key for key in range_summaries[0] if key != "object_coverage"]
    _write_csv(range_csv, range_summaries, range_fields)

    class_rows = []
    for acc in accumulators:
        for class_id in range(1, config.num_classes):
            purities = acc["class_purities"][class_id]
            class_rows.append({
                "range_name": acc["name"], "area_min": acc["minimum"], "area_max": acc["maximum"],
                "class_id": class_id, "class_name": config.classes[class_id],
                "gt_nodes": int(acc["class_gt"][class_id]),
                "gt_nodes_A": int(acc["class_gt_a"][class_id]),
                "gt_nodes_B": int(acc["class_gt_b"][class_id]),
                "mean_per_image": _safe_ratio(acc["class_gt"][class_id], len(cache_paths)),
                "images_with_class": int(acc["class_images"][class_id]),
                "max_consecutive_images_without_class": _longest_false_run(acc["class_presence"][class_id]),
                "purity_mean": float(np.mean(purities)) if purities else 0.0,
                "purity_median": float(np.median(purities)) if purities else 0.0,
                "changed_pixel_coverage": _safe_ratio(
                    acc["class_covered"][class_id], acc["class_pixels"][class_id]),
            })
    class_csv = output_dir / "sam_gt_class_distribution.csv"
    _write_csv(class_csv, class_rows, list(class_rows[0]))
    _create_plots(output_dir, area_ratios, accumulators, config, args.plot_range)

    report = {
        "dataset": config.name, "split": args.split, "images": len(cache_paths),
        "shared_masks": total_masks, "num_classes": config.num_classes,
        "class_names": list(config.classes),
        "changed_definition": "(label_A != 0) | (label_B != 0)",
        "threshold_semantics": "current training behavior: area >= min_mask; change_ratio, purity, changed_pixels, tau_u use strict >",
        "thresholds": {
            "tau_gt": args.tau_gt, "tau_purity": args.tau_purity, "tau_u": args.tau_u,
            "min_mask_pixels": args.min_mask_pixels,
            "min_changed_pixels": args.min_changed_pixels,
            "component_cover_threshold": args.component_cover_threshold,
        },
        "component_distribution": {
            "count": len(component_sizes), "small_max": small_max, "medium_max": medium_max,
            "derivation": "small <= component area p50; medium <= p90; large > p90 unless CLI overrides",
            "p25": _percentile(component_sizes, 25), "p50": _percentile(component_sizes, 50),
            "p75": _percentile(component_sizes, 75), "p90": _percentile(component_sizes, 90),
            "p95": _percentile(component_sizes, 95), "max": int(max(component_sizes)) if component_sizes else 0,
        },
        "area_distribution": area_rows, "ranges": range_summaries,
        "class_distribution": class_rows,
        "outputs": {
            "area_csv": str(area_csv.resolve()), "range_csv": str(range_csv.resolve()),
            "class_csv": str(class_csv.resolve()), "mask_records_csv": str(details_path.resolve()),
            "plots": [str((output_dir / name).resolve()) for name in (
                "mask_area_ratio_hist.png", "gt_nodes_per_image_hist.png",
                "gt_purity_hist.png", "gt_nodes_by_class.png")],
        },
    }
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("audit_report={}".format(report_path.resolve()), flush=True)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
