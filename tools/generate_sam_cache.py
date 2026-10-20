#!/usr/bin/env python3
"""
generate_sam_cache —— 离线预生成 MobileSAM 超像素 mask 缓存

功能:
    对成对的 T1/T2 时相影像，分别跑 MobileSAM 的 automatic mask generator，
    然后把两时相的 mask 合并去重、按面积分尺度 (fine/mid/coarse)、打包压缩成 .npz 文件。

    训练时 Graph 分支直接 load 这些缓存，避免每个 epoch 都重新跑 SAM，
    大幅加速训练。同时去重逻辑保证 T1/T2 共享一组 mask，天然满足"同一组 mask 覆盖两个时相"
    的要求 (见 sam_node_audit.py)。

输出 .npz 格式 (CACHE_VERSION=2):
    原始 A/B mask → packbits 压缩存 packed_A / packed_B
    去重后的共享 mask → 存 shared_source (0=T1/1=T2) + shared_index (原始索引)
    每个 mask 的属性: area, bbox, iou, stability, scale (0=fine, 1=mid, 2=coarse)
    元数据: sample_id, image_shape, generator_config, scale_thresholds

用法:
    python tools/generate_sam_cache.py \
        --dir-a datasets/RS_ST/T1 --dir-b datasets/RS_ST/T2 \
        --output sam_mask_cache/RS_ST --checkpoint weights/mobile_sam.pt
"""

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 路径设置: 把项目根目录加到 sys.path，方便直接 import utils 模块
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sam_cache_format import (  # noqa: E402
    SCALE_COARSE,   # = 2, 大尺度 mask (面积占比 > mid_threshold)
    SCALE_FINE,     # = 0, 小尺度 mask (面积占比 < fine_threshold)
    SCALE_MID,      # = 1, 中尺度 mask
    load_sam_cache_v2,    # 读取缓存 (训练时用)
    save_sam_cache_v2,    # 写入缓存 (本工具用)
)


# 支持的图像后缀
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _image_files(directory):
    """扫描目录下所有支持的图像文件，返回 {文件名: 完整路径} 的字典。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("image directory does not exist: {}".format(directory))
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def pair_image_paths(dir_a, dir_b, limit=None):
    """按文件名严格配对 A/B 时相图像。

    两个目录必须包含完全相同的文件名集合，缺任何一边都会报错
    (避免静默跳过导致训练数据对不齐)。

    Args:
        dir_a:  T1 时相图像目录
        dir_b:  T2 时相图像目录
        limit:  最多处理多少对 (调试用)

    Returns:
        [(path_a, path_b), ...] 按文件名排序的配对列表
    """
    files_a, files_b = _image_files(dir_a), _image_files(dir_b)

    # 检查两边文件名是否完全一致
    missing_b = sorted(set(files_a).difference(files_b))   # A 有 B 没有
    missing_a = sorted(set(files_b).difference(files_a))   # B 有 A 没有
    if missing_a or missing_b:
        details = []
        if missing_b:
            details.append("missing in B: {}".format(", ".join(missing_b[:20])))
        if missing_a:
            details.append("missing in A: {}".format(", ".join(missing_a[:20])))
        raise ValueError("A/B filenames do not match ({})".format("; ".join(details)))

    names = sorted(files_a)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        names = names[:limit]
    return [(files_a[name], files_b[name]) for name in names]


def read_rgb(path):
    """读取图像并做格式校验，返回 HxWx3 的 uint8 RGB 数组。

    必须是 3 通道 uint8，否则报错 (SAM 不支持灰度或 16-bit)。
    OpenCV 默认读 BGR，这里转成 RGB 再喂给 SAM。
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("cannot read image: {}".format(path))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "expected uint8 HxWx3 image at {}, got dtype={} shape={}".format(
                path, image.dtype, image.shape
            )
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_parser():
    """构建命令行参数解析器。

    MobileSAM 相关参数 (通过 SamAutomaticMaskGenerator 传入):
        --points-per-side:               每条边上采样多少提示点 (默认 64)，越多 mask 越密
        --pred-iou-thresh:               SAM 内部预测的 mask IoU 阈值 (默认 0.75)
        --stability-score-thresh:        SAM 内部的稳定性分数阈值 (默认 0.80)
        --crop-n-layers:                 图像裁剪层数 (默认 2)，多层裁剪提高小目标召回
        --crop-n-points-downscale-factor: 裁剪时点数下采样因子 (默认 2)
        --min-mask-region-area:          最小 mask 区域像素数 (默认 16)

    去重/分尺度参数 (save_sam_cache_v2 内部用):
        --duplicate-iou-thresh:     两 mask IoU > 此值视为重复 (默认 0.90)
        --containment-thresh:       小 mask 被大 mask 包含比例 > 此值视为重复 (默认 0.95)
        --containment-area-ratio:  触发 containment 判定前，小/大面积比须 > 此值 (默认 0.80)
        --fine-threshold:           fine 尺度上限: 面积占比 < 此值 → FINE (默认 0.01)
        --mid-threshold:            mid 尺度上限: 面积占比 < 此值 → MID，否则 COARSE (默认 0.05)
    """
    parser = argparse.ArgumentParser(
        description="Generate safe compressed MobileSAM caches from paired A/B images."
    )
    parser.add_argument("--dir-a", type=Path, required=True)
    parser.add_argument("--dir-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.75)
    parser.add_argument("--stability-score-thresh", type=float, default=0.80)
    parser.add_argument("--crop-n-layers", type=int, default=2)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=int, default=16)
    parser.add_argument("--duplicate-iou-thresh", type=float, default=0.90)
    parser.add_argument("--containment-thresh", type=float, default=0.95)
    parser.add_argument("--containment-area-ratio", type=float, default=0.80)
    parser.add_argument("--fine-threshold", type=float, default=0.01)
    parser.add_argument("--mid-threshold", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-log", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args):
    """校验命令行参数合法性，提前拦截非法组合。"""
    if not args.checkpoint.is_file():
        raise ValueError("checkpoint does not exist: {}".format(args.checkpoint))
    if args.points_per_side <= 0:
        raise ValueError("points-per-side must be positive")
    if args.crop_n_layers < 0 or args.crop_n_points_downscale_factor <= 0:
        raise ValueError("crop parameters must be non-negative/positive")
    if args.min_mask_region_area < 0:
        raise ValueError("min-mask-region-area must be non-negative")

    # 所有阈值必须在 [0, 1] 且为有限值
    thresholds = (
        args.pred_iou_thresh,
        args.stability_score_thresh,
        args.duplicate_iou_thresh,
        args.containment_thresh,
        args.containment_area_ratio,
        args.fine_threshold,
        args.mid_threshold,
    )
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("all score/area thresholds must be finite and in [0, 1]")

    # fine 阈值不能大于 mid 阈值 (否则分尺度逻辑反了)
    if args.fine_threshold > args.mid_threshold:
        raise ValueError("fine-threshold must not exceed mid-threshold")


def _append_error(path, record):
    """把单条错误记录以 JSONL 格式追加到错误日志文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _scale_counts(cache):
    """统计缓存中共享 mask 的尺度分布 (fine / mid / coarse 各多少个)。

    cache 中每个共享 mask 对应一个原始 A/B mask，通过 shared_source + shared_index
    找到它来自哪一边、原始索引多少，再从 scale_A / scale_B 数组查尺度值。
    """
    source = cache["shared_source"].astype(np.int64)    # 0=T1, 1=T2
    indices = cache["shared_index"].astype(np.int64)    # 原始索引
    scales_a, scales_b = cache["scale_A"], cache["scale_B"]
    counts = Counter()
    for side, index in zip(source, indices):
        scale = int(scales_a[index] if side == 0 else scales_b[index])
        counts[scale] += 1
    return counts


def _config_from_args(args):
    """从命令行参数提取 SAM 生成器配置字典，存入缓存元数据。

    训练时 load 缓存可以还原当时的生成配置，方便复现实验。
    """
    return {
        "model_type": "vit_t",
        "checkpoint": str(args.checkpoint.resolve()),
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "duplicate_iou_thresh": args.duplicate_iou_thresh,
        "containment_thresh": args.containment_thresh,
        "containment_area_ratio": args.containment_area_ratio,
        "fine_threshold": args.fine_threshold,
        "mid_threshold": args.mid_threshold,
    }


def main(argv=None):
    """主流程: 遍历配对图像 → SAM 生成 → 去重 → 缓存 → 统计报告。"""
    args = build_parser().parse_args(argv)
    _validate_args(args)

    # 配对 A/B 图像
    pairs = pair_image_paths(args.dir_a, args.dir_b, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    error_log = args.error_log or args.output / "generation_errors.jsonl"

    # 固定随机种子 (SAM 内部也有随机性)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 延迟导入: MobileSAM 很重，只在离线生成工具里 import
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    # 加载 ViT-T 版本的 MobileSAM
    model = sam_model_registry["vit_t"](checkpoint=str(args.checkpoint))
    model.to(device=args.device)
    model.eval()

    # 构建 automatic mask generator (SAM 的自动分割模式)
    generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )
    generator_config = _config_from_args(args)

    # 统计计数器
    summary = Counter(total=len(pairs))
    elapsed_times, cache_sizes = [], []

    for position, (path_a, path_b) in enumerate(pairs, start=1):
        sample_id = path_a.stem                                    # 文件名 (无后缀) 作为样本 ID
        cache_path = args.output / (sample_id + ".npz")

        # 已存在且不覆盖 → 跳过 (断点续跑)
        if cache_path.exists() and not args.overwrite:
            summary["skipped"] += 1
            print("[{}/{}] skip {}".format(position, len(pairs), sample_id), flush=True)
            continue

        started = time.perf_counter()
        try:
            # 读图
            image_a, image_b = read_rgb(path_a), read_rgb(path_b)
            if image_a.shape != image_b.shape:
                raise ValueError("A/B shape mismatch: {} vs {}".format(image_a.shape, image_b.shape))

            # 分别跑 SAM 自动分割 (两时相独立生成 mask)
            records_a = generator.generate(image_a)
            records_b = generator.generate(image_b)

            # 保存为 v2 格式缓存 (内部会合并去重 + 按面积分尺度)
            save_sam_cache_v2(
                cache_path,
                sample_id,
                image_a.shape[:2],
                records_a,
                records_b,
                generator_config,
                duplicate_iou_thresh=args.duplicate_iou_thresh,
                containment_thresh=args.containment_thresh,
                containment_area_ratio=args.containment_area_ratio,
                fine_threshold=args.fine_threshold,
                mid_threshold=args.mid_threshold,
            )

            # 回读验证 + 统计
            cache = load_sam_cache_v2(cache_path)
            duration = time.perf_counter() - started
            scale_counts = _scale_counts(cache)
            shared_count = len(cache["shared_masks"])
            summary.update({
                "success": 1,
                "masks_a": len(records_a),                 # T1 原始 mask 数
                "masks_b": len(records_b),                 # T2 原始 mask 数
                "union": len(records_a) + len(records_b),  # 合并后总数
                "deduplicated": shared_count,               # 去重后共享 mask 数
                "fine": scale_counts[SCALE_FINE],           # 小尺度 mask 数
                "mid": scale_counts[SCALE_MID],             # 中尺度 mask 数
                "coarse": scale_counts[SCALE_COARSE],       # 大尺度 mask 数
            })
            elapsed_times.append(duration)
            cache_sizes.append(cache_path.stat().st_size)

            print(
                "[{}/{}] {} A={} B={} shared={} scales={}/{}/{} {:.2f}s {:.1f}KiB".format(
                    position, len(pairs), sample_id, len(records_a), len(records_b),
                    shared_count, scale_counts[SCALE_FINE], scale_counts[SCALE_MID],
                    scale_counts[SCALE_COARSE], duration, cache_sizes[-1] / 1024.0
                ),
                flush=True,
            )
        except Exception as error:
            # 单样本失败不中断整体流程，记录后继续
            summary["failure"] += 1
            record = {
                "sample_id": sample_id,
                "path_a": str(path_a),
                "path_b": str(path_b),
                "error": "{}: {}".format(type(error).__name__, error),
                "traceback": traceback.format_exc(),
            }
            _append_error(error_log, record)
            print("[{}/{}] FAILED {}: {}".format(position, len(pairs), sample_id, error), flush=True)

    # 汇总统计报告
    successes = max(summary["success"], 1)
    report = dict(summary)
    report.update({
        "mean_masks_a": summary["masks_a"] / float(successes),
        "mean_masks_b": summary["masks_b"] / float(successes),
        "mean_union": summary["union"] / float(successes),
        "mean_deduplicated": summary["deduplicated"] / float(successes),
        "mean_seconds": float(np.mean(elapsed_times)) if elapsed_times else 0.0,
        "cache_bytes": int(sum(cache_sizes)),
        "mean_cache_bytes": float(np.mean(cache_sizes)) if cache_sizes else 0.0,
        "output": str(args.output.resolve()),
        "error_log": str(error_log.resolve()),
    })
    print("SUMMARY " + json.dumps(report, sort_keys=True), flush=True)

    # 有失败返回 1，全部成功返回 0
    return 1 if summary["failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
