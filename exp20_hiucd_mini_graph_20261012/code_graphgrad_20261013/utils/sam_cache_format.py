"""
sam_cache_format —— SAM 超像素 mask 缓存的读写与格式定义

缓存格式版本: CACHE_VERSION = 2 (.npz 压缩存储)

为什么需要这个文件?
    MobileSAM 自动生成超像素 mask 非常耗时 (每对图几秒到几十秒)。
    训练时 Graph 分支要对每个样本都用同一组 mask，所以离线跑一次 SAM，
    把结果打包成 .npz，训练时直接 load，速度提升数量级。

与 sam_node_audit.py 的关系:
    generate_sam_cache.py (离线) → 本文件 save_sam_cache_v2 → .npz
    train.py (训练)             → 本文件 load_sam_cache_v2 → 得到 shared_masks
    sam_node_audit.py           → 用 shared_masks + GT 做节点状态审计

npz 内的字段 (CACHE_VERSION=2):
    format_version    — int64, 版本号 (=2)
    sample_id         — str, 样本 ID (文件名 stem)
    image_shape       — [H, W] int32, 图像原始分辨率
    mask_width        — int32, mask 宽度 (unpack 时需要)
    generator_config  — str (JSON), SAM 生成器配置快照
    scale_thresholds  — [fine, mid] float32, 面积占比分尺度阈值

    === T1 时相原始 mask ===
    packed_A          — [N, H, ceil(W/8)] uint8, packbits 压缩的 bool mask
    area_A            — [N] int32, 每个 mask 的像素面积
    bbox_A            — [N, 4] float32, (x, y, w, h)
    iou_A             — [N] float32, SAM 预测的 mask IoU
    stability_A       — [N] float32, SAM 的稳定性分数
    scale_A           — [N] uint8, 0=FINE / 1=MID / 2=COARSE

    === T2 时相原始 mask ===
    packed_B / area_B / bbox_B / iou_B / stability_B / scale_B  (同上)

    === 去重后的共享 mask (两时合并去重) ===
    shared_source     — [S] uint8, 0=来自 T1(A), 1=来自 T2(B)
    shared_index      — [S] int32, 在原始 mask 数组中的索引

面积分尺度逻辑 (_record_arrays):
    ratio = area / (H * W)
    ratio < fine_threshold  → SCALE_FINE  (0, 小目标)
    ratio ≤ mid_threshold   → SCALE_MID   (1, 中尺度)
    其余                    → SCALE_COARSE (2, 大区域)
"""

import json
import os
from pathlib import Path

import numpy as np


CACHE_VERSION = 2

# 面积分尺度常量
#   SCALE_FINE  (0): 小目标，面积占比 < fine_threshold
#   SCALE_MID   (1): 中尺度，fine_threshold ≤ 面积占比 < mid_threshold
#   SCALE_COARSE(2): 大区域，面积占比 ≥ mid_threshold
SCALE_FINE, SCALE_MID, SCALE_COARSE = 0, 1, 2


def pack_masks(masks):
    """把 bool mask 数组做 packbits 位压缩。

    SAM 输出的 mask 是 bool 类型 [N, H, W]，每个像素 1 bit 就够了。
    np.packbits 把连续 8 个 bool 打包成 1 个 uint8，宽度维度从 W 缩到 ceil(W/8)，
    磁盘占用减少 ~8 倍。训练时再用 unpack_masks 还原。

    Args:
        masks: [N, H, W] bool

    Returns:
        [N, H, ceil(W/8)] uint8, bitorder="little" (低位对应最左像素)
    """
    masks = np.asarray(masks, dtype=np.bool_)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N,H,W]")
    return np.packbits(masks, axis=-1, bitorder="little")


def unpack_masks(packed, width):
    """把 packbits 压缩的 mask 还原成 bool 数组。

    Args:
        packed: [N, H, ceil(W/8)] uint8, packbits 压缩的 mask
        width:  int, 原始图像宽度 (packbits 向上取整可能多几个 bit 要截掉)

    Returns:
        [N, H, W] bool
    """
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 3:
        raise ValueError("packed masks must have shape [N,H,ceil(W/8)]")
    # unpackbits → [N, H, ceil(W/8)*8] bool，再截到原始宽度
    return np.unpackbits(packed, axis=-1, bitorder="little")[..., :int(width)].astype(np.bool_)


def _mask_bbox(mask):
    """计算单个 mask 的轴对齐包围盒 [x, y, w, h] (左上角坐标 + 宽高)。

    Args:
        mask: [H, W] bool

    Returns:
        [x, y, w, h] float32；空 mask 返回全零
    """
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return [0.0, 0.0, 0.0, 0.0]
    y0, x0 = coordinates.min(0); y1, x1 = coordinates.max(0)
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def _normalized_record(record):
    """把一条 SAM mask record 规范化成统一格式。

    SAM generator 返回的 record 是 dict，字段可能不全（比如 bbox 有时缺失），
    这里做一次兜底: 缺 bbox 就自己算，缺 area 就用 segmentation.sum()。

    Args:
        record: SAM 原始 dict，至少含 "segmentation" [H,W] bool 和 "predicted_iou"/"stability_score"

    Returns:
        规范化 dict: {segmentation, area, bbox, predicted_iou, stability_score}
    """
    mask = np.asarray(record["segmentation"], dtype=np.bool_)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("SAM segmentation must be a non-empty [H,W] mask")
    return {
        "segmentation": mask,
        "area": int(record.get("area", mask.sum())),          # 缺 area 就从 mask 算
        "bbox": np.asarray(record.get("bbox", _mask_bbox(mask)), dtype=np.float32),
        "predicted_iou": float(record["predicted_iou"]),
        "stability_score": float(record["stability_score"]),
    }


def deduplicate_records(records, duplicate_iou_thresh=.90,
                        containment_thresh=.95, containment_area_ratio=.80):
    """合并去重 T1/T2 两时相的 SAM mask，保留高质量的那一批。

    为什么需要去重?
        同一物体在 T1 和 T2 可能都被 SAM 检测到，生成高度重叠甚至完全相同的 mask。
        如果不去重，Graph 分支会出现"两个几乎一样的超像素对应同一物体"，
        浪费计算还可能导致 GAT 传播时出现冗余自环。

    去重两阶段:
        1. 按质量排序: quality = pred_iou * stability_score，优先保留高质量 mask
        2. 对每个候选 mask，与已保留 mask 比较:
           a. IoU > duplicate_iou_thresh → 重复，丢弃
           b. 或者 containment (小的被大的包含比例) > containment_thresh
              且 小/大面积比 > containment_area_ratio → 视为重复 (包含关系)，丢弃
           c. 否则保留

    Args:
        records:  T1 + T2 合并的 SAM record 列表
        duplicate_iou_thresh:   IoU 重复阈值 (默认 0.90)
        containment_thresh:     containment 阈值 (默认 0.95)
        containment_area_ratio: containment 触发前的面积比下限 (默认 0.80)

    Returns:
        kept: list[int], 保留下来的 record 索引 (原始合并数组中的位置)
    """
    normalized = [_normalized_record(record) for record in records]

    # 按质量排序: quality = iou * stability，乘积高的先被考虑
    # 同质量时优先 area 小的 (更精确)，索引小的 (稳定)
    order = sorted(range(len(normalized)), key=lambda index: (
        -(normalized[index]["predicted_iou"] * normalized[index]["stability_score"]),
        -normalized[index]["predicted_iou"], -normalized[index]["stability_score"],
        normalized[index]["area"], index))

    kept = []
    for index in order:
        candidate = normalized[index]
        duplicate = False
        cx, cy, cw, ch = candidate["bbox"]

        for kept_index in kept:
            other = normalized[kept_index]
            ox, oy, ow, oh = other["bbox"]

            # 快速 bbox 排除: 包围盒都不相交 → 不可能重复，跳过
            if cx + cw <= ox or ox + ow <= cx or cy + ch <= oy or oy + oh <= cy:
                continue

            # 精确 mask 交并比
            intersection = int(np.logical_and(candidate["segmentation"], other["segmentation"]).sum())
            if intersection == 0:
                continue

            smaller = min(candidate["area"], other["area"])
            larger = max(candidate["area"], other["area"])
            union = candidate["area"] + other["area"] - intersection
            iou = intersection / float(max(union, 1))
            containment = intersection / float(max(smaller, 1))

            # 规则 a: IoU 过高 → 重复
            # 规则 b: 包含关系 (大的包住小的) 且面积比接近 → 重复
            if iou > duplicate_iou_thresh or (
                    containment > containment_thresh
                    and smaller / float(max(larger, 1)) > containment_area_ratio):
                duplicate = True
                break

        if not duplicate:
            kept.append(index)

    return kept


def _record_arrays(records, image_shape, fine_threshold, mid_threshold):
    """把归一化的 SAM records 打包成多个 numpy 数组，并按面积分尺度。

    Args:
        records:      已规范化的 SAM record 列表
        image_shape:  (H, W) 图像分辨率
        fine_threshold: 面积占比上限 → FINE
        mid_threshold:  面积占比上限 → MID

    Returns:
        dict: {packed, area, bbox, iou, stability, scale}
              - packed: [N, H, ceil(W/8)] uint8, packbits 压缩 mask
              - area:   [N] int32
              - bbox:   [N, 4] float32
              - iou:    [N] float32
              - stability: [N] float32
              - scale:  [N] uint8 (0=FINE, 1=MID, 2=COARSE)
    """
    height, width = image_shape
    normalized = [_normalized_record(record) for record in records]

    # mask 堆起来 (空列表时构造零形状数组)
    masks = np.stack([record["segmentation"] for record in normalized]) if normalized else np.empty((0, height, width), bool)

    area = np.asarray([record["area"] for record in normalized], dtype=np.int32)
    ratios = area.astype(np.float32) / float(height * width)

    # 按面积占比分三档
    #   ratio < fine_threshold  → FINE (小目标)
    #   ratio ≤ mid_threshold   → MID  (中尺度)
    #   其余                    → COARSE (大区域)
    scale = np.where(ratios < fine_threshold, SCALE_FINE,
                     np.where(ratios <= mid_threshold, SCALE_MID, SCALE_COARSE)).astype(np.uint8)

    return {
        "packed": pack_masks(masks), "area": area,
        "bbox": np.asarray([record["bbox"] for record in normalized], dtype=np.float32).reshape(-1, 4),
        "iou": np.asarray([record["predicted_iou"] for record in normalized], dtype=np.float32),
        "stability": np.asarray([record["stability_score"] for record in normalized], dtype=np.float32),
        "scale": scale,
    }


def save_sam_cache_v2(path, sample_id, image_shape, records_a, records_b,
                      generator_config, duplicate_iou_thresh=.90,
                      containment_thresh=.95, containment_area_ratio=.80,
                      fine_threshold=.01, mid_threshold=.05):
    """保存 SAM mask 缓存为 v2 格式 .npz。

    完整流程:
        1. 分别打包 T1/T2 的原始 mask (packbits 压缩 + 分尺度)
        2. 合并 T1+T2 做去重，得到共享 mask 索引
        3. 记录每个共享 mask 来自哪一边 + 原始索引
        4. 写入临时文件再原子 rename (避免中途崩溃留下半成品)

    Args:
        path:     输出 .npz 路径
        sample_id: 样本 ID (文件名 stem)
        image_shape: (H, W)
        records_a: T1 时相 SAM 原始 records (list[dict])
        records_b: T2 时相 SAM 原始 records
        generator_config: SAM 生成器配置 (dict)，存进元数据方便复现
        duplicate_iou_thresh / containment_thresh / containment_area_ratio: 去重参数
        fine_threshold / mid_threshold: 面积分尺度阈值
    """
    path = Path(path); height, width = map(int, image_shape)

    # 分别打包 T1 / T2 原始 mask
    arrays_a = _record_arrays(records_a, (height, width), fine_threshold, mid_threshold)
    arrays_b = _record_arrays(records_b, (height, width), fine_threshold, mid_threshold)

    # 合并去重: T1+T2 放一起，保留高质量且不重复的
    combined = list(records_a) + list(records_b)
    keep = deduplicate_records(combined, duplicate_iou_thresh, containment_thresh, containment_area_ratio)

    # keep 中的索引是合并数组里的位置: 0..len(records_a)-1 是 T1, 其余是 T2
    # shared_source: 0=来自 T1(A), 1=来自 T2(B)
    # shared_index: 在原始 T1 或 T2 数组中的索引
    source = np.asarray([0 if index < len(records_a) else 1 for index in keep], dtype=np.uint8)
    source_index = np.asarray([index if index < len(records_a) else index - len(records_a) for index in keep], dtype=np.int32)

    # 构造 npz payload
    payload = {
        "format_version": np.asarray(CACHE_VERSION, dtype=np.int64),
        "sample_id": np.asarray(str(sample_id)), "image_shape": np.asarray([height, width], dtype=np.int32),
        "mask_width": np.asarray(width, dtype=np.int32),
        "shared_source": source, "shared_index": source_index,
        "generator_config": np.asarray(json.dumps(generator_config, sort_keys=True)),
        "scale_thresholds": np.asarray([fine_threshold, mid_threshold], dtype=np.float32),
    }
    # 把 T1/A 和 T2/B 的各个数组分别加进去 (加后缀 A/B 区分)
    for suffix, arrays in (("A", arrays_a), ("B", arrays_b)):
        for name, value in arrays.items(): payload[name + "_" + suffix] = value

    path.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入: 先写 .tmp.npz，再 os.replace (跨平台原子 rename)
    # 避免写一半崩溃导致正式文件损坏
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(str(temporary), **payload)
    os.replace(str(temporary), str(path))


def load_sam_cache_v2(path):
    """加载 v2 格式 SAM mask 缓存，返回展开后的完整字典。

    返回字典的额外字段 (训练时 Graph 分支直接用):
        masks_A / masks_B       — [N, H, W] bool, 原始 T1/T2 mask (unpack 还原)
        shared_masks            — [S, H, W] bool, 去重后的共享 mask
        sample_id               — str
        generator_config        — dict (从 JSON 反序列化)

    Args:
        path: .npz 缓存文件路径

    Returns:
        dict, 包含所有原始字段 + 展开后的 masks
    """
    with np.load(str(path), allow_pickle=False) as payload:
        # 版本校验
        if int(np.asarray(payload["format_version"]).reshape(-1)[0]) != CACHE_VERSION:
            raise ValueError("not a SAM cache v2 file")

        width = int(payload["mask_width"])

        # 还原 packbits 压缩的 mask
        masks_a = unpack_masks(payload["packed_A"], width)
        masks_b = unpack_masks(payload["packed_B"], width)

        # 根据 shared_source + shared_index 组装共享 mask
        source = payload["shared_source"].astype(np.uint8)
        indices = payload["shared_index"].astype(np.int64)
        shared = [masks_a[index] if side == 0 else masks_b[index] for side, index in zip(source, indices)]

        height = int(payload["image_shape"][0])
        result = {key: np.asarray(payload[key]).copy() for key in payload.files}

    result.update({
        "sample_id": str(result["sample_id"]),
        "masks_A": masks_a,
        "masks_B": masks_b,
        "shared_masks": np.stack(shared) if shared else np.empty((0, height, width), bool),
        "generator_config": json.loads(str(result["generator_config"])),
    })
    return result
