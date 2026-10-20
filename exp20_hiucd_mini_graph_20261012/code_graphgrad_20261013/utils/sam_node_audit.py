"""
sam_node_audit —— SAM 超像素节点状态审计工具

功能:
    给定一组共享的 SAM mask（同一组 mask 覆盖 T1、T2 两个时相）和 GT 标签，
    为每个超像素节点判定三种状态之一:
      IGNORE     —— 丢弃（面积过小 或 变化像素太少，不参与图构建）
      GT         —— 高质量标注节点（变化语义单一且纯度足够高）
      UNLABELED  —— 未标注节点（稳定区域，可作为 GAT 传播的目标）

核心函数:
    audit_sam_nodes()           —— 入口: 对一批共享 mask 做完整审计
    _temporal_assignments()     —— 单时相节点状态/类别分配
    _validate_inputs()          —— 输入形状和值域校验
    summarize_sam_node_audits() —— 聚合多批审计结果，输出统计摘要
"""

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

# ---------------------------------------------------------------------------
# 节点状态常量
# ---------------------------------------------------------------------------
IGNORE = 0       # 忽略: 面积太小 / 变化像素太少 / 纯度不够，不参与图构建
GT = 1           # 标注: 高质量变化区域，纯度 > tau_purity
UNLABELED = 2    # 未标注: 稳定区域 (变化比例低)，GAT 传播目标


@dataclass(frozen=True)
class SamNodeAuditConfig:
    """节点审计超参数配置（frozen 不可变，避免运行时被误改）。"""

    tau_gt: float = 0.8        # GT 判定阈值: change_ratio > tau_gt 才考虑 GT
    tau_purity: float = 0.9    # 纯度阈值: 变化像素中主类占比 > tau_purity 才算 GT
    tau_u: float = 0.95        # 未标注判定阈值: 未变化比例 > tau_u → UNLABELED
    min_mask_pixels: int = 4   # mask 最小像素数: 小于此数直接 IGNORE
    min_changed_pixels: int = 4  # 变化像素最小数量: 小于此数不算 GT 节点

    def __post_init__(self):
        # 运行时校验阈值合法性
        for name in ("tau_gt", "tau_purity", "tau_u"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))
        if self.min_mask_pixels < 0 or self.min_changed_pixels < 0:
            raise ValueError("minimum pixel counts must be non-negative")


@dataclass
class SamNodeAudit:
    """单个样本的 SAM 节点审计结果。

    同一组共享 mask 在 T1、T2 两个时相分别得到独立的状态/类别，
    因为同一超像素在两个时相的 GT 标签分布可能不同。
    """

    states_a: torch.Tensor      # [N] T1 时相每个节点的状态 (IGNORE/GT/UNLABELED)
    states_b: torch.Tensor      # [N] T2 时相每个节点的状态
    classes_a: torch.Tensor     # [N] T1 时相每个节点的类别 (仅 GT 节点有值，其余为 0)
    classes_b: torch.Tensor     # [N] T2 时相每个节点的类别
    change_ratio: torch.Tensor  # [N] 每个 mask 内"变化像素"占总像素的比例
    purity_a: torch.Tensor     # [N] T1 时相 GT 节点的主类纯度 (仅 GT 有值)
    purity_b: torch.Tensor     # [N] T2 时相 GT 节点的主类纯度
    changed_pixels: torch.Tensor # [N] 每个 mask 内变化像素的绝对数量
    total_changed_pixels: int   # 整张图 GT 变化像素总数
    covered_changed_pixels: int # 被 SAM mask 覆盖的变化像素数 (任意 mask)
    gt_covered_changed_pixels: int # 被 GT 节点覆盖的变化像素数


def _validate_inputs(shared_masks, labels_a, labels_b, num_classes):
    """校验输入的形状和值域，提前报错避免后续静默出错。

    Args:
        shared_masks: [N, H, W] bool, 共享的 SAM mask
        labels_a/b:  [H, W] long, T1/T2 时相的 GT 标签 (class 0 表示背景/未变化)
        num_classes: int, 总类别数 (含 class 0)

    Returns:
        (masks, labels_a, labels_b) 校验后转到 CPU 的 tensor
    """
    masks = torch.as_tensor(shared_masks, dtype=torch.bool, device="cpu")
    labels_a = torch.as_tensor(labels_a, dtype=torch.long, device="cpu")
    labels_b = torch.as_tensor(labels_b, dtype=torch.long, device="cpu")
    if masks.ndim != 3:
        raise ValueError("shared_masks must have shape [N, H, W]")
    if labels_a.ndim != 2 or labels_b.shape != labels_a.shape:
        raise ValueError("labels must have matching [H, W] shapes")
    if tuple(masks.shape[1:]) != tuple(labels_a.shape):
        raise ValueError("shared_masks and labels must share one spatial shape")
    if num_classes < 2:
        raise ValueError("num_classes must include class 0 and at least one semantic class")
    for name, labels in (("label_A", labels_a), ("label_B", labels_b)):
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
            raise ValueError("{} contains a label outside [0, {})".format(name, num_classes))
    return masks, labels_a, labels_b


def _temporal_assignments(
    masks,
    labels,
    changed,
    mask_areas,
    changed_pixels,
    change_ratio,
    num_classes,
    config,
):
    """对单个时相执行节点状态和类别分配。

    判定逻辑 (按优先级):
      1. mask 面积 < min_mask_pixels → IGNORE
      2. 未变化比例 (1 - change_ratio) > tau_u → UNLABELED
         (即: 这个区域里大部分像素没变，当成稳定的未标注区域)
      3. change_ratio > tau_gt 且 变化像素数 > min_changed_pixels:
         → 统计该 mask 内变化像素的主类纯度
         → 纯度 > tau_purity → GT，记录类别 ID
         → 纯度不够 → UNLABELED (变化语义不单一，不放心当 GT)

    Args:
        masks:        [N, H, W] bool, 共享的 SAM mask
        labels:       [H, W] long, 当前时相的 GT 标签
        changed:      [H, W] bool, 整图哪些像素发生了变化
        mask_areas:   [N] long, 每个 mask 的像素面积
        changed_pixels: [N] long, 每个 mask 内变化像素数
        change_ratio: [N] float, changed_pixels / mask_areas
        num_classes:  int, 总类别数
        config:       SamNodeAuditConfig, 阈值配置

    Returns:
        states:  [N] long, 每个节点的状态 (IGNORE/GT/UNLABELED)
        classes: [N] long, GT 节点的类别 (UNLABELED/IGNORE 为 0)
        purity:  [N] float, GT 节点的纯度 (UNLABELED/IGNORE 为 0)
    """
    count = masks.shape[0]
    states = torch.full((count,), IGNORE, dtype=torch.long)   # 默认全 IGNORE
    classes = torch.zeros((count,), dtype=torch.long)
    purity = torch.zeros((count,), dtype=torch.float32)

    # 未变化比例 > tau_u 的 mask → UNLABELED (稳定区域，GAT 传播目标)
    unchanged_ratio = 1.0 - change_ratio
    unlabeled = (
        (mask_areas >= config.min_mask_pixels)   # 面积够大
        & (unchanged_ratio > config.tau_u)       # 绝大部分像素没变化
    )
    states[unlabeled] = UNLABELED

    # 剩下的 mask 里挑 GT 节点
    for index in range(count):
        if mask_areas[index] < config.min_mask_pixels:
            continue                             # 面积太小，保持 IGNORE
        if not (change_ratio[index] > config.tau_gt):
            continue                             # 变化比例不够，保持之前的 UNLABELED 或 IGNORE
        if not (changed_pixels[index] > config.min_changed_pixels):
            continue                             # 变化像素太少，不放心

        # 计算该 mask 内变化像素的主类纯度
        support = masks[index] & changed          # mask 内 且 发生了变化的像素
        class_counts = torch.stack([
            (support & (labels == class_id)).sum()   # 变化像素中每个语义类的数量
            for class_id in range(1, num_classes)      # class 0 是背景，跳过
        ])
        best_count, best_offset = class_counts.max(dim=0)
        best_purity = best_count.float() / changed_pixels[index].float().clamp_min(1.0)
        purity[index] = best_purity

        if best_purity > config.tau_purity:
            # 纯度够高 → GT 节点，best_offset 是 class_id-1 (因为 range(1,num_classes))
            states[index] = GT
            classes[index] = int(best_offset) + 1

    return states, classes, purity


def audit_sam_nodes(
    shared_masks,
    labels_a,
    labels_b,
    num_classes,
    config=SamNodeAuditConfig(),
):
    """入口函数: 对一组共享 SAM mask 做完整的双时相节点审计。

    同一组 mask 同时覆盖 T1 和 T2，所以先算 mask 级的公共统计量
    (面积、变化像素数、变化比例)，再分别对 T1/T2 标签分配状态。

    Args:
        shared_masks: [N, H, W] bool, 共享的 SAM mask
        labels_a:     [H, W] long, T1 时相 GT
        labels_b:     [H, W] long, T2 时相 GT
        num_classes:  int, 总类别数 (含 class 0 背景)
        config:       SamNodeAuditConfig, 阈值配置

    Returns:
        SamNodeAudit, 包含双时相的状态、类别、纯度及覆盖率统计
    """
    masks, labels_a, labels_b = _validate_inputs(
        shared_masks, labels_a, labels_b, int(num_classes))

    # 整图哪些像素发生了变化 (GT 有标注就算变化)
    changed = (labels_a != 0) | (labels_b != 0)

    # 每个 mask 的公共统计量 (与时相无关)
    mask_areas = masks.flatten(1).sum(1).long()                              # [N] 面积
    changed_pixels = (masks & changed.unsqueeze(0)).flatten(1).sum(1).long()  # [N] 变化像素数
    change_ratio = changed_pixels.float() / mask_areas.float().clamp_min(1.0)  # [N] 变化比例

    # 分别对 T1/T2 标签分配节点状态和类别
    states_a, classes_a, purity_a = _temporal_assignments(
        masks, labels_a, changed, mask_areas, changed_pixels, change_ratio,
        int(num_classes), config)
    states_b, classes_b, purity_b = _temporal_assignments(
        masks, labels_b, changed, mask_areas, changed_pixels, change_ratio,
        int(num_classes), config)

    # 覆盖率统计
    any_coverage = masks.any(dim=0) if masks.shape[0] else torch.zeros_like(changed)
    gt_masks = (states_a == GT) | (states_b == GT)
    gt_coverage = masks[gt_masks].any(dim=0) if gt_masks.any() else torch.zeros_like(changed)

    return SamNodeAudit(
        states_a=states_a,
        states_b=states_b,
        classes_a=classes_a,
        classes_b=classes_b,
        change_ratio=change_ratio,
        purity_a=purity_a,
        purity_b=purity_b,
        changed_pixels=changed_pixels,
        total_changed_pixels=int(changed.sum()),
        covered_changed_pixels=int((changed & any_coverage).sum()),
        gt_covered_changed_pixels=int((changed & gt_coverage).sum()),
    )


def summarize_sam_node_audits(
    audits: Iterable[SamNodeAudit], num_classes: int
) -> Dict[str, object]:
    """聚合多批 SamNodeAudit 结果，输出人类可读的统计摘要。

    常用于训练脚本的日志打印，快速检查:
      - 有多少 GT / UNLABELED / IGNORE 节点
      - GT 节点按类别分布是否均衡
      - 有多少样本完全没有 GT 节点 (原型 bank 可能初始化不了)
      - SAM mask 对变化像素的覆盖率如何

    Args:
        audits:     SamNodeAudit 实例的可迭代对象 (通常是一个 batch 的结果)
        num_classes: int, 总类别数

    Returns:
        dict, 包含各类计数值和覆盖率指标
    """
    audits = list(audits)
    class_counts = torch.zeros(int(num_classes), dtype=torch.long)
    state_counts = {"gt": 0, "unlabeled": 0, "ignore": 0}
    total_masks = 0
    no_gt_images = 0
    total_changed = 0
    covered_changed = 0
    gt_covered_changed = 0

    for audit in audits:
        total_masks += int(audit.states_a.numel())

        # 统计双时相节点的状态分布
        states = torch.cat([audit.states_a, audit.states_b])
        classes = torch.cat([audit.classes_a, audit.classes_b])

        state_counts["gt"] += int((states == GT).sum())
        state_counts["unlabeled"] += int((states == UNLABELED).sum())
        state_counts["ignore"] += int((states == IGNORE).sum())

        if not (states == GT).any():
            no_gt_images += 1   # 这张图完全没有 GT 节点

        # GT 节点的类别直方图
        gt_classes = classes[states == GT]
        if gt_classes.numel():
            class_counts += torch.bincount(gt_classes, minlength=int(num_classes))

        total_changed += audit.total_changed_pixels
        covered_changed += audit.covered_changed_pixels
        gt_covered_changed += audit.gt_covered_changed_pixels

    # 正负样本对数 (hard negative mining 用得到)
    # 正样本对: 同类 GT 节点两两配对 C(n,2)
    positive_pairs = int(sum(int(count) * (int(count) - 1) // 2 for count in class_counts))
    total_gt = int(class_counts.sum())
    all_pairs = total_gt * (total_gt - 1) // 2

    denominator = float(max(total_changed, 1))
    return {
        "total_shared_masks": total_masks,
        "gt_nodes": state_counts["gt"],
        "unlabeled_nodes": state_counts["unlabeled"],
        "ignore_nodes": state_counts["ignore"],
        "gt_nodes_per_class": class_counts.tolist(),
        "images_without_gt_nodes": no_gt_images,
        "positive_gt_pairs": positive_pairs,
        "negative_gt_pairs": all_pairs - positive_pairs,
        "changed_coverage": covered_changed / denominator if total_changed else 0.0,
        "reliable_gt_changed_coverage": gt_covered_changed / denominator if total_changed else 0.0,
    }
