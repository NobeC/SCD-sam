from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.sam_node_audit import UNLABELED


@dataclass
class GraphPixelPrediction:
    pred: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


@dataclass
class GraphVetoResult:
    target: torch.Tensor
    valid: torch.Tensor
    teacher_candidate_count: int
    veto_count: int
    fallback_count: int
    accepted_count: int
    class_histogram: torch.Tensor


@dataclass
class GraphPixelBatch:
    pred_a: torch.Tensor
    conf_a: torch.Tensor
    valid_a: torch.Tensor
    pred_b: torch.Tensor
    conf_b: torch.Tensor
    valid_b: torch.Tensor


@torch.no_grad()
def hard_reproject_u_nodes(
    masks,
    states,
    node_pred,
    node_conf,
    node_valid,
):
    masks = torch.as_tensor(masks, dtype=torch.bool)
    device = node_pred.device if torch.is_tensor(node_pred) else masks.device
    masks = masks.to(device=device)
    states = torch.as_tensor(states, device=device, dtype=torch.long)
    node_pred = torch.as_tensor(node_pred, device=device, dtype=torch.long)
    node_conf = torch.as_tensor(node_conf, device=device, dtype=torch.float32)
    node_valid = torch.as_tensor(node_valid, device=device, dtype=torch.bool)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    count, height, width = masks.shape
    tensors = (states, node_pred, node_conf, node_valid)
    if any(tensor.shape != (count,) for tensor in tensors):
        raise ValueError("node reprojection tensors must have shape [N]")
    if not torch.isfinite(node_conf).all():
        raise FloatingPointError("graph node confidence contains NaN or Inf")

    pred = torch.zeros((height, width), device=device, dtype=torch.long)
    confidence = torch.zeros((height, width), device=device, dtype=torch.float32)
    valid = torch.zeros((height, width), device=device, dtype=torch.bool)
    owner_area = torch.full(
        (height, width), torch.iinfo(torch.long).max, device=device, dtype=torch.long)
    owner_index = torch.full(
        (height, width), torch.iinfo(torch.long).max, device=device, dtype=torch.long)
    for index in range(count):
        if (int(states[index]) != UNLABELED or not bool(node_valid[index])
                or int(node_pred[index]) <= 0):
            continue
        mask = masks[index]
        if not mask.any():
            continue
        candidate_conf = node_conf[index]
        candidate_area = mask.sum().long()
        wins = mask & (
            (~valid)
            | (candidate_conf > confidence)
            | ((candidate_conf == confidence) & (candidate_area < owner_area))
            | ((candidate_conf == confidence) & (candidate_area == owner_area)
               & (index < owner_index))
        )
        if wins.any():
            pred[wins] = node_pred[index]
            confidence[wins] = candidate_conf
            valid[wins] = True
            owner_area[wins] = candidate_area
            owner_index[wins] = index
    return GraphPixelPrediction(pred=pred, confidence=confidence, valid=valid)


@torch.no_grad()
def reproject_stage2_batch(stage2_result, audits, masks):
    if len(audits) != len(masks) or len(audits) != len(stage2_result.predictions_a):
        raise ValueError("Stage2 predictions, audits, and mask batches must align")
    projected_a, projected_b = [], []
    for index, (audit, sample_masks) in enumerate(zip(audits, masks)):
        projected_a.append(hard_reproject_u_nodes(
            sample_masks, audit.states_a,
            stage2_result.predictions_a[index],
            stage2_result.confidences_a[index],
            stage2_result.valid_a[index]))
        projected_b.append(hard_reproject_u_nodes(
            sample_masks, audit.states_b,
            stage2_result.predictions_b[index],
            stage2_result.confidences_b[index],
            stage2_result.valid_b[index]))
    return GraphPixelBatch(
        pred_a=torch.stack([item.pred for item in projected_a]),
        conf_a=torch.stack([item.confidence for item in projected_a]),
        valid_a=torch.stack([item.valid for item in projected_a]),
        pred_b=torch.stack([item.pred for item in projected_b]),
        conf_b=torch.stack([item.confidence for item in projected_b]),
        valid_b=torch.stack([item.valid for item in projected_b]),
    )


@torch.no_grad()
def teacher_graph_veto(
    teacher_probs_a,
    teacher_probs_b,
    teacher_change_logits,
    labels_a,
    labels_b,
    graph_pred_a,
    graph_conf_a,
    graph_valid_a,
    graph_pred_b,
    graph_conf_b,
    graph_valid_b,
    teacher_threshold,
    teacher_fallback_threshold,
    graph_confidence_threshold,
    change_threshold,
    num_classes,
    ignore_index=255,
):
    if not 0.0 <= teacher_threshold <= teacher_fallback_threshold <= 1.0:
        raise ValueError("teacher thresholds must satisfy 0 <= base <= fallback <= 1")
    if not 0.0 <= graph_confidence_threshold <= 1.0:
        raise ValueError("graph confidence threshold must be in [0, 1]")
    if not 0.0 <= change_threshold <= 1.0:
        raise ValueError("change threshold must be in [0, 1]")
    if teacher_probs_a.shape != teacher_probs_b.shape or teacher_probs_a.ndim != 4:
        raise ValueError("teacher probabilities must have matching [B, C, H, W] shapes")
    if teacher_probs_a.shape[1] != int(num_classes):
        raise ValueError("teacher class count does not match num_classes")
    device = teacher_probs_a.device
    labels_a = labels_a.to(device=device, dtype=torch.long)
    labels_b = labels_b.to(device=device, dtype=torch.long)
    graph_pred_a = graph_pred_a.to(device=device, dtype=torch.long)
    graph_pred_b = graph_pred_b.to(device=device, dtype=torch.long)
    graph_conf_a = graph_conf_a.to(device=device, dtype=torch.float32)
    graph_conf_b = graph_conf_b.to(device=device, dtype=torch.float32)
    graph_valid_a = graph_valid_a.to(device=device, dtype=torch.bool)
    graph_valid_b = graph_valid_b.to(device=device, dtype=torch.bool)
    spatial_shape = labels_a.shape
    graph_tensors = (
        graph_pred_a, graph_pred_b, graph_conf_a, graph_conf_b,
        graph_valid_a, graph_valid_b,
    )
    if labels_b.shape != spatial_shape or any(tensor.shape != spatial_shape for tensor in graph_tensors):
        raise ValueError("teacher, graph, and label spatial shapes must match")
    if teacher_change_logits.shape != (spatial_shape[0], 1, spatial_shape[1], spatial_shape[2]):
        raise ValueError("teacher change logits must have shape [B, 1, H, W]")
    finite_tensors = (teacher_probs_a, teacher_probs_b, teacher_change_logits,
                      graph_conf_a, graph_conf_b)
    if any(not torch.isfinite(tensor.float()).all() for tensor in finite_tensors):
        # Fail closed: invalid evidence must not terminate training or become a target.
        target = torch.full(
            spatial_shape, int(ignore_index), device=device, dtype=torch.long)
        return GraphVetoResult(
            target=target,
            valid=torch.zeros(spatial_shape, device=device, dtype=torch.bool),
            teacher_candidate_count=0,
            veto_count=0,
            fallback_count=0,
            accepted_count=0,
            class_histogram=torch.zeros(
                int(num_classes), device=device, dtype=torch.long),
        )

    teacher_conf_a, teacher_pred_a = teacher_probs_a.float().max(dim=1)
    teacher_conf_b, teacher_pred_b = teacher_probs_b.float().max(dim=1)
    changed = (labels_a > 0) | (labels_b > 0)
    teacher_unchanged = torch.sigmoid(teacher_change_logits.float()).squeeze(1) < change_threshold
    semantic = (teacher_pred_a > 0) & (teacher_pred_a < int(num_classes))
    teacher_candidate = (
        (~changed)
        & teacher_unchanged
        & semantic
        & (teacher_pred_a == teacher_pred_b)
        & (teacher_conf_a >= teacher_threshold)
        & (teacher_conf_b >= teacher_threshold)
    )
    effective_a = (
        graph_valid_a
        & (graph_conf_a >= graph_confidence_threshold)
        & (graph_pred_a > 0)
        & (graph_pred_a < int(num_classes))
    )
    effective_b = (
        graph_valid_b
        & (graph_conf_b >= graph_confidence_threshold)
        & (graph_pred_b > 0)
        & (graph_pred_b < int(num_classes))
    )
    conflict = (
        (effective_a & (graph_pred_a != teacher_pred_a))
        | (effective_b & (graph_pred_b != teacher_pred_a))
    )
    has_graph = effective_a | effective_b
    accepted_with_graph = teacher_candidate & has_graph & (~conflict)
    fallback = (
        teacher_candidate
        & (~has_graph)
        & (teacher_conf_a >= teacher_fallback_threshold)
        & (teacher_conf_b >= teacher_fallback_threshold)
    )
    valid = accepted_with_graph | fallback
    if bool((valid & changed).any()):
        raise AssertionError("graph pseudo labels leaked into changed pixels")
    target = torch.full(spatial_shape, int(ignore_index), device=device, dtype=torch.long)
    target[valid] = teacher_pred_a[valid]
    if bool(((target[valid] <= 0) | (target[valid] >= int(num_classes))).any()):
        raise AssertionError("graph pseudo target contains an invalid semantic class")
    histogram = torch.bincount(
        target[valid], minlength=int(num_classes)) if valid.any() else torch.zeros(
            int(num_classes), device=device, dtype=torch.long)
    histogram[0] = 0
    return GraphVetoResult(
        target=target.detach(),
        valid=valid.detach(),
        teacher_candidate_count=int(teacher_candidate.sum()),
        veto_count=int((teacher_candidate & conflict).sum()),
        fallback_count=int(fallback.sum()),
        accepted_count=int(valid.sum()),
        class_histogram=histogram.detach(),
    )


def graph_pixel_pseudo_loss(
    logits_a,
    logits_b,
    target,
    valid,
    ignore_index=255,
):
    if logits_a.shape != logits_b.shape or logits_a.ndim != 4:
        raise ValueError("student logits must have matching [B, C, H, W] shapes")
    target = target.to(device=logits_a.device, dtype=torch.long).detach()
    valid = valid.to(device=logits_a.device, dtype=torch.bool).detach()
    if target.shape != logits_a.shape[:1] + logits_a.shape[2:] or valid.shape != target.shape:
        raise ValueError("pseudo target and valid mask shapes do not match logits")
    if not valid.any():
        return (logits_a.sum() + logits_b.sum()) * 0.0
    supervised_target = torch.full_like(target, int(ignore_index))
    supervised_target[valid] = target[valid]
    return 0.5 * (
        F.cross_entropy(logits_a.float(), supervised_target, ignore_index=int(ignore_index))
        + F.cross_entropy(logits_b.float(), supervised_target, ignore_index=int(ignore_index))
    )
