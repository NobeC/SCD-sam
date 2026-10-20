"""Helpers for combining semantic and graph pseudo-label coverage."""

import torch


def complement_graph_targets(
    graph_targets,
    graph_valid,
    semantic_targets,
    ignore_index=255,
):
    """Keep graph labels only where semantic pseudo-labels are absent."""
    if graph_targets.shape != graph_valid.shape:
        raise ValueError("graph targets and validity mask must have identical shapes")
    if (graph_targets.device != semantic_targets.device
            or graph_targets.device != graph_valid.device):
        raise ValueError("all pseudo-label tensors must be on the same device")

    graph_valid = graph_valid.to(dtype=torch.bool)
    semantic_valid = semantic_targets != ignore_index
    if semantic_targets.shape == graph_targets.shape:
        semantic_covered = semantic_valid
        overlap = graph_valid & semantic_covered
        conflict = overlap & (graph_targets != semantic_targets)
    elif (semantic_targets.ndim == graph_targets.ndim + 1
          and semantic_targets.shape[0] == graph_targets.shape[0]
          and semantic_targets.shape[2:] == graph_targets.shape[1:]):
        semantic_covered = semantic_valid.any(dim=1)
        overlap = graph_valid & semantic_covered
        conflict = graph_valid & (
            semantic_valid & (graph_targets.unsqueeze(1) != semantic_targets)
        ).any(dim=1)
    else:
        raise ValueError("semantic and graph target dimensions are incompatible")
    graph_only = graph_valid & ~semantic_covered
    targets = graph_targets.masked_fill(~graph_only, ignore_index)
    return targets, graph_only, overlap, conflict
