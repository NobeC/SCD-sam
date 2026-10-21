from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph.graph_builder import build_stage2_edges, flatten_stage2_nodes
from models.graph.relation_gat import RelationAwareGAT
from utils.sam_node_audit import GT


@dataclass
class Stage2Result:
    loss_graph_gt: torch.Tensor
    loss_temporal: torch.Tensor
    probabilities_a: list
    predictions_a: list
    confidences_a: list
    valid_a: list
    probabilities_b: list
    predictions_b: list
    confidences_b: list
    valid_b: list
    node_count: int
    valid_node_count: int
    edge_count: int
    propagation_edge_count: int
    spatial_edge_count: int
    temporal_edge_count: int
    mean_edge_reliability: float
    graph_gt_count: int
    graph_gt_correct: int
    temporal_pair_count: int


class SamGraphStage2(nn.Module):
    """Batch-local relation graph propagation using Stage1 edge/prototype state."""

    def __init__(
        self,
        num_classes,
        node_dim=128,
        heads=4,
        dropout=0.1,
        local_topk=8,
        edge_threshold=0.5,
        edge_bias_beta=1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.node_dim = int(node_dim)
        self.local_topk = int(local_topk)
        self.edge_threshold = float(edge_threshold)
        self.gat = RelationAwareGAT(
            node_dim=self.node_dim,
            heads=int(heads),
            dropout=float(dropout),
            edge_bias_beta=float(edge_bias_beta),
        )

    @staticmethod
    def _symmetric_kl(left, right):
        left = left.float().clamp_min(1e-8)
        right = right.float().clamp_min(1e-8)
        return 0.5 * (
            F.kl_div(left.log(), right, reduction="sum")
            + F.kl_div(right.log(), left, reduction="sum")
        )

    def _aligned_predictions(self, batch, probabilities, predictions, confidence, valid):
        probs_a, probs_b = [], []
        pred_a, conf_a, valid_a = [], [], []
        pred_b, conf_b, valid_b = [], [], []
        device = predictions.device
        for sample_index, count in enumerate(batch.original_counts):
            temporal = []
            for time_index in (0, 1):
                probs = torch.zeros(
                    (count, self.num_classes), device=device, dtype=torch.float32)
                pred = torch.zeros(count, device=device, dtype=torch.long)
                conf = torch.zeros(count, device=device, dtype=torch.float32)
                is_valid = torch.zeros(count, device=device, dtype=torch.bool)
                positions = batch.positions(sample_index, time_index)
                if positions.numel():
                    regions = batch.region_indices[positions]
                    probs[regions] = probabilities[positions]
                    pred[regions] = predictions[positions]
                    conf[regions] = confidence[positions]
                    is_valid[regions] = valid[positions]
                temporal.append((probs, pred, conf, is_valid))
            probs_a.append(temporal[0][0])
            pred_a.append(temporal[0][1])
            conf_a.append(temporal[0][2])
            valid_a.append(temporal[0][3])
            probs_b.append(temporal[1][0])
            pred_b.append(temporal[1][1])
            conf_b.append(temporal[1][2])
            valid_b.append(temporal[1][3])
        return probs_a, pred_a, conf_a, valid_a, probs_b, pred_b, conf_b, valid_b

    def forward(self, node_outputs, audits, stage1):
        device = self.gat.query.weight.device
        batch = flatten_stage2_nodes(node_outputs, audits, device=device)
        if batch.nodes.shape[1] != self.node_dim:
            raise ValueError("Stage2 node dimension does not match GAT configuration")
        edges = build_stage2_edges(
            batch, stage1.edge_net,
            local_topk=self.local_topk,
            edge_threshold=self.edge_threshold,
        )
        propagated = self.gat(
            batch.nodes,
            edges.source,
            edges.destination,
            edges.relation,
            edges.reliability,
        )
        initialized = stage1.prototypes.initialized.to(device=device).clone()
        initialized[0] = False
        has_semantic_prototype = bool(initialized.any())
        node_count = int(propagated.shape[0])
        zero = propagated.sum() * 0.0
        if node_count and has_semantic_prototype:
            projected = stage1.prototype_projection(propagated.float())
            logits = stage1.prototypes.logits(projected)
            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predictions = probabilities.max(dim=1)
            valid = torch.isfinite(confidence) & initialized[predictions]

            # Prototype consistency gate: compare post-GAT prediction against
            # pre-GAT prototype lookup. Nodes where propagation flipped the
            # prediction away from the nearest prototype are unreliable.
            with torch.no_grad():
                orig_projected = stage1.prototype_projection(batch.nodes.float())
                orig_logits = stage1.prototypes.logits(orig_projected)
                orig_pred = orig_logits.argmax(dim=1)
            consistent = (predictions == orig_pred) | ~initialized[orig_pred.clamp(0, self.num_classes - 1)]
            valid = valid & consistent
        else:
            probabilities = propagated.new_zeros((node_count, self.num_classes))
            predictions = torch.zeros(node_count, device=device, dtype=torch.long)
            confidence = torch.zeros(node_count, device=device, dtype=torch.float32)
            valid = torch.zeros(node_count, device=device, dtype=torch.bool)

        gt = (batch.states == GT) & (batch.classes > 0)
        if node_count and has_semantic_prototype:
            gt = gt & initialized[batch.classes.clamp(0, self.num_classes - 1)]
        else:
            gt = torch.zeros_like(gt)
        if gt.any():
            loss_graph_gt = F.cross_entropy(logits[gt], batch.classes[gt])
            graph_gt_correct = int((predictions[gt] == batch.classes[gt]).sum())
        else:
            loss_graph_gt = zero
            graph_gt_correct = 0

        temporal_losses = []
        for left, right in batch.temporal_pairs():
            if bool(valid[left] & valid[right]):
                temporal_losses.append(self._symmetric_kl(
                    probabilities[left], probabilities[right]))
        loss_temporal = (
            torch.stack(temporal_losses).mean() if temporal_losses else zero)
        aligned = self._aligned_predictions(
            batch, probabilities.detach(), predictions.detach(),
            confidence.detach(), valid.detach())
        return Stage2Result(
            loss_graph_gt=loss_graph_gt,
            loss_temporal=loss_temporal,
            probabilities_a=aligned[0],
            predictions_a=aligned[1],
            confidences_a=aligned[2],
            valid_a=aligned[3],
            probabilities_b=aligned[4],
            predictions_b=aligned[5],
            confidences_b=aligned[6],
            valid_b=aligned[7],
            node_count=node_count,
            valid_node_count=int(valid.sum()),
            edge_count=int(edges.source.numel()),
            propagation_edge_count=edges.propagation_count,
            spatial_edge_count=edges.spatial_count,
            temporal_edge_count=edges.temporal_count,
            mean_edge_reliability=float(edges.mean_reliability.detach()),
            graph_gt_count=int(gt.sum()),
            graph_gt_correct=graph_gt_correct,
            temporal_pair_count=len(temporal_losses),
        )
