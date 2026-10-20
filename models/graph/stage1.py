from dataclasses import dataclass

import torch
import torch.nn as nn

from models.graph.edges import (
    ClassBalancedNodeQueue,
    EdgeNet,
    build_gt_edge_pairs,
    edge_loss,
)
from models.graph.prototypes import EMAPrototypeBank
from utils.sam_node_audit import GT


@dataclass
class Stage1Result:
    loss_graph_gt: torch.Tensor
    loss_edge_gt: torch.Tensor
    gt_node_count: int
    positive_pair_count: int
    negative_pair_count: int
    prototype_initialized: torch.Tensor


class SamGraphStage1(nn.Module):
    """Coordinate GT-only prototype and edge objectives for graph Stage 1."""

    def __init__(
        self,
        num_classes,
        node_dim=128,
        prototype_momentum=0.99,
        prototype_temperature=0.1,
        queue_capacity=256,
        max_pairs=256,
        hard_negative_ratio=0.5,
        edge_hidden_dim=256,
        relation_dim=8,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.node_dim = int(node_dim)
        self.max_pairs = int(max_pairs)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.prototype_projection = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.prototypes = EMAPrototypeBank(
            self.num_classes, self.node_dim,
            momentum=prototype_momentum, temperature=prototype_temperature)
        self.edge_net = EdgeNet(
            self.node_dim, relation_dim=relation_dim, hidden_dim=edge_hidden_dim)
        self.queue = ClassBalancedNodeQueue(
            self.num_classes, self.node_dim, capacity_per_class=queue_capacity)

    def _collect_gt(self, node_outputs, audits):
        if len(node_outputs) != len(audits):
            raise ValueError("node output and audit batch lengths must match")
        selected_nodes = []
        selected_classes = []
        device = self.prototype_projection.weight.device
        for output, audit in zip(node_outputs, audits):
            count = output.nodes_a.shape[0]
            tensors = (
                output.nodes_b,
                output.valid,
                audit.states_a,
                audit.states_b,
                audit.classes_a,
                audit.classes_b,
            )
            if output.nodes_a.ndim != 2 or output.nodes_a.shape[1] != self.node_dim:
                raise ValueError("temporal nodes must have shape [N, node_dim]")
            if output.nodes_b.shape != output.nodes_a.shape or any(tensor.shape[0] != count for tensor in tensors):
                raise ValueError("node and audit lengths must match for every sample")
            valid = output.valid.to(device=output.nodes_a.device, dtype=torch.bool)
            states_a = audit.states_a.to(device=output.nodes_a.device)
            states_b = audit.states_b.to(device=output.nodes_b.device)
            classes_a = audit.classes_a.to(device=output.nodes_a.device, dtype=torch.long)
            classes_b = audit.classes_b.to(device=output.nodes_b.device, dtype=torch.long)
            keep_a = valid & (states_a == GT)
            keep_b = valid.to(output.nodes_b.device) & (states_b == GT)
            if keep_a.any():
                selected_nodes.append(output.nodes_a[keep_a].to(device))
                selected_classes.append(classes_a[keep_a].to(device))
            if keep_b.any():
                selected_nodes.append(output.nodes_b[keep_b].to(device))
                selected_classes.append(classes_b[keep_b].to(device))
        if not selected_nodes:
            return (
                torch.empty((0, self.node_dim), device=device),
                torch.empty((0,), device=device, dtype=torch.long),
            )
        nodes = torch.cat(selected_nodes, dim=0)
        classes = torch.cat(selected_classes, dim=0)
        if (classes <= 0).any() or (classes >= self.num_classes).any():
            raise ValueError("GT graph classes must be in [1, num_classes)")
        return nodes, classes

    def forward(self, node_outputs, audits):
        raw_nodes, classes = self._collect_gt(node_outputs, audits)
        if raw_nodes.shape[0]:
            nodes = self.prototype_projection(raw_nodes.float())
        else:
            nodes = raw_nodes + self.prototype_projection.weight.sum() * 0.0

        self.prototypes.update(nodes, classes)
        loss_graph = self.prototypes.classification_loss(nodes, classes)
        pairs = build_gt_edge_pairs(
            nodes, classes, self.queue,
            max_pairs=self.max_pairs,
            hard_negative_ratio=self.hard_negative_ratio)
        logits = self.edge_net(pairs.left, pairs.right, pairs.relation_ids)
        loss_edge = edge_loss(self.edge_net, nodes, logits, pairs.targets)
        self.queue.enqueue(nodes, classes)
        return Stage1Result(
            loss_graph_gt=loss_graph,
            loss_edge_gt=loss_edge,
            gt_node_count=int(nodes.shape[0]),
            positive_pair_count=pairs.positive_count,
            negative_pair_count=pairs.negative_count,
            prototype_initialized=self.prototypes.initialized.detach().clone(),
        )
