from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedNodeQueue:
    """Detached CPU FIFO storage partitioned by true semantic class."""

    def __init__(self, num_classes, node_dim=128, capacity_per_class=256):
        self.num_classes = int(num_classes)
        self.node_dim = int(node_dim)
        self.capacity_per_class = int(capacity_per_class)
        if self.num_classes < 2 or self.node_dim <= 0 or self.capacity_per_class <= 0:
            raise ValueError("queue dimensions and capacity must be positive")
        self._queues = {
            class_id: deque(maxlen=self.capacity_per_class)
            for class_id in range(1, self.num_classes)
        }

    def enqueue(self, nodes, classes):
        classes = torch.as_tensor(classes, dtype=torch.long, device=nodes.device)
        if nodes.ndim != 2 or nodes.shape[1] != self.node_dim:
            raise ValueError("nodes must have shape [N, {}]".format(self.node_dim))
        if classes.ndim != 1 or classes.shape[0] != nodes.shape[0]:
            raise ValueError("classes must have shape [N]")
        if classes.numel() and (int(classes.min()) < 0 or int(classes.max()) >= self.num_classes):
            raise ValueError("queue class ID is outside the configured dataset range")
        normalized = F.normalize(nodes.detach().float(), dim=1, eps=1e-6).cpu()
        for node, class_id in zip(normalized, classes.detach().cpu().tolist()):
            if class_id == 0:
                continue
            self._queues[int(class_id)].append(node.clone())

    def snapshot(self, device):
        nodes = []
        classes = []
        for class_id in range(1, self.num_classes):
            for node in self._queues[class_id]:
                nodes.append(node)
                classes.append(class_id)
        if not nodes:
            return (
                torch.empty((0, self.node_dim), device=device, dtype=torch.float32),
                torch.empty((0,), device=device, dtype=torch.long),
            )
        return (
            torch.stack(nodes).to(device=device, dtype=torch.float32),
            torch.tensor(classes, device=device, dtype=torch.long),
        )


@dataclass
class EdgePairs:
    left: torch.Tensor
    right: torch.Tensor
    targets: torch.Tensor
    relation_ids: torch.Tensor
    positive_count: int
    negative_count: int


def _candidate_pair_indices(classes, current_count):
    """Return legacy-ordered positive/negative pairs without Python CUDA syncs."""
    classes = torch.as_tensor(classes, dtype=torch.long)
    total_count = int(classes.shape[0])
    current_count = int(current_count)
    if current_count < 0 or current_count > total_count:
        raise ValueError("current_count must be within the combined node count")
    if total_count < 2 or current_count == 0:
        empty = torch.empty((0, 2), device=classes.device, dtype=torch.long)
        return empty, empty.clone()

    pair_columns = torch.triu_indices(
        total_count, total_count, offset=1, device=classes.device)
    pair_columns = pair_columns[:, pair_columns[0] < current_count]
    pairs = pair_columns.t().contiguous()
    same_class = classes[pairs[:, 0]] == classes[pairs[:, 1]]
    return pairs[same_class], pairs[~same_class]


def _select_negative_indices(candidates, combined, limit, hard_negative_ratio):
    candidate_count = int(candidates.shape[0])
    limit = min(int(limit), candidate_count)
    if limit <= 0:
        return torch.empty((0,), device=combined.device, dtype=torch.long)

    hard_count = min(limit, int(round(limit * hard_negative_ratio)))
    if hard_count:
        normalized = F.normalize(combined.float(), dim=1, eps=1e-6)
        scores = (
            normalized[candidates[:, 0]] * normalized[candidates[:, 1]]
        ).sum(dim=1)
        ranked = torch.argsort(scores, descending=True, stable=True)
        hard_selected = ranked[:hard_count]
    else:
        hard_selected = torch.empty(
            (0,), device=combined.device, dtype=torch.long)

    remaining_count = limit - hard_count
    if remaining_count <= 0:
        return hard_selected
    available = torch.ones(
        candidate_count, device=combined.device, dtype=torch.bool)
    available[hard_selected] = False
    fill = torch.arange(
        candidate_count, device=combined.device, dtype=torch.long
    )[available][:remaining_count]
    return torch.cat((hard_selected, fill))


def build_gt_edge_pairs(
    nodes,
    classes,
    queue,
    max_pairs=256,
    hard_negative_ratio=0.5,
):
    """Construct balanced GT-only current/current and current/queue edge pairs."""
    classes = torch.as_tensor(classes, dtype=torch.long, device=nodes.device)
    if nodes.ndim != 2 or nodes.shape[1] != queue.node_dim:
        raise ValueError("nodes do not match queue node_dim")
    if classes.ndim != 1 or classes.shape[0] != nodes.shape[0]:
        raise ValueError("classes must have shape [N]")
    if classes.numel() and (int(classes.min()) <= 0 or int(classes.max()) >= queue.num_classes):
        raise ValueError("edge targets require true semantic classes in [1, num_classes)")
    if max_pairs < 0 or not 0.0 <= hard_negative_ratio <= 1.0:
        raise ValueError("invalid edge pair sampling configuration")

    queued_nodes, queued_classes = queue.snapshot(nodes.device)
    combined = torch.cat((nodes.float(), queued_nodes), dim=0)
    combined_classes = torch.cat((classes, queued_classes), dim=0)
    positives, negatives = _candidate_pair_indices(
        combined_classes, current_count=nodes.shape[0])

    if len(positives) and len(negatives):
        side_limit = min(len(positives), len(negatives), max_pairs // 2)
        selected_positive = torch.arange(
            side_limit, device=nodes.device, dtype=torch.long)
        selected_negative = _select_negative_indices(
            negatives, combined, side_limit, hard_negative_ratio)
    elif len(positives):
        selected_positive = torch.arange(
            min(len(positives), max_pairs), device=nodes.device, dtype=torch.long)
        selected_negative = torch.empty(
            (0,), device=nodes.device, dtype=torch.long)
    else:
        selected_positive = torch.empty(
            (0,), device=nodes.device, dtype=torch.long)
        selected_negative = _select_negative_indices(
            negatives, combined, min(len(negatives), max_pairs),
            hard_negative_ratio)

    selected_pairs = torch.cat((
        positives[selected_positive],
        negatives[selected_negative],
    ))
    positive_count = int(selected_positive.numel())
    negative_count = int(selected_negative.numel())
    if selected_pairs.numel() == 0:
        empty_nodes = nodes.new_empty((0, queue.node_dim), dtype=torch.float32)
        return EdgePairs(
            empty_nodes, empty_nodes.clone(),
            nodes.new_empty((0,), dtype=torch.float32),
            nodes.new_empty((0,), dtype=torch.long), 0, 0)

    targets = torch.cat((
        torch.ones(positive_count, device=nodes.device, dtype=torch.float32),
        torch.zeros(negative_count, device=nodes.device, dtype=torch.float32),
    ))
    return EdgePairs(
        left=combined[selected_pairs[:, 0]],
        right=combined[selected_pairs[:, 1]],
        targets=targets,
        relation_ids=torch.zeros(
            len(selected_pairs), device=nodes.device, dtype=torch.long),
        positive_count=positive_count,
        negative_count=negative_count,
    )


class EdgeNet(nn.Module):
    def __init__(self, node_dim=128, relation_dim=8, hidden_dim=256, num_relations=1):
        super().__init__()
        self.node_dim = int(node_dim)
        self.num_relations = int(num_relations)
        if min(self.node_dim, relation_dim, hidden_dim, self.num_relations) <= 0:
            raise ValueError("EdgeNet dimensions must be positive")
        self.relation_embedding = nn.Embedding(self.num_relations, int(relation_dim))
        input_dim = 4 * self.node_dim + int(relation_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, left, right, relation_ids):
        if left.ndim != 2 or right.shape != left.shape or left.shape[1] != self.node_dim:
            raise ValueError("edge endpoints must have matching [E, node_dim] shapes")
        relation_ids = torch.as_tensor(relation_ids, dtype=torch.long, device=left.device)
        if relation_ids.shape != (left.shape[0],):
            raise ValueError("relation_ids must have shape [E]")
        if relation_ids.numel() and (int(relation_ids.min()) < 0 or int(relation_ids.max()) >= self.num_relations):
            raise ValueError("relation ID is outside EdgeNet range")
        # Edge logits are especially sensitive to FP16 overflow: the pairwise
        # product can amplify otherwise finite node values, and both Linear
        # layers accumulate over many features.  Keep this small auxiliary MLP
        # in FP32 even when the surrounding training step uses autocast.
        with torch.autocast(device_type=left.device.type, enabled=False):
            left = left.float()
            right = right.float()
            relation = self.relation_embedding(relation_ids).float()
            raw = torch.cat(
                (left, right, (left - right).abs(), left * right, relation), dim=1)
            return self.network(raw).squeeze(1)


def edge_loss(edge_net, reference_nodes, logits, targets):
    targets = torch.as_tensor(targets, dtype=torch.float32, device=logits.device)
    if logits.shape != targets.shape:
        raise ValueError("edge logits and targets must have matching shapes")
    if logits.numel() == 0:
        parameter_zero = sum(parameter.sum() * 0.0 for parameter in edge_net.parameters())
        return reference_nodes.sum() * 0.0 + parameter_zero
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
