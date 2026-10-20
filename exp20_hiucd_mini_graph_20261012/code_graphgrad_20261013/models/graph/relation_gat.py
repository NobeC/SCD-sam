import math

import torch
import torch.nn as nn

from models.graph.graph_builder import NUM_RELATIONS, REL_SELF


class RelationAwareGAT(nn.Module):
    """One-layer relation-aware graph attention over current batch nodes."""

    def __init__(
        self,
        node_dim=128,
        heads=4,
        dropout=0.1,
        edge_bias_beta=1.0,
        num_relations=NUM_RELATIONS,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.heads = int(heads)
        self.num_relations = int(num_relations)
        self.edge_bias_beta = float(edge_bias_beta)
        if self.node_dim <= 0 or self.heads <= 0 or self.node_dim % self.heads:
            raise ValueError("node_dim must be positive and divisible by heads")
        if self.num_relations <= 0 or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("invalid relation count or dropout")
        self.head_dim = self.node_dim // self.heads
        self.query = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.key = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.value = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.relation_key = nn.Embedding(
            self.num_relations, self.node_dim)
        self.relation_value = nn.Embedding(
            self.num_relations, self.node_dim)
        self.relation_bias = nn.Embedding(
            self.num_relations, self.heads)
        self.output = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.node_dim)

    def _validate(self, nodes, source, destination, relation, reliability):
        if nodes.ndim != 2 or nodes.shape[1] != self.node_dim:
            raise ValueError("nodes must have shape [N, node_dim]")
        edge_count = int(source.numel())
        tensors = (source, destination, relation, reliability)
        if any(tensor.ndim != 1 or tensor.numel() != edge_count for tensor in tensors):
            raise ValueError("edge tensors must have matching shape [E]")
        if edge_count:
            node_count = int(nodes.shape[0])
            if (int(source.min()) < 0 or int(destination.min()) < 0
                    or int(source.max()) >= node_count
                    or int(destination.max()) >= node_count):
                raise ValueError("edge index is outside current node range")
            if int(relation.min()) < 0 or int(relation.max()) >= self.num_relations:
                raise ValueError("relation ID is outside configured range")
            if (not torch.isfinite(reliability.float()).all()
                    or (reliability <= 0).any() or (reliability > 1).any()):
                raise FloatingPointError("edge reliability must be finite in (0, 1]")
        if not torch.isfinite(nodes.float()).all():
            raise FloatingPointError("GAT nodes contain NaN or Inf")

    def _with_self_loops(self, node_count, source, destination, relation, reliability):
        if node_count == 0:
            return source, destination, relation, reliability
        present = torch.zeros(node_count, dtype=torch.bool, device=source.device)
        if source.numel():
            self_edges = (source == destination) & (relation == REL_SELF)
            if self_edges.any():
                present[source[self_edges]] = True
        missing = torch.nonzero(~present, as_tuple=False).squeeze(1)
        if not missing.numel():
            return source, destination, relation, reliability
        return (
            torch.cat((source, missing)),
            torch.cat((destination, missing)),
            torch.cat((relation, torch.full_like(missing, REL_SELF))),
            torch.cat((reliability, torch.ones(
                missing.numel(), device=reliability.device, dtype=torch.float32))),
        )

    @staticmethod
    def _segment_softmax(scores, destination, node_count):
        attention = torch.zeros_like(scores, dtype=torch.float32)
        for node_index in range(node_count):
            selected = destination == node_index
            if selected.any():
                attention[selected] = torch.softmax(scores[selected].float(), dim=0)
        return attention

    def forward(self, nodes, source, destination, relation, reliability):
        device = nodes.device
        source = torch.as_tensor(source, device=device, dtype=torch.long)
        destination = torch.as_tensor(destination, device=device, dtype=torch.long)
        relation = torch.as_tensor(relation, device=device, dtype=torch.long)
        reliability = torch.as_tensor(reliability, device=device, dtype=torch.float32)
        self._validate(nodes, source, destination, relation, reliability)
        node_count = int(nodes.shape[0])
        if node_count == 0:
            parameter_zero = sum(parameter.sum() * 0.0 for parameter in self.parameters())
            return nodes.float() + parameter_zero
        source, destination, relation, reliability = self._with_self_loops(
            node_count, source, destination, relation, reliability)

        with torch.autocast(device_type=device.type, enabled=False):
            current = nodes.float()
            query = self.query(current).reshape(node_count, self.heads, self.head_dim)
            key = self.key(current).reshape(node_count, self.heads, self.head_dim)
            value = self.value(current).reshape(node_count, self.heads, self.head_dim)
            relation_key = self.relation_key(relation).float().reshape(
                -1, self.heads, self.head_dim)
            relation_value = self.relation_value(relation).float().reshape(
                -1, self.heads, self.head_dim)
            edge_key = key[source] + relation_key
            edge_value = value[source] + relation_value
            scores = (query[destination] * edge_key).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = scores + self.relation_bias(relation).float()
            scores = scores + self.edge_bias_beta * torch.log(
                reliability.clamp_min(1e-6)).unsqueeze(1)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("GAT attention scores contain NaN or Inf")
            attention = self._segment_softmax(scores, destination, node_count)
            accumulator = torch.zeros(
                (node_count, self.heads, self.head_dim),
                device=device,
                dtype=torch.float32,
            )
            accumulator.index_add_(
                0, destination, attention.unsqueeze(-1) * edge_value.float())
            propagated = self.output(accumulator.reshape(node_count, self.node_dim))
            output = self.norm(current + self.dropout(propagated.float()))
            if not torch.isfinite(output).all():
                raise FloatingPointError("GAT output contains NaN or Inf")
            return output.float()
