from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.sam_node_audit import GT, IGNORE, UNLABELED


REL_PROPAGATION = 0
REL_SPATIAL = 1
REL_TEMPORAL = 2
REL_SELF = 3
NUM_RELATIONS = 4


@dataclass
class Stage2NodeBatch:
    nodes: torch.Tensor
    states: torch.Tensor
    classes: torch.Tensor
    geometry: torch.Tensor
    sample_indices: torch.Tensor
    time_indices: torch.Tensor
    region_indices: torch.Tensor
    original_counts: tuple

    def temporal_pairs(self):
        positions = {}
        for position, key in enumerate(zip(
                self.sample_indices.tolist(),
                self.time_indices.tolist(),
                self.region_indices.tolist())):
            positions[key] = position
        pairs = []
        for (sample, time_index, region), left in sorted(positions.items()):
            if time_index != 0 or int(self.states[left]) != UNLABELED:
                continue
            right = positions.get((sample, 1, region))
            if right is not None and int(self.states[right]) == UNLABELED:
                pairs.append((left, right))
        return pairs

    def positions(self, sample_index, time_index):
        selected = (
            (self.sample_indices == int(sample_index))
            & (self.time_indices == int(time_index))
        )
        return torch.nonzero(selected, as_tuple=False).squeeze(1)


@dataclass
class SparseGraphEdges:
    source: torch.Tensor
    destination: torch.Tensor
    relation: torch.Tensor
    reliability: torch.Tensor
    propagation_count: int
    spatial_count: int
    temporal_count: int
    self_count: int

    @property
    def mean_reliability(self):
        if self.reliability.numel() == 0:
            return self.reliability.new_zeros(())
        return self.reliability.float().mean()


def _validate_temporal_output(output, audit, node_dim):
    count = int(output.nodes_a.shape[0])
    if output.nodes_a.ndim != 2 or output.nodes_a.shape[1] != node_dim:
        raise ValueError("temporal nodes must have shape [N, node_dim]")
    tensors = (
        output.nodes_b, output.geometry, output.valid,
        audit.states_a, audit.states_b, audit.classes_a, audit.classes_b,
    )
    if output.nodes_b.shape != output.nodes_a.shape:
        raise ValueError("A/B temporal node shapes must match")
    if output.geometry.shape != (count, 5):
        raise ValueError("node geometry must have shape [N, 5]")
    if any(tensor.shape[0] != count for tensor in tensors[2:]):
        raise ValueError("node output and audit lengths must match")


def flatten_stage2_nodes(node_outputs, audits, device):
    if len(node_outputs) != len(audits):
        raise ValueError("node output and audit batch lengths must match")
    node_dim = int(node_outputs[0].nodes_a.shape[1]) if node_outputs else 0
    nodes = []
    states = []
    classes = []
    geometries = []
    samples = []
    times = []
    regions = []
    counts = []
    for sample_index, (output, audit) in enumerate(zip(node_outputs, audits)):
        _validate_temporal_output(output, audit, node_dim)
        count = int(output.nodes_a.shape[0])
        counts.append(count)
        valid = output.valid.to(device=device, dtype=torch.bool)
        geometry = output.geometry.to(device=device, dtype=torch.float32)
        for time_index, (temporal_nodes, temporal_states, temporal_classes) in enumerate((
                (output.nodes_a, audit.states_a, audit.classes_a),
                (output.nodes_b, audit.states_b, audit.classes_b))):
            temporal_nodes = temporal_nodes.to(device)
            temporal_states = temporal_states.to(device=device, dtype=torch.long)
            temporal_classes = temporal_classes.to(device=device, dtype=torch.long)
            keep = valid & (temporal_states != IGNORE)
            selected = torch.nonzero(keep, as_tuple=False).squeeze(1)
            for region_index in selected.tolist():
                nodes.append(temporal_nodes[region_index])
                states.append(temporal_states[region_index])
                classes.append(temporal_classes[region_index])
                geometries.append(geometry[region_index])
                samples.append(sample_index)
                times.append(time_index)
                regions.append(region_index)
    if nodes:
        stacked_nodes = torch.stack(nodes)
        stacked_geometry = torch.stack(geometries)
    else:
        dtype = node_outputs[0].nodes_a.dtype if node_outputs else torch.float32
        stacked_nodes = torch.empty((0, node_dim), device=device, dtype=dtype)
        stacked_geometry = torch.empty((0, 5), device=device, dtype=torch.float32)
    return Stage2NodeBatch(
        nodes=stacked_nodes,
        states=torch.as_tensor(states, device=device, dtype=torch.long),
        classes=torch.as_tensor(classes, device=device, dtype=torch.long),
        geometry=stacked_geometry,
        sample_indices=torch.as_tensor(samples, device=device, dtype=torch.long),
        time_indices=torch.as_tensor(times, device=device, dtype=torch.long),
        region_indices=torch.as_tensor(regions, device=device, dtype=torch.long),
        original_counts=tuple(counts),
    )


def _rank_by_cosine(nodes, query_index, candidate_indices, limit):
    if limit <= 0 or not candidate_indices:
        return []
    query = F.normalize(nodes[query_index].float(), dim=0, eps=1e-6)
    ranked = []
    for candidate in candidate_indices:
        value = F.cosine_similarity(
            query.unsqueeze(0), nodes[candidate].float().unsqueeze(0), dim=1, eps=1e-6)[0]
        ranked.append((-float(value.detach()), int(candidate)))
    return [candidate for _, candidate in sorted(ranked)[:limit]]


def _rank_by_center(geometry, query_index, candidate_indices, limit):
    ranked = []
    center = geometry[query_index, :2].float()
    for candidate in candidate_indices:
        distance = torch.linalg.vector_norm(center - geometry[candidate, :2].float())
        ranked.append((float(distance.detach()), int(candidate)))
    return [candidate for _, candidate in sorted(ranked)[:max(int(limit), 0)]]


def build_stage2_edges(node_batch, edge_net, local_topk=8, edge_threshold=0.5):
    if int(local_topk) < 0:
        raise ValueError("local_topk must be non-negative")
    if not 0.0 <= float(edge_threshold) <= 1.0:
        raise ValueError("edge_threshold must be in [0, 1]")
    count = int(node_batch.nodes.shape[0])
    device = node_batch.nodes.device
    records = {}

    def add(source, destination, relation, reliability):
        key = (int(source), int(destination), int(relation))
        value = float(reliability)
        if not torch.isfinite(torch.tensor(value)) or not 0.0 < value <= 1.0:
            raise FloatingPointError("graph edge reliability must be finite in (0, 1]")
        records[key] = max(value, records.get(key, 0.0))

    for index in range(count):
        add(index, index, REL_SELF, 1.0)

    for sample_index in sorted(set(node_batch.sample_indices.tolist())):
        for time_index in (0, 1):
            positions = node_batch.positions(sample_index, time_index).tolist()
            gt_positions = [p for p in positions if int(node_batch.states[p]) == GT]
            u_positions = [p for p in positions if int(node_batch.states[p]) == UNLABELED]
            candidate_pairs = []
            for gt_position in gt_positions:
                for u_position in _rank_by_cosine(
                        node_batch.nodes, gt_position, u_positions, int(local_topk)):
                    candidate_pairs.append((gt_position, u_position))
            if candidate_pairs:
                left = torch.stack([node_batch.nodes[left] for left, _ in candidate_pairs])
                right = torch.stack([node_batch.nodes[right] for _, right in candidate_pairs])
                relation_ids = torch.zeros(len(candidate_pairs), device=device, dtype=torch.long)
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    reliability = torch.sigmoid(
                        edge_net(left.float(), right.float(), relation_ids)).float()
                for (left_index, right_index), weight in zip(
                        candidate_pairs, reliability.detach().cpu().tolist()):
                    if weight >= float(edge_threshold):
                        add(left_index, right_index, REL_PROPAGATION, weight)
                        add(right_index, left_index, REL_PROPAGATION, weight)
            for u_position in u_positions:
                candidates = [other for other in u_positions if other != u_position]
                for other in _rank_by_center(
                        node_batch.geometry, u_position, candidates, int(local_topk)):
                    distance = torch.linalg.vector_norm(
                        node_batch.geometry[u_position, :2].float()
                        - node_batch.geometry[other, :2].float())
                    prior = 1.0 / (1.0 + float(distance.detach()))
                    add(other, u_position, REL_SPATIAL, prior)

    for left, right in node_batch.temporal_pairs():
        add(left, right, REL_TEMPORAL, 1.0)
        add(right, left, REL_TEMPORAL, 1.0)

    ordered = sorted(records.items(), key=lambda item: item[0])
    source = torch.tensor([key[0] for key, _ in ordered], device=device, dtype=torch.long)
    destination = torch.tensor([key[1] for key, _ in ordered], device=device, dtype=torch.long)
    relation = torch.tensor([key[2] for key, _ in ordered], device=device, dtype=torch.long)
    reliability = torch.tensor([value for _, value in ordered], device=device, dtype=torch.float32)
    relation_list = relation.tolist()
    return SparseGraphEdges(
        source=source,
        destination=destination,
        relation=relation,
        reliability=reliability,
        propagation_count=relation_list.count(REL_PROPAGATION),
        spatial_count=relation_list.count(REL_SPATIAL),
        temporal_count=relation_list.count(REL_TEMPORAL),
        self_count=relation_list.count(REL_SELF),
    )
