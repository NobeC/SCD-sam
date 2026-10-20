import unittest

import torch
import torch.nn as nn

from models.graph.node_projector import TemporalNodeOutput
from utils.sam_node_audit import GT, IGNORE, UNLABELED, SamNodeAudit


class ConstantEdgeNet(nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = float(logit)

    def forward(self, left, right, relation_ids):
        return left.new_full((left.shape[0],), self.logit, dtype=torch.float32)


def audit(states_a, states_b, classes_a, classes_b):
    count = len(states_a)
    return SamNodeAudit(
        states_a=torch.tensor(states_a),
        states_b=torch.tensor(states_b),
        classes_a=torch.tensor(classes_a),
        classes_b=torch.tensor(classes_b),
        change_ratio=torch.zeros(count),
        purity_a=torch.ones(count),
        purity_b=torch.ones(count),
        changed_pixels=torch.zeros(count, dtype=torch.long),
        total_changed_pixels=0,
        covered_changed_pixels=0,
        gt_covered_changed_pixels=0,
    )


class GraphStage2BuilderTest(unittest.TestCase):
    def setUp(self):
        from models.graph.graph_builder import flatten_stage2_nodes

        output = TemporalNodeOutput(
            nodes_a=torch.tensor([
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]),
            nodes_b=torch.tensor([
                [0.8, 0.2],
                [0.7, 0.3],
                [0.0, 1.0],
            ]),
            geometry=torch.tensor([
                [0.10, 0.10, 0.1, 0.1, 0.01],
                [0.20, 0.10, 0.1, 0.1, 0.01],
                [0.90, 0.90, 0.1, 0.1, 0.01],
            ]),
            valid=torch.tensor([True, True, True]),
        )
        node_audit = audit(
            [GT, UNLABELED, IGNORE],
            [UNLABELED, UNLABELED, GT],
            [1, 0, 0],
            [0, 0, 2],
        )
        self.batch = flatten_stage2_nodes([output], [node_audit], torch.device("cpu"))

    def test_flatten_excludes_ignore_and_keeps_temporal_identity(self):
        self.assertEqual(self.batch.nodes.shape, (5, 2))
        self.assertNotIn(
            (0, 0, 2),
            list(zip(
                self.batch.sample_indices.tolist(),
                self.batch.time_indices.tolist(),
                self.batch.region_indices.tolist(),
            )),
        )
        paired = self.batch.temporal_pairs()
        self.assertEqual(paired, [(1, 3)])

    def test_builds_gt_u_spatial_temporal_and_self_relations(self):
        from models.graph.graph_builder import (
            REL_PROPAGATION,
            REL_SELF,
            REL_SPATIAL,
            REL_TEMPORAL,
            build_stage2_edges,
        )

        edges = build_stage2_edges(
            self.batch,
            ConstantEdgeNet(10.0),
            local_topk=1,
            edge_threshold=0.5,
        )

        relations = edges.relation.tolist()
        self.assertEqual(relations.count(REL_SELF), 5)
        self.assertGreaterEqual(relations.count(REL_PROPAGATION), 4)
        self.assertEqual(relations.count(REL_SPATIAL), 2)
        self.assertEqual(relations.count(REL_TEMPORAL), 2)
        self.assertTrue(torch.all(edges.reliability > 0))
        self.assertTrue(torch.all(edges.reliability <= 1))

    def test_edge_threshold_removes_low_reliability_gt_u(self):
        from models.graph.graph_builder import REL_PROPAGATION, build_stage2_edges

        edges = build_stage2_edges(
            self.batch,
            ConstantEdgeNet(-10.0),
            local_topk=1,
            edge_threshold=0.5,
        )

        self.assertNotIn(REL_PROPAGATION, edges.relation.tolist())

    def test_candidate_order_is_deterministic(self):
        from models.graph.graph_builder import build_stage2_edges

        first = build_stage2_edges(
            self.batch, ConstantEdgeNet(10.0), local_topk=2, edge_threshold=0.5)
        second = build_stage2_edges(
            self.batch, ConstantEdgeNet(10.0), local_topk=2, edge_threshold=0.5)

        self.assertTrue(torch.equal(first.source, second.source))
        self.assertTrue(torch.equal(first.destination, second.destination))
        self.assertTrue(torch.equal(first.relation, second.relation))
        self.assertTrue(torch.equal(first.reliability, second.reliability))


if __name__ == "__main__":
    unittest.main()
