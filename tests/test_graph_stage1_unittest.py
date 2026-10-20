import unittest

import torch

from models.graph.node_projector import TemporalNodeOutput
from models.graph.stage1 import SamGraphStage1
from utils.sam_node_audit import GT, UNLABELED, SamNodeAudit


def _audit(states_a, states_b, classes_a, classes_b):
    count = len(states_a)
    return SamNodeAudit(
        states_a=torch.tensor(states_a),
        states_b=torch.tensor(states_b),
        classes_a=torch.tensor(classes_a),
        classes_b=torch.tensor(classes_b),
        change_ratio=torch.ones(count),
        purity_a=torch.ones(count),
        purity_b=torch.ones(count),
        changed_pixels=torch.ones(count, dtype=torch.long),
        total_changed_pixels=count,
        covered_changed_pixels=count,
        gt_covered_changed_pixels=count,
    )


class GraphStage1Test(unittest.TestCase):
    def test_normal_gt_nodes_produce_finite_losses_and_pairs(self):
        stage = SamGraphStage1(num_classes=3, node_dim=4, queue_capacity=8, max_pairs=8)
        output = TemporalNodeOutput(
            nodes_a=torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], requires_grad=True),
            nodes_b=torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]], requires_grad=True),
            geometry=torch.zeros(2, 5),
            valid=torch.tensor([True, True]),
        )
        audit = _audit([GT, GT], [GT, GT], [1, 2], [1, 2])

        result = stage([output], [audit])
        total = result.loss_graph_gt + result.loss_edge_gt
        total.backward()

        self.assertEqual(result.gt_node_count, 4)
        self.assertGreater(result.positive_pair_count, 0)
        self.assertGreater(result.negative_pair_count, 0)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(result.prototype_initialized.tolist()[1:])

    def test_empty_gt_returns_connected_zero_losses(self):
        stage = SamGraphStage1(num_classes=3, node_dim=4)
        nodes_a = torch.randn(1, 4, requires_grad=True)
        nodes_b = torch.randn(1, 4, requires_grad=True)
        output = TemporalNodeOutput(nodes_a, nodes_b, torch.zeros(1, 5), torch.tensor([True]))
        audit = _audit([UNLABELED], [UNLABELED], [0], [0])

        result = stage([output], [audit])
        (result.loss_graph_gt + result.loss_edge_gt).backward()

        self.assertEqual(result.gt_node_count, 0)
        self.assertEqual(float(result.loss_graph_gt), 0.0)
        self.assertEqual(float(result.loss_edge_gt), 0.0)

    def test_queue_is_used_only_after_first_call_pair_construction(self):
        stage = SamGraphStage1(num_classes=3, node_dim=2, queue_capacity=4, max_pairs=4)
        audit = _audit([GT], [UNLABELED], [1], [0])
        first_output = TemporalNodeOutput(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 5), torch.tensor([True]))

        first = stage([first_output], [audit])
        second = stage([first_output], [audit])

        self.assertEqual(first.positive_pair_count, 0)
        self.assertEqual(second.positive_pair_count, 1)

    def test_node_and_audit_length_mismatch_is_rejected(self):
        stage = SamGraphStage1(num_classes=3, node_dim=2)
        output = TemporalNodeOutput(
            torch.ones(1, 2), torch.ones(1, 2),
            torch.zeros(1, 5), torch.tensor([True]))
        audit = _audit([GT, GT], [GT, GT], [1, 1], [1, 1])
        with self.assertRaisesRegex(ValueError, "length"):
            stage([output], [audit])


if __name__ == "__main__":
    unittest.main()
