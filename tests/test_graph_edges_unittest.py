import unittest

import torch

from models.graph.edges import (
    ClassBalancedNodeQueue,
    EdgeNet,
    _candidate_pair_indices,
    build_gt_edge_pairs,
    edge_loss,
)


class GraphEdgesTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 autocast")
    def test_edge_net_logits_and_loss_remain_fp32_under_cuda_autocast(self):
        edge_net = EdgeNet(node_dim=2, relation_dim=2, hidden_dim=8).cuda()
        with torch.no_grad():
            for parameter in edge_net.parameters():
                parameter.fill_(1.0)
        left = torch.full((2, 2), 100.0, device="cuda", requires_grad=True)
        right = torch.full((2, 2), 100.0, device="cuda", requires_grad=True)
        relation_ids = torch.zeros(2, device="cuda", dtype=torch.long)
        targets = torch.tensor([1.0, 0.0], device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = edge_net(left, right, relation_ids)
            loss = edge_loss(edge_net, left, logits, targets)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(left.grad).all())

    def test_queue_is_detached_cpu_fifo_and_ignores_class_zero(self):
        queue = ClassBalancedNodeQueue(3, node_dim=2, capacity_per_class=2)
        nodes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True)
        queue.enqueue(nodes, torch.tensor([1, 1, 1]))
        queue.enqueue(torch.tensor([[1.0, 1.0]]), torch.tensor([0]))

        stored, classes = queue.snapshot(torch.device("cpu"))

        self.assertEqual(stored.shape, (2, 2))
        self.assertEqual(classes.tolist(), [1, 1])
        self.assertFalse(stored.requires_grad)
        self.assertEqual(stored.device.type, "cpu")
        torch.testing.assert_close(stored[0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(stored[1], torch.tensor([-1.0, 0.0]))

    def test_candidate_pairs_keep_legacy_order_and_exclude_queue_only_pairs(self):
        positive, negative = _candidate_pair_indices(
            torch.tensor([1, 2, 1, 2]), current_count=2)

        torch.testing.assert_close(
            positive, torch.tensor([[0, 2], [1, 3]], dtype=torch.long))
        torch.testing.assert_close(
            negative, torch.tensor([[0, 1], [0, 3], [1, 2]], dtype=torch.long))

    def test_pair_builder_balances_positive_and_negative_pairs(self):
        nodes = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
        classes = torch.tensor([1, 1, 2, 2])
        queue = ClassBalancedNodeQueue(3, 2, capacity_per_class=4)

        pairs = build_gt_edge_pairs(nodes, classes, queue, max_pairs=4, hard_negative_ratio=0.5)

        self.assertEqual(pairs.positive_count, 2)
        self.assertEqual(pairs.negative_count, 2)
        self.assertEqual(sorted(pairs.targets.tolist()), [0.0, 0.0, 1.0, 1.0])

    def test_hard_negative_selects_highest_cosine_pair(self):
        nodes = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        classes = torch.tensor([1, 2, 3])
        queue = ClassBalancedNodeQueue(4, 2, capacity_per_class=2)

        pairs = build_gt_edge_pairs(nodes, classes, queue, max_pairs=1, hard_negative_ratio=1.0)

        self.assertEqual(pairs.negative_count, 1)
        torch.testing.assert_close(pairs.left[0], nodes[0])
        torch.testing.assert_close(pairs.right[0], nodes[1])

    def test_edge_net_loss_is_finite(self):
        edge_net = EdgeNet(node_dim=2, relation_dim=2, hidden_dim=8)
        left = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        right = torch.tensor([[0.9, 0.1], [1.0, 0.0]], requires_grad=True)
        logits = edge_net(left, right, torch.zeros(2, dtype=torch.long))

        loss = edge_loss(edge_net, left, logits, torch.tensor([1.0, 0.0]))
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(left.grad)

    def test_empty_edge_loss_is_connected_to_nodes_and_parameters(self):
        edge_net = EdgeNet(node_dim=2, relation_dim=2, hidden_dim=8)
        nodes = torch.empty((0, 2), requires_grad=True)
        loss = edge_loss(edge_net, nodes, torch.empty((0,)), torch.empty((0,)))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(nodes.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in edge_net.parameters()))


if __name__ == "__main__":
    unittest.main()
