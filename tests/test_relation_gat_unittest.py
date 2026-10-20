import unittest

import torch


class RelationAwareGATTest(unittest.TestCase):
    def test_empty_graph_returns_connected_empty_output(self):
        from models.graph.relation_gat import RelationAwareGAT

        layer = RelationAwareGAT(node_dim=8, heads=2, dropout=0.0)
        nodes = torch.empty(0, 8, requires_grad=True)
        empty_long = torch.empty(0, dtype=torch.long)
        output = layer(
            nodes, empty_long, empty_long, empty_long,
            torch.empty(0, dtype=torch.float32))

        self.assertEqual(output.shape, (0, 8))
        output.sum().backward()
        self.assertIsNotNone(nodes.grad)

    def test_single_node_self_loop_is_finite_and_differentiable(self):
        from models.graph.graph_builder import REL_SELF
        from models.graph.relation_gat import RelationAwareGAT

        layer = RelationAwareGAT(node_dim=8, heads=2, dropout=0.0)
        nodes = torch.randn(1, 8, requires_grad=True)
        output = layer(
            nodes,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([REL_SELF]),
            torch.tensor([1.0]),
        )
        output.square().sum().backward()

        self.assertEqual(output.shape, nodes.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(nodes.grad).all())
        self.assertTrue(any(parameter.grad is not None for parameter in layer.parameters()))

    def test_invalid_edge_index_is_rejected(self):
        from models.graph.graph_builder import REL_SELF
        from models.graph.relation_gat import RelationAwareGAT

        layer = RelationAwareGAT(node_dim=8, heads=2)
        with self.assertRaisesRegex(ValueError, "edge index"):
            layer(
                torch.randn(1, 8),
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([REL_SELF]),
                torch.tensor([1.0]),
            )

    def test_cuda_autocast_output_and_gradients_are_finite(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        from models.graph.graph_builder import REL_SELF, REL_SPATIAL
        from models.graph.relation_gat import RelationAwareGAT

        layer = RelationAwareGAT(node_dim=8, heads=2, dropout=0.0).cuda()
        nodes = torch.randn(2, 8, device="cuda", requires_grad=True)
        with torch.cuda.amp.autocast(enabled=True):
            output = layer(
                nodes,
                torch.tensor([0, 1, 0, 1], device="cuda"),
                torch.tensor([0, 1, 1, 0], device="cuda"),
                torch.tensor([REL_SELF, REL_SELF, REL_SPATIAL, REL_SPATIAL], device="cuda"),
                torch.tensor([1.0, 1.0, 0.8, 0.8], device="cuda"),
            )
            loss = output.square().mean()
        loss.backward()

        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(nodes.grad).all())


if __name__ == "__main__":
    unittest.main()
