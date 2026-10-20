import unittest

import torch

from models.graph.node_projector import TemporalNodeProjector


class GraphNodeProjectorTest(unittest.TestCase):
    def test_projects_variable_aligned_nodes_to_requested_dimension(self):
        projector = TemporalNodeProjector(
            stage_channels=2, neck_channels=3, node_dim=16,
            embedding_dim=4, dropout=0.0)
        stage_a = torch.randn(2, 2, 4, 4)
        stage_b = torch.randn(2, 2, 4, 4)
        neck_a = torch.randn(2, 3, 2, 2)
        neck_b = torch.randn(2, 3, 2, 2)
        masks_0 = torch.zeros((2, 8, 8), dtype=torch.bool)
        masks_0[0, 0:4, 0:4] = True
        masks_0[1, 4:8, 4:8] = True
        masks_1 = torch.empty((0, 8, 8), dtype=torch.bool)

        outputs = projector(stage_a, neck_a, stage_b, neck_b, [masks_0, masks_1])

        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].nodes_a.shape), (2, 16))
        self.assertEqual(tuple(outputs[0].nodes_b.shape), (2, 16))
        self.assertEqual(tuple(outputs[0].geometry.shape), (2, 5))
        self.assertEqual(outputs[0].valid.tolist(), [True, True])
        self.assertEqual(tuple(outputs[1].nodes_a.shape), (0, 16))
        self.assertTrue(torch.isfinite(outputs[0].nodes_a).all())

    def test_different_time_embeddings_produce_different_nodes(self):
        torch.manual_seed(3)
        projector = TemporalNodeProjector(
            stage_channels=1, neck_channels=1, node_dim=8,
            embedding_dim=2, dropout=0.0)
        feature = torch.ones(1, 1, 2, 2)
        mask = torch.ones((1, 4, 4), dtype=torch.bool)

        output = projector(feature, feature, feature, feature, [mask])[0]

        self.assertFalse(torch.equal(output.nodes_a, output.nodes_b))

    def test_channel_mismatch_is_rejected(self):
        projector = TemporalNodeProjector(stage_channels=2, neck_channels=3, node_dim=8)
        with self.assertRaisesRegex(ValueError, "channel"):
            projector(
                torch.ones(1, 1, 2, 2), torch.ones(1, 3, 2, 2),
                torch.ones(1, 1, 2, 2), torch.ones(1, 3, 2, 2),
                [torch.ones((1, 4, 4), dtype=torch.bool)],
            )


if __name__ == "__main__":
    unittest.main()
