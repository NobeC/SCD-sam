import unittest
from types import SimpleNamespace

import torch

from models.graph.stage1 import SamGraphStage1
from utils.sam_node_audit import GT, IGNORE, UNLABELED


class SamGraphStage2Test(unittest.TestCase):
    def _inputs(self):
        nodes_a = torch.tensor([
            [1., 0., 0., 0.],
            [0.8, 0.2, 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
        ], requires_grad=True)
        nodes_b = torch.tensor([
            [1., 0., 0., 0.],
            [0.7, 0.3, 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
        ], requires_grad=True)
        output = SimpleNamespace(
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            geometry=torch.tensor([
                [0.1, 0.1, 0.1, 0.1, 1.0],
                [0.2, 0.2, 0.1, 0.1, 1.0],
                [0.8, 0.8, 0.1, 0.1, 1.0],
                [0.5, 0.5, 0.1, 0.1, 1.0],
            ]),
            valid=torch.ones(4, dtype=torch.bool),
        )
        audit = SimpleNamespace(
            states_a=torch.tensor([GT, UNLABELED, UNLABELED, IGNORE]),
            states_b=torch.tensor([GT, UNLABELED, UNLABELED, IGNORE]),
            classes_a=torch.tensor([1, 0, 0, 0]),
            classes_b=torch.tensor([1, 0, 0, 0]),
        )
        return [output], [audit]

    def _models(self):
        from models.graph.stage2 import SamGraphStage2
        stage1 = SamGraphStage1(
            num_classes=3, node_dim=4, queue_capacity=8, max_pairs=8)
        with torch.no_grad():
            stage1.prototype_projection.weight.copy_(torch.eye(4))
            stage1.prototypes.prototypes[1].copy_(torch.tensor([1., 0., 0., 0.]))
            stage1.prototypes.prototypes[2].copy_(torch.tensor([0., 1., 0., 0.]))
            stage1.prototypes.initialized[1:] = True
            for parameter in stage1.edge_net.parameters():
                parameter.zero_()
        return SamGraphStage2(
            num_classes=3, node_dim=4, heads=2, dropout=0.0,
            local_topk=2, edge_threshold=0.4), stage1

    def test_outputs_are_aligned_to_original_masks_and_exclude_ignore(self):
        stage2, stage1 = self._models()
        outputs, audits = self._inputs()
        result = stage2(outputs, audits, stage1)

        self.assertEqual(result.predictions_a[0].shape, (4,))
        self.assertEqual(result.predictions_b[0].shape, (4,))
        self.assertFalse(bool(result.valid_a[0][3]))
        self.assertFalse(bool(result.valid_b[0][3]))
        self.assertEqual(result.node_count, 6)
        self.assertGreater(result.edge_count, 0)
        self.assertTrue(torch.isfinite(result.loss_graph_gt))
        self.assertTrue(torch.isfinite(result.loss_temporal))

    def test_graph_losses_backpropagate_to_gat_and_live_nodes(self):
        stage2, stage1 = self._models()
        outputs, audits = self._inputs()
        result = stage2(outputs, audits, stage1)
        loss = result.loss_graph_gt + result.loss_temporal
        loss.backward()

        self.assertTrue(any(p.grad is not None for p in stage2.parameters()))
        self.assertIsNotNone(outputs[0].nodes_a.grad)

    def test_edge_net_is_used_frozen_during_stage2(self):
        stage2, stage1 = self._models()
        outputs, audits = self._inputs()
        result = stage2(outputs, audits, stage1)
        (result.loss_graph_gt + result.loss_temporal).backward()

        self.assertTrue(all(p.grad is None for p in stage1.edge_net.parameters()))

    def test_no_initialized_semantic_prototype_disables_predictions(self):
        stage2, stage1 = self._models()
        stage1.prototypes.initialized.zero_()
        outputs, audits = self._inputs()
        result = stage2(outputs, audits, stage1)

        self.assertEqual(result.valid_node_count, 0)
        self.assertEqual(float(result.loss_graph_gt), 0.0)
        self.assertEqual(float(result.loss_temporal), 0.0)


if __name__ == "__main__":
    unittest.main()
