import unittest

import torch

from models.graph.node_projector import TemporalNodeProjector
from models.graph.stage1 import SamGraphStage1
from utils.graph_experiment_state import add_graph_training_state, restore_graph_training_state


class GraphExperimentStateTest(unittest.TestCase):
    def _modules(self):
        projector = TemporalNodeProjector(stage_channels=2, neck_channels=2, node_dim=4,
                                          embedding_dim=2, hidden_dim=8, dropout=0.0)
        stage1 = SamGraphStage1(num_classes=3, node_dim=4, edge_hidden_dim=8,
                                relation_dim=2, queue_capacity=4, max_pairs=4)
        return projector, stage1

    def test_round_trip_restores_modules_optimizer_and_queue(self):
        projector, stage1 = self._modules()
        optimizer = torch.optim.AdamW(list(projector.parameters()) + list(stage1.parameters()), lr=1e-3)
        stage1.queue.enqueue(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([1]))
        payload = add_graph_training_state({"format_version": 1}, projector, stage1, optimizer,
                                           graph_config={"node_dim": 4}, graph_stage=1)
        restored_projector, restored_stage1 = self._modules()
        restored_optimizer = torch.optim.AdamW(
            list(restored_projector.parameters()) + list(restored_stage1.parameters()), lr=0.1)
        restored = restore_graph_training_state(
            payload, restored_projector, restored_stage1, restored_optimizer)
        self.assertTrue(restored)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 1e-3)
        nodes, classes = restored_stage1.queue.snapshot(torch.device("cpu"))
        self.assertEqual(tuple(nodes.shape), (1, 4))
        self.assertEqual(classes.tolist(), [1])
        for expected, actual in zip(projector.parameters(), restored_projector.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_old_checkpoint_is_accepted_without_mutation(self):
        projector, stage1 = self._modules()
        optimizer = torch.optim.AdamW(list(projector.parameters()) + list(stage1.parameters()), lr=1e-3)
        before = [parameter.detach().clone() for parameter in projector.parameters()]
        restored = restore_graph_training_state({"format_version": 1}, projector, stage1, optimizer)
        self.assertFalse(restored)
        for expected, actual in zip(before, projector.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
