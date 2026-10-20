import unittest

import torch

from graph_pseudo_complement import complement_graph_targets


class ComplementaryPseudoLabelTests(unittest.TestCase):
    def test_graph_fills_semantic_gaps_and_semantic_wins_overlap(self):
        ignore = 255
        semantic = torch.tensor([[[[1, ignore, 2, ignore]],
                                  [[ignore, ignore, 3, ignore]]]])
        graph = torch.tensor([[[3, 4, 2, 5]]])
        graph_valid = torch.tensor([[[True, True, True, False]]])

        targets, valid, overlap, conflict = complement_graph_targets(
            graph, graph_valid, semantic, ignore_index=ignore
        )

        self.assertEqual(valid.tolist(), [[[False, True, False, False]]])
        self.assertEqual(targets.tolist(), [[[ignore, 4, ignore, ignore]]])
        self.assertEqual(int(overlap.sum()), 2)
        self.assertEqual(int(conflict.sum()), 2)

    def test_empty_semantic_labels_preserve_graph_coverage(self):
        ignore = 255
        semantic = torch.full((1, 2, 2, 2), ignore)
        graph = torch.tensor([[[1, 2], [3, 4]]])
        graph_valid = torch.ones_like(graph, dtype=torch.bool)

        targets, valid, overlap, conflict = complement_graph_targets(
            graph, graph_valid, semantic, ignore_index=ignore
        )

        self.assertTrue(torch.equal(targets, graph))
        self.assertTrue(torch.equal(valid, graph_valid))
        self.assertEqual(int(overlap.sum()), 0)
        self.assertEqual(int(conflict.sum()), 0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            complement_graph_targets(
                torch.zeros(1, 2, 2, dtype=torch.long),
                torch.ones(1, 2, 2, dtype=torch.bool),
                torch.full((1, 1, 3, 2), 255),
            )


if __name__ == "__main__":
    unittest.main()
