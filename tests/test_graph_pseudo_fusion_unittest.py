import unittest

import torch

from utils.sam_node_audit import GT, IGNORE, UNLABELED


class GraphPseudoFusionTest(unittest.TestCase):
    def test_overlap_prefers_confidence_then_smaller_area_then_index(self):
        from models.graph.pseudo_fusion import hard_reproject_u_nodes

        masks = torch.tensor([
            [[1, 1], [1, 1]],
            [[1, 1], [0, 0]],
            [[1, 1], [0, 0]],
        ], dtype=torch.bool)
        result = hard_reproject_u_nodes(
            masks=masks,
            states=torch.tensor([UNLABELED, UNLABELED, UNLABELED]),
            node_pred=torch.tensor([1, 2, 3]),
            node_conf=torch.tensor([0.9, 0.9, 0.9]),
            node_valid=torch.tensor([True, True, True]),
        )

        self.assertEqual(result.pred.tolist(), [[2, 2], [1, 1]])
        self.assertTrue(result.valid.all())

    def test_gt_and_ignore_nodes_never_reproject(self):
        from models.graph.pseudo_fusion import hard_reproject_u_nodes

        masks = torch.ones(2, 2, 2, dtype=torch.bool)
        result = hard_reproject_u_nodes(
            masks,
            torch.tensor([GT, IGNORE]),
            torch.tensor([1, 2]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([True, True]),
        )
        self.assertFalse(result.valid.any())

    def _teacher_probs(self, confidence=0.95, cls=1):
        probs = torch.full((1, 3, 2, 2), (1.0 - confidence) / 2.0)
        probs[:, cls] = confidence
        return probs

    def _veto(self, **overrides):
        from models.graph.pseudo_fusion import teacher_graph_veto

        kwargs = dict(
            teacher_probs_a=self._teacher_probs(),
            teacher_probs_b=self._teacher_probs(),
            teacher_change_logits=torch.full((1, 1, 2, 2), -10.0),
            labels_a=torch.zeros(1, 2, 2, dtype=torch.long),
            labels_b=torch.zeros(1, 2, 2, dtype=torch.long),
            graph_pred_a=torch.ones(1, 2, 2, dtype=torch.long),
            graph_conf_a=torch.ones(1, 2, 2),
            graph_valid_a=torch.ones(1, 2, 2, dtype=torch.bool),
            graph_pred_b=torch.ones(1, 2, 2, dtype=torch.long),
            graph_conf_b=torch.ones(1, 2, 2),
            graph_valid_b=torch.ones(1, 2, 2, dtype=torch.bool),
            teacher_threshold=0.8,
            teacher_fallback_threshold=0.9,
            graph_confidence_threshold=0.7,
            change_threshold=0.3,
            num_classes=3,
        )
        kwargs.update(overrides)
        return teacher_graph_veto(**kwargs)

    def test_teacher_graph_agreement_is_accepted(self):
        result = self._veto()
        self.assertTrue(result.valid.all())
        self.assertTrue((result.target[result.valid] == 1).all())
        self.assertEqual(result.veto_count, 0)

    def test_graph_conflict_vetoes_teacher_candidate(self):
        result = self._veto(graph_pred_a=torch.full((1, 2, 2), 2, dtype=torch.long))
        self.assertFalse(result.valid.any())
        self.assertEqual(result.veto_count, 4)

    def test_graph_invalid_uses_stricter_teacher_fallback(self):
        invalid = torch.zeros(1, 2, 2, dtype=torch.bool)
        accepted = self._veto(graph_valid_a=invalid, graph_valid_b=invalid)
        rejected = self._veto(
            teacher_probs_a=self._teacher_probs(0.85),
            teacher_probs_b=self._teacher_probs(0.85),
            graph_valid_a=invalid,
            graph_valid_b=invalid,
        )
        self.assertTrue(accepted.valid.all())
        self.assertEqual(accepted.fallback_count, 4)
        self.assertFalse(rejected.valid.any())

    def test_changed_pixels_never_receive_pseudo_labels(self):
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        labels[0, 0, 0] = 1
        result = self._veto(labels_a=labels)
        self.assertFalse(result.valid[0, 0, 0])
        self.assertEqual(int((result.valid & ((labels > 0))).sum()), 0)

    def test_no_accepted_pseudo_returns_connected_zero_loss(self):
        from models.graph.pseudo_fusion import graph_pixel_pseudo_loss

        logits_a = torch.randn(1, 3, 2, 2, requires_grad=True)
        logits_b = torch.randn(1, 3, 2, 2, requires_grad=True)
        loss = graph_pixel_pseudo_loss(
            logits_a,
            logits_b,
            torch.full((1, 2, 2), 255, dtype=torch.long),
            torch.zeros(1, 2, 2, dtype=torch.bool),
        )
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(logits_a.grad)
        self.assertIsNotNone(logits_b.grad)

    def test_pseudo_loss_has_nonzero_student_logit_gradient(self):
        from models.graph.pseudo_fusion import graph_pixel_pseudo_loss

        logits_a = torch.zeros(1, 3, 2, 2, requires_grad=True)
        logits_b = torch.zeros(1, 3, 2, 2, requires_grad=True)
        target = torch.ones(1, 2, 2, dtype=torch.long)
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        loss = graph_pixel_pseudo_loss(logits_a, logits_b, target, valid)
        loss.backward()
        self.assertGreater(float(logits_a.grad.abs().sum()), 0.0)
        self.assertGreater(float(logits_b.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
