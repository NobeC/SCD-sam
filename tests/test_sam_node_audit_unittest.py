import unittest

import torch

from utils.sam_node_audit import (
    GT,
    IGNORE,
    UNLABELED,
    SamNodeAuditConfig,
    audit_sam_nodes,
    summarize_sam_node_audits,
)


class SamNodeAuditTest(unittest.TestCase):
    def test_assigns_gt_per_time_and_unlabeled_nodes(self):
        labels_a = torch.zeros((6, 6), dtype=torch.long)
        labels_b = torch.zeros((6, 6), dtype=torch.long)
        labels_a[0:2, 0:2] = 1
        labels_b[0:2, 0:2] = 2
        masks = torch.zeros((2, 6, 6), dtype=torch.bool)
        masks[0, 0:2, 0:2] = True
        masks[1, 3:6, 3:6] = True
        config = SamNodeAuditConfig(
            tau_gt=0.9, tau_purity=0.9, tau_u=0.9,
            min_mask_pixels=2, min_changed_pixels=2,
        )

        audit = audit_sam_nodes(masks, labels_a, labels_b, 3, config)

        self.assertEqual(audit.states_a.tolist(), [GT, UNLABELED])
        self.assertEqual(audit.states_b.tolist(), [GT, UNLABELED])
        self.assertEqual(audit.classes_a.tolist(), [1, 0])
        self.assertEqual(audit.classes_b.tolist(), [2, 0])
        self.assertTrue(torch.isfinite(audit.change_ratio).all())

    def test_mixed_and_too_small_changed_regions_are_ignored(self):
        labels_a = torch.zeros((4, 4), dtype=torch.long)
        labels_b = torch.zeros((4, 4), dtype=torch.long)
        labels_a[0, 0] = 1
        labels_a[0, 1] = 2
        masks = torch.zeros((1, 4, 4), dtype=torch.bool)
        masks[0, 0:2, 0:2] = True

        audit = audit_sam_nodes(
            masks, labels_a, labels_b, 3,
            SamNodeAuditConfig(
                tau_gt=0.4, tau_purity=0.75, tau_u=0.9,
                min_mask_pixels=1, min_changed_pixels=1),
        )

        self.assertEqual(audit.states_a.item(), IGNORE)
        self.assertEqual(audit.states_b.item(), IGNORE)

    def test_empty_audit_and_summary_are_finite(self):
        labels = torch.zeros((4, 4), dtype=torch.long)
        audit = audit_sam_nodes(
            torch.empty((0, 4, 4), dtype=torch.bool), labels, labels, 5,
            SamNodeAuditConfig())
        summary = summarize_sam_node_audits([audit], 5)

        self.assertEqual(audit.states_a.numel(), 0)
        self.assertEqual(summary["total_shared_masks"], 0)
        self.assertEqual(summary["positive_gt_pairs"], 0)
        self.assertEqual(summary["negative_gt_pairs"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(summary["changed_coverage"])))

    def test_summary_counts_classes_and_pairs(self):
        labels_a = torch.zeros((4, 4), dtype=torch.long)
        labels_b = torch.zeros((4, 4), dtype=torch.long)
        masks = torch.zeros((3, 4, 4), dtype=torch.bool)
        for index, (row, col, cls) in enumerate(((0, 0, 1), (0, 2, 1), (2, 0, 2))):
            masks[index, row:row + 2, col:col + 2] = True
            labels_a[row:row + 2, col:col + 2] = cls
            labels_b[row:row + 2, col:col + 2] = cls
        audit = audit_sam_nodes(
            masks, labels_a, labels_b, 3,
            SamNodeAuditConfig(
                tau_gt=0.9, tau_purity=0.9, tau_u=0.9,
                min_mask_pixels=1, min_changed_pixels=1),
        )

        summary = summarize_sam_node_audits([audit], 3)

        self.assertEqual(summary["gt_nodes_per_class"], [0, 4, 2])
        self.assertEqual(summary["positive_gt_pairs"], 7)
        self.assertEqual(summary["negative_gt_pairs"], 8)

    def test_invalid_label_id_is_rejected(self):
        labels = torch.zeros((2, 2), dtype=torch.long)
        labels[0, 0] = 5
        with self.assertRaisesRegex(ValueError, "label"):
            audit_sam_nodes(
                torch.ones((1, 2, 2), dtype=torch.bool), labels, labels, 5,
                SamNodeAuditConfig())


if __name__ == "__main__":
    unittest.main()
