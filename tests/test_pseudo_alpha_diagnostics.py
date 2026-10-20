import unittest

import torch

from utils.pseudo_label import (
    accumulate_pseudo_alpha_diagnostics,
    create_pseudo_alpha_diagnostics,
    pseudo_alpha_diagnostic_rows,
)


class PseudoAlphaDiagnosticsTest(unittest.TestCase):
    def test_low_entropy_selection_reports_valid_precision_and_unlabeled_coverage(self):
        logits = torch.tensor(
            [[
                [[0.0, 0.0, 0.0, 0.0]],
                [[5.0, 1.0, 4.0, 0.5]],
                [[0.0, 0.0, 0.0, 0.0]],
            ]]
        )
        labels = torch.tensor([[[1, 2, 0, 0]]])
        measured_mask = torch.tensor([[[True, True, False, False]]])
        unlabeled_mask = torch.tensor([[[False, False, True, True]]])

        state = create_pseudo_alpha_diagnostics(
            num_classes=3, ratios=(0.5, 1.0), device=logits.device)
        accumulate_pseudo_alpha_diagnostics(
            state, logits, labels, measured_mask, unlabeled_mask)
        rows = pseudo_alpha_diagnostic_rows(state)

        class_one = [row for row in rows if row["class_id"] == 1]
        half = [row for row in class_one if row["ratio"] == 0.5][0]
        full = [row for row in class_one if row["ratio"] == 1.0][0]

        self.assertEqual(half["measured_accepted"], 1)
        self.assertEqual(half["measured_tp"], 1)
        self.assertEqual(half["measured_fp"], 0)
        self.assertEqual(half["measured_precision"], 1.0)
        self.assertEqual(half["unlabeled_accepted"], 1)
        self.assertEqual(half["unlabeled_candidate"], 2)
        self.assertEqual(half["unlabeled_coverage"], 0.5)

        self.assertEqual(full["measured_accepted"], 2)
        self.assertEqual(full["measured_tp"], 1)
        self.assertEqual(full["measured_fp"], 1)
        self.assertEqual(full["measured_precision"], 0.5)
        self.assertEqual(full["unlabeled_coverage"], 1.0)

    def test_unlabeled_pixels_do_not_contribute_to_measured_precision(self):
        logits = torch.tensor(
            [[
                [[0.0, 0.0]],
                [[4.0, 4.0]],
                [[0.0, 0.0]],
            ]]
        )
        labels = torch.zeros((1, 1, 2), dtype=torch.long)
        measured_mask = torch.zeros((1, 1, 2), dtype=torch.bool)
        unlabeled_mask = torch.ones((1, 1, 2), dtype=torch.bool)

        state = create_pseudo_alpha_diagnostics(
            num_classes=3, ratios=(1.0,), device=logits.device)
        accumulate_pseudo_alpha_diagnostics(
            state, logits, labels, measured_mask, unlabeled_mask)
        row = [
            item for item in pseudo_alpha_diagnostic_rows(state)
            if item["class_id"] == 1
        ][0]

        self.assertEqual(row["measured_accepted"], 0)
        self.assertIsNone(row["measured_precision"])
        self.assertEqual(row["unlabeled_accepted"], 2)
        self.assertEqual(row["unlabeled_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
