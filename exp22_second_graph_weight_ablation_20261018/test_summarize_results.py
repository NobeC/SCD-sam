import tempfile
import unittest
from pathlib import Path

import summarize_results


SAMPLE = """
[epoch 65] [iter 1150 / 1188 1.0s] [lr 0.000010]
1.0s Val loss: 0.61 Fscd: 60.10 IoU: 71.20 Sek: 20.30 Accuracy: 87.10
[Warning] non-finite gradient at epoch 66 iter 10; optimizer and EMA updates skipped.
[epoch 66] [iter 1150 / 1188 1.0s] [lr 0.000009]
1.0s Val loss: 0.60 Fscd: 60.50 IoU: 71.70 Sek: 20.80 Accuracy: 87.30
[epoch 67] [iter 50 / 1188 1.0s] [lr 0.000008]
"""


class ResultSummaryTests(unittest.TestCase):
    def test_ignores_incomplete_final_epoch_and_counts_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(SAMPLE)
            result = summarize_results.parse_log(path)

        self.assertEqual([row["epoch"] for row in result["epochs"]], [65, 66])
        self.assertEqual(result["nonfinite_skips"], 1)
        self.assertEqual(result["best_fscd"]["epoch"], 66)
        self.assertAlmostEqual(result["best_fscd"]["fscd"], 60.5)

    def test_comparison_reports_tail_means(self):
        parsed = {
            "epochs": [
                {"epoch": 65, "fscd": 60.0, "sek": 20.0},
                {"epoch": 66, "fscd": 61.0, "sek": 22.0},
            ],
            "nonfinite_skips": 0,
        }
        summary = summarize_results.tail_summary(parsed, 65, 69)
        self.assertEqual(summary["completed_epochs"], [65, 66])
        self.assertAlmostEqual(summary["mean_fscd"], 60.5)
        self.assertAlmostEqual(summary["mean_sek"], 21.0)


if __name__ == "__main__":
    unittest.main()
