import unittest

from utils.checkpoint_manager import updated_best_metrics


class BestMetricsTest(unittest.TestCase):
    def test_metrics_update_together_only_when_fscd_improves(self):
        previous = {"epoch": 90, "Fscd": 0.82, "mIoU": 0.84,
                    "Sek": 0.46, "accuracy": 0.93, "loss": 0.14,
                    "train_acc": 0.94}
        worse = {"Fscd": 0.81, "mIoU": 0.85, "Sek": 0.47,
                 "accuracy": 0.94, "loss": 0.13}
        self.assertIsNone(updated_best_metrics(previous, 91, worse, train_acc=0.95))

        better = {"Fscd": 0.8347, "mIoU": 0.8481, "Sek": 0.4842,
                  "accuracy": 0.9421, "loss": 0.13}
        self.assertEqual(updated_best_metrics(previous, 98, better, train_acc=0.96), {
            "epoch": 98, "Fscd": 0.8347, "mIoU": 0.8481,
            "Sek": 0.4842, "accuracy": 0.9421, "loss": 0.13,
            "train_acc": 0.96,
        })


if __name__ == "__main__":
    unittest.main()
