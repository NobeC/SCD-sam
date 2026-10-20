import unittest
from unittest import mock

import torch

from utils.training_performance import (
    IntervalTimer,
    TrainingMetricAccumulator,
    change_accuracy,
    configure_cudnn,
    is_sample_step,
    loader_worker_options,
)


class TrainingPerformanceTest(unittest.TestCase):
    def test_zero_workers_omits_multiprocessing_only_options(self):
        self.assertEqual(loader_worker_options(0), {})

    def test_positive_workers_enable_persistence_and_prefetch(self):
        self.assertEqual(
            loader_worker_options(4),
            {"persistent_workers": True, "prefetch_factor": 2},
        )

    def test_positive_workers_can_disable_persistence(self):
        self.assertEqual(
            loader_worker_options(4, persistent_workers=False),
            {"persistent_workers": False, "prefetch_factor": 2},
        )

    def test_cudnn_throughput_mode_enables_benchmark(self):
        original = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            state = configure_cudnn(deterministic=False)
            self.assertEqual(state, {"benchmark": True, "deterministic": False})
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original

    def test_cudnn_deterministic_mode_disables_benchmark(self):
        original = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            state = configure_cudnn(deterministic=True)
            self.assertEqual(state, {"benchmark": False, "deterministic": True})
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original

    def test_interval_timer_rejects_negative_interval(self):
        with self.assertRaisesRegex(ValueError, "interval"):
            IntervalTimer(-1)

    @mock.patch("utils.training_performance.time.perf_counter")
    def test_interval_timer_samples_without_syncing_unsampled_steps(self, clock):
        clock.side_effect = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0]
        synchronize = mock.Mock()
        timer = IntervalTimer(interval=2)

        timer.start_data()
        timer.start_compute()
        self.assertIsNone(timer.finish_step(1, synchronize=synchronize))
        synchronize.assert_not_called()

        timer.start_data()
        timer.start_compute()
        sample = timer.finish_step(2, synchronize=synchronize)

        synchronize.assert_called_once_with()
        self.assertAlmostEqual(sample["data_seconds"], 0.3)
        self.assertAlmostEqual(sample["compute_seconds"], 0.45)
        self.assertAlmostEqual(sample["step_seconds"], 0.75)

    def test_negative_workers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_workers"):
            loader_worker_options(-1)

    def test_change_accuracy_matches_per_sample_legacy_mean(self):
        outputs_a = torch.tensor([
            [[[0.0, 0.0]], [[3.0, 3.0]], [[1.0, 1.0]]],
            [[[0.0, 0.0]], [[1.0, 1.0]], [[3.0, 3.0]]],
        ])
        outputs_b = outputs_a.clone()
        change_logits = torch.tensor([[[[2.0, -2.0]]], [[[2.0, 2.0]]]])
        labels_a = torch.tensor([[[1, 0]], [[2, 1]]])
        labels_b = torch.tensor([[[1, 0]], [[2, 2]]])

        actual = change_accuracy(
            outputs_a, outputs_b, change_logits, labels_a, labels_b)

        # Sample accuracies are A/B=(1,1) and A/B=(1/2,1), averaged equally.
        self.assertAlmostEqual(float(actual), 0.875)

    def test_metric_accumulator_preserves_last_and_exact_average(self):
        metrics = TrainingMetricAccumulator(("loss", "accuracy"), device="cpu")
        metrics.update(loss=torch.tensor(2.0), accuracy=torch.tensor(0.5))
        metrics.update(loss=torch.tensor(4.0), accuracy=torch.tensor(1.0))

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["loss"], {"last": 4.0, "average": 3.0})
        self.assertEqual(snapshot["accuracy"], {"last": 1.0, "average": 0.75})

    def test_metric_accumulator_rejects_missing_values(self):
        metrics = TrainingMetricAccumulator(("loss", "accuracy"), device="cpu")
        with self.assertRaisesRegex(ValueError, "metric names"):
            metrics.update(loss=torch.tensor(1.0))

    def test_sampling_cadence_includes_first_and_periodic_steps(self):
        self.assertTrue(is_sample_step(1, 50))
        self.assertFalse(is_sample_step(2, 50))
        self.assertTrue(is_sample_step(50, 50))
        self.assertFalse(is_sample_step(51, 50))

    def test_zero_sampling_interval_disables_all_steps(self):
        self.assertFalse(is_sample_step(1, 0))
        self.assertFalse(is_sample_step(200, 0))

    def test_invalid_sampling_cadence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interval"):
            is_sample_step(1, -1)
        with self.assertRaisesRegex(ValueError, "iteration"):
            is_sample_step(0, 10)


if __name__ == "__main__":
    unittest.main()
