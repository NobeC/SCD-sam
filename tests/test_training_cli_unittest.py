import argparse
import subprocess
import sys
import unittest

from utils.training_performance import add_performance_arguments


class TrainingCliTest(unittest.TestCase):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        add_performance_arguments(parser)
        return parser

    def test_throughput_defaults(self):
        options = self.build_parser().parse_args([])
        self.assertFalse(options.gradient_checkpointing)
        self.assertFalse(options.deterministic)
        self.assertFalse(options.persistent_workers)
        self.assertEqual(options.timing_interval, 50)

    def test_reproducible_and_memory_saving_overrides(self):
        options = self.build_parser().parse_args([
            "--gradient-checkpointing",
            "--deterministic",
            "--persistent-workers",
            "--timing-interval", "25",
        ])
        self.assertTrue(options.gradient_checkpointing)
        self.assertTrue(options.deterministic)
        self.assertTrue(options.persistent_workers)
        self.assertEqual(options.timing_interval, 25)

    def test_negative_timing_interval_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.build_parser().parse_args(["--timing-interval", "-1"])

    def test_train_help_documents_performance_switches(self):
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for option in (
            "--gradient-checkpointing",
            "--cudnn-benchmark",
            "--deterministic",
            "--persistent-workers",
            "--timing-interval",
        ):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
