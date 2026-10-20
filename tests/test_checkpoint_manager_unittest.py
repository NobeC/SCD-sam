from pathlib import Path
import tempfile
import unittest

from utils.checkpoint_manager import checkpoint_filename, configuration_id, save_best_checkpoint


def config(lam=0.2):
    return {"dataset": "Landsat", "mode": "pseudo", "seed": 42,
            "lambda_sem_u": lam, "pseudo_alpha_start": 0.1,
            "pseudo_alpha_end": 0.3, "pseudo_min_ratio": 0.02,
            "pseudo_change_thr": 0.3, "pseudo_start_epoch": 30,
            "pseudo_rampup_epoch": 10, "ema_decay": 0.999}


def metrics(fscd=0.8347):
    return {"mIoU": 0.8481, "Sek": 0.4842, "Fscd": fscd,
            "accuracy": 0.9421, "loss": 0.13}


def write_payload(payload, path):
    Path(path).write_text(str(payload))


class CheckpointManagerTest(unittest.TestCase):
    def test_identity_and_filename(self):
        identity = configuration_id(config())
        self.assertEqual(identity, "Landsat_pseudo_lam0.2_a0.1-0.3_min0.02_thr0.3_start30_ramp10_ema0.999_seed42")
        self.assertTrue(checkpoint_filename(config(), 98, metrics()).endswith(
            "_e098_mIoU84.81_Sek48.42_Fscd83.47_OA94.21.pth"))

    def test_replacement_is_isolated_and_failure_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            old = save_best_checkpoint(path, config(), 90, metrics(0.82), {}, save_fn=write_payload)
            other = save_best_checkpoint(path, config(0.3), 90, metrics(0.81), {}, save_fn=write_payload)
            new = save_best_checkpoint(path, config(), 98, metrics(), {}, save_fn=write_payload)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertTrue(other.exists())

            def fail_save(payload, output_path):
                raise OSError("disk full")

            with self.assertRaisesRegex(OSError, "disk full"):
                save_best_checkpoint(path, config(), 99, metrics(0.84), {}, save_fn=fail_save)
            self.assertTrue(new.exists())
            self.assertEqual(list(path.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
