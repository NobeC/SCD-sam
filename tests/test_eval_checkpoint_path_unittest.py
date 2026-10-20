import tempfile
import unittest
from pathlib import Path

from utils.eval_checkpoint import resolve_checkpoint_path


class EvalCheckpointPathTests(unittest.TestCase):
    def test_resolves_relative_path_from_project_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "checkpoints" / "Landsat" / "model.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            resolved = resolve_checkpoint_path(
                "checkpoints/Landsat/model.pth", tmp_dir
            )

            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_preserves_existing_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model.pth"
            checkpoint.touch()

            resolved = resolve_checkpoint_path(str(checkpoint), "/unused")

            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_missing_checkpoint_reports_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expected = Path(tmp_dir) / "checkpoints" / "missing.pth"

            with self.assertRaisesRegex(FileNotFoundError, str(expected)):
                resolve_checkpoint_path("checkpoints/missing.pth", tmp_dir)


if __name__ == "__main__":
    unittest.main()
