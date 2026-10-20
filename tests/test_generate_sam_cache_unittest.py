import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_sam_cache.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_sam_cache", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerateSamCacheTest(unittest.TestCase):
    def test_pair_image_paths_is_sorted_and_limited(self):
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dir_a, dir_b = root / "A", root / "B"
            dir_a.mkdir(); dir_b.mkdir()
            for name in ("z.png", "a.png", "ignored.txt"):
                (dir_a / name).write_bytes(b"x")
            for name in ("a.png", "z.png"):
                (dir_b / name).write_bytes(b"x")

            pairs = module.pair_image_paths(dir_a, dir_b, limit=1)

            self.assertEqual([(a.name, b.name) for a, b in pairs], [("a.png", "a.png")])

    def test_pair_image_paths_reports_missing_counterpart(self):
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dir_a, dir_b = root / "A", root / "B"
            dir_a.mkdir(); dir_b.mkdir()
            (dir_a / "only_a.png").write_bytes(b"x")

            with self.assertRaisesRegex(ValueError, "only_a.png"):
                module.pair_image_paths(dir_a, dir_b)


if __name__ == "__main__":
    unittest.main()
