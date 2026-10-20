import unittest

from tools.prepare_hrscd import make_split_indices, tile_starts


class PrepareHrscdTests(unittest.TestCase):
    def test_make_split_indices_is_deterministic_and_disjoint(self):
        first = make_split_indices(291, train=117, val=29, test=145, seed=42)
        second = make_split_indices(291, train=117, val=29, test=145, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(values) for name, values in first.items()},
            {"train": 117, "val": 29, "test": 145},
        )
        self.assertTrue(set(first["train"]).isdisjoint(first["val"]))
        self.assertTrue(set(first["train"]).isdisjoint(first["test"]))
        self.assertTrue(set(first["val"]).isdisjoint(first["test"]))
        self.assertEqual(set().union(*map(set, first.values())), set(range(291)))


    def test_tile_starts_covers_non_divisible_edge(self):
        starts = tile_starts(10000, 512)

        self.assertEqual(len(starts), 20)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 9488)
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))


    def test_tile_starts_rejects_image_smaller_than_tile(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            tile_starts(511, 512)


if __name__ == "__main__":
    unittest.main()
