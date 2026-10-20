import unittest

import numpy as np

from utils.sam_mask_augmentation import rand_rot90_flip_SCD_masks


class SamMaskAugmentationModuleTest(unittest.TestCase):
    def test_rotation_and_flip_keep_every_input_aligned(self):
        values = np.arange(12).reshape(3, 4)
        image_a = np.repeat(values[:, :, None], 3, axis=2)
        image_b = image_a + 20
        labels_a = values.copy()
        labels_b = values + 40
        masks = np.stack([values == 1, values == 10])

        image_a, image_b, labels_a, labels_b, masks = rand_rot90_flip_SCD_masks(
            image_a, image_b, labels_a, labels_b, masks,
            rotate=True, flip="horizontal",
        )

        expected = np.flip(np.rot90(values), axis=1)
        np.testing.assert_array_equal(image_a[:, :, 0], expected)
        np.testing.assert_array_equal(image_b[:, :, 0] - 20, expected)
        np.testing.assert_array_equal(labels_a, expected)
        np.testing.assert_array_equal(labels_b - 40, expected)
        np.testing.assert_array_equal(masks[0], expected == 1)
        self.assertEqual(masks.dtype, np.bool_)

    def test_empty_masks_follow_rotated_shape(self):
        values = np.arange(12).reshape(3, 4)
        image = np.repeat(values[:, :, None], 3, axis=2)
        empty = np.empty((0, 3, 4), dtype=np.bool_)
        *_, masks = rand_rot90_flip_SCD_masks(
            image, image, values, values, empty, rotate=True, flip="none")
        self.assertEqual(masks.shape, (0, 4, 3))

    def test_invalid_flip_is_rejected(self):
        values = np.arange(12).reshape(3, 4)
        image = np.repeat(values[:, :, None], 3, axis=2)
        with self.assertRaisesRegex(ValueError, "flip"):
            rand_rot90_flip_SCD_masks(
                image, image, values, values,
                np.empty((0, 3, 4), dtype=np.bool_),
                rotate=False, flip="diagonal")


if __name__ == "__main__":
    unittest.main()
