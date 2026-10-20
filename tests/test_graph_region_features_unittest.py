import unittest

import torch
import torch.nn as nn

from models.graph.region_features import (
    SamGraphFeatureProvider,
    mask_geometry,
    pool_mask_regions,
)


class _FiveOutputBackbone(nn.Module):
    def forward(self, image):
        batch = image.shape[0]
        base = image.mean(dim=1, keepdim=True)
        return (
            base.repeat(1, 2, 1, 1),
            base.repeat(1, 3, 1, 1),
            base.repeat(1, 4, 1, 1),
            base.repeat(1, 5, 1, 1),
            base.repeat(1, 6, 1, 1),
        )


class GraphRegionFeaturesTest(unittest.TestCase):
    def test_provider_selects_stage3_and_neck_and_detaches(self):
        image_a = torch.ones((2, 3, 4, 4), requires_grad=True)
        image_b = torch.ones((2, 3, 4, 4), requires_grad=True) * 2
        provider = SamGraphFeatureProvider(_FiveOutputBackbone())

        stage_a, neck_a, stage_b, neck_b = provider(image_a, image_b)

        self.assertEqual(tuple(stage_a.shape), (2, 4, 4, 4))
        self.assertEqual(tuple(neck_a.shape), (2, 6, 4, 4))
        self.assertEqual(float(stage_b.mean()), 2.0)
        self.assertFalse(stage_a.requires_grad)
        self.assertFalse(neck_b.requires_grad)

    def test_pooling_computes_exact_mean_max_and_invalid_mask(self):
        feature = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        masks = torch.zeros((2, 4, 4), dtype=torch.bool)
        masks[0, 0:2, :] = True
        masks[1, 1, 1] = True

        mean, maximum, valid = pool_mask_regions(feature, masks)

        self.assertEqual(valid.tolist(), [True, False])
        self.assertAlmostEqual(float(mean[0, 0]), 1.5)
        self.assertAlmostEqual(float(maximum[0, 0]), 2.0)
        self.assertEqual(mean.dtype, torch.float32)

    def test_empty_masks_return_empty_channel_tensors(self):
        feature = torch.ones((1, 3, 2, 2))
        mean, maximum, valid = pool_mask_regions(
            feature, torch.empty((0, 4, 4), dtype=torch.bool))
        self.assertEqual(tuple(mean.shape), (0, 3))
        self.assertEqual(tuple(maximum.shape), (0, 3))
        self.assertEqual(tuple(valid.shape), (0,))

    def test_geometry_is_normalized(self):
        masks = torch.zeros((1, 4, 8), dtype=torch.bool)
        masks[0, 1:3, 2:6] = True

        geometry = mask_geometry(masks)

        expected = torch.tensor([[3.5 / 7.0, 1.5 / 3.0, 4.0 / 8.0, 2.0 / 4.0, 8.0 / 32.0]])
        torch.testing.assert_close(geometry, expected)


if __name__ == "__main__":
    unittest.main()
