import unittest

import torch
import torch.nn as nn

from models.Encoder.DFI import Attention


class DFIAttentionAMPStabilityTest(unittest.TestCase):
    def test_large_fp16_attention_inputs_remain_finite(self):
        attention = Attention().half()
        attention.query = nn.Identity()
        attention.key = nn.Identity()
        attention.value = nn.Identity()
        attention.out = nn.Identity()
        attention.attn_dropout = nn.Identity()
        attention.proj_dropout = nn.Identity()

        x1 = torch.full((1, 1, 768), 1000.0, dtype=torch.float16)
        x2 = torch.full((1, 1, 768), 1000.0, dtype=torch.float16)

        output = attention(x1, x2)

        self.assertEqual(output.shape, x1.shape)
        self.assertEqual(output.dtype, x1.dtype)
        self.assertTrue(torch.isfinite(output).all())

    def test_large_fp16_projection_outputs_remain_finite(self):
        attention = Attention().half()
        attention.out = nn.Identity()
        attention.attn_dropout = nn.Identity()
        attention.proj_dropout = nn.Identity()
        with torch.no_grad():
            attention.query.weight.fill_(100.0)
            attention.key.weight.fill_(100.0)
            attention.value.weight.fill_(1.0 / 768.0)
            attention.query.bias.zero_()
            attention.key.bias.zero_()
            attention.value.bias.zero_()

        x1 = torch.full((1, 1, 768), 1000.0, dtype=torch.float16)
        x2 = torch.full((1, 1, 768), 1000.0, dtype=torch.float16)

        output = attention(x1, x2)

        self.assertEqual(output.shape, x1.shape)
        self.assertEqual(output.dtype, x1.dtype)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
