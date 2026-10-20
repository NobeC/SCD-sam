import unittest

import torch
import torch.nn as nn

from models.Decoder.FPN import FPNNeck


class RecordDtype(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dtype = None

    def forward(self, value):
        self.input_dtype = value.dtype
        return value


class FPNAMPStabilityTest(unittest.TestCase):
    def test_stage4_receives_fp32_input_inside_autocast(self):
        neck = FPNNeck(inplanes=1, neck_name="fpn")
        recorder = RecordDtype()
        neck.stage4_Conv1 = recorder
        fa4 = torch.randn(1, 8, 4, 4)
        fb4 = torch.randn(1, 8, 4, 4)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = neck._forward_stage4(fa4, fb4)

        self.assertEqual(recorder.input_dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
