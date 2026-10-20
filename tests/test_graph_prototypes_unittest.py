import unittest

import torch

from models.graph.prototypes import EMAPrototypeBank


class GraphPrototypeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 autocast")
    def test_logits_and_loss_remain_fp32_under_cuda_autocast(self):
        bank = EMAPrototypeBank(num_classes=3, node_dim=4).cuda()
        nodes = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device="cuda", requires_grad=True)
        classes = torch.tensor([1, 2], device="cuda")
        bank.update(nodes, classes)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = bank.logits(nodes)
            loss = bank.classification_loss(nodes, classes)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(nodes.grad).all())

    def test_class_zero_is_never_initialized(self):
        bank = EMAPrototypeBank(num_classes=5, node_dim=4, momentum=0.9)
        nodes = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        classes = torch.tensor([0, 1])

        bank.update(nodes, classes)

        self.assertFalse(bool(bank.initialized[0]))
        self.assertTrue(bool(bank.initialized[1]))
        torch.testing.assert_close(bank.prototypes[0], torch.zeros(4))

    def test_slots_follow_dynamic_dataset_class_count(self):
        self.assertEqual(EMAPrototypeBank(5, 8).prototypes.shape, (5, 8))
        self.assertEqual(EMAPrototypeBank(7, 8).prototypes.shape, (7, 8))

    def test_update_normalizes_batch_and_ema_prototype(self):
        bank = EMAPrototypeBank(num_classes=3, node_dim=2, momentum=0.5)
        bank.update(torch.tensor([[2.0, 0.0]]), torch.tensor([1]))
        bank.update(torch.tensor([[0.0, 3.0]]), torch.tensor([1]))
        expected = torch.tensor([2 ** -0.5, 2 ** -0.5])
        torch.testing.assert_close(bank.prototypes[1], expected)
        self.assertEqual(bank.prototypes.dtype, torch.float32)

    def test_logits_mask_zero_and_uninitialized_classes(self):
        bank = EMAPrototypeBank(num_classes=4, node_dim=2, temperature=0.5)
        bank.update(torch.tensor([[1.0, 0.0]]), torch.tensor([2]))

        logits = bank.logits(torch.tensor([[1.0, 0.0]]))

        minimum = torch.finfo(torch.float32).min
        self.assertEqual(logits.shape, (1, 4))
        self.assertEqual(float(logits[0, 0]), minimum)
        self.assertEqual(float(logits[0, 1]), minimum)
        self.assertAlmostEqual(float(logits[0, 2]), 2.0)

    def test_classification_loss_is_finite_and_connected(self):
        bank = EMAPrototypeBank(num_classes=3, node_dim=2)
        bank.update(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([1, 2]))
        nodes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

        loss = bank.classification_loss(nodes, torch.tensor([1, 2]))
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(nodes.grad)

    def test_empty_or_uninitialized_loss_is_connected_zero(self):
        bank = EMAPrototypeBank(num_classes=3, node_dim=2)
        nodes = torch.empty((0, 2), requires_grad=True)
        loss = bank.classification_loss(nodes, torch.empty((0,), dtype=torch.long))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(nodes.grad)


if __name__ == "__main__":
    unittest.main()
