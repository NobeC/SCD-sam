import unittest
from unittest import mock

import torch
import torch.nn as nn

from utils.experiment_state import build_training_state, restore_training_state


def objects():
    student = nn.Linear(2, 2)
    teacher = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return student, teacher, optimizer, scheduler


def payload():
    student, teacher, optimizer, scheduler = objects()
    return build_training_state(
        student, teacher, optimizer, scheduler, epoch=3, optimizer_step=4,
        class_precision=torch.ones(2), best_metrics={},
        run_config={"dataset": "Landsat", "num_classes": 2, "seed": 42,
                    "mode": "pseudo"})


class ExperimentStateFiniteTest(unittest.TestCase):
    def test_supervised_checkpoint_initializes_requested_teacher_from_student(self):
        checkpoint = payload()
        checkpoint["teacher"] = None
        checkpoint["student"]["weight"].fill_(2.0)
        checkpoint["student"]["bias"].fill_(3.0)
        student, teacher, optimizer, scheduler = objects()
        with torch.no_grad():
            teacher.weight.fill_(-1.0)
            teacher.bias.fill_(-1.0)

        restore_training_state(
            checkpoint, student, teacher, optimizer, scheduler)

        for name, value in checkpoint["student"].items():
            self.assertTrue(torch.equal(student.state_dict()[name], value))
            self.assertTrue(torch.equal(teacher.state_dict()[name], value))

    def test_checkpoint_teacher_takes_precedence_over_student(self):
        checkpoint = payload()
        checkpoint["student"]["weight"].fill_(2.0)
        checkpoint["student"]["bias"].fill_(3.0)
        checkpoint["teacher"]["weight"].fill_(4.0)
        checkpoint["teacher"]["bias"].fill_(5.0)
        student, teacher, optimizer, scheduler = objects()

        restore_training_state(
            checkpoint, student, teacher, optimizer, scheduler)

        for name, value in checkpoint["teacher"].items():
            self.assertTrue(torch.equal(teacher.state_dict()[name], value))

    @mock.patch("utils.experiment_state.torch.cuda.set_rng_state_all")
    @mock.patch("utils.experiment_state.torch.cuda.device_count", return_value=1)
    @mock.patch("utils.experiment_state.torch.cuda.is_available", return_value=True)
    def test_cuda_rng_restore_truncates_states_to_visible_devices(
            self, _available, _device_count, set_rng_state_all):
        checkpoint = payload()
        checkpoint["rng_state"]["cuda"] = [
            torch.tensor([1], dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
        ]
        student, teacher, optimizer, scheduler = objects()

        restore_training_state(
            checkpoint, student, teacher, optimizer, scheduler)

        restored_states = set_rng_state_all.call_args.args[0]
        self.assertEqual(len(restored_states), 1)
        self.assertTrue(torch.equal(restored_states[0], checkpoint["rng_state"]["cuda"][0]))

    def test_nan_student_is_rejected_before_loading(self):
        checkpoint = payload()
        checkpoint["student"]["weight"][0, 0] = float("nan")
        student, teacher, optimizer, scheduler = objects()
        original = student.weight.detach().clone()

        with self.assertRaisesRegex(FloatingPointError, "student.*weight"):
            restore_training_state(checkpoint, student, teacher, optimizer, scheduler)

        self.assertTrue(torch.equal(student.weight, original))

    def test_inf_teacher_is_rejected_before_loading_student(self):
        checkpoint = payload()
        checkpoint["teacher"]["bias"][0] = float("inf")
        student, teacher, optimizer, scheduler = objects()
        original = student.weight.detach().clone()

        with self.assertRaisesRegex(FloatingPointError, "teacher.*bias"):
            restore_training_state(checkpoint, student, teacher, optimizer, scheduler)

        self.assertTrue(torch.equal(student.weight, original))


if __name__ == "__main__":
    unittest.main()
