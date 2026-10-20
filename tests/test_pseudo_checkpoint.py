import unittest
from collections import OrderedDict

import torch

from utils.pseudo_checkpoint import (
    infer_semantic_num_classes,
    select_model_state,
    validate_checkpoint_num_classes,
)


def state_dict(num_classes=5, prefix=""):
    return OrderedDict({
        prefix + "backbone.weight": torch.ones(1),
        prefix + "head.head1.2.weight": torch.ones(num_classes, 96, 1, 1),
        prefix + "head.head2.2.weight": torch.ones(num_classes, 96, 1, 1),
    })


class PseudoCheckpointTest(unittest.TestCase):
    def test_selects_requested_teacher_from_training_state(self):
        teacher = state_dict()
        selected, owner = select_model_state(
            {"student": state_dict(), "teacher": teacher}, "teacher")

        self.assertEqual(list(selected), list(teacher))
        self.assertTrue(torch.equal(
            selected["head.head1.2.weight"],
            teacher["head.head1.2.weight"]))
        self.assertEqual(owner, "teacher")

    def test_selects_requested_student_from_training_state(self):
        student = state_dict()
        selected, owner = select_model_state(
            {"student": student, "teacher": state_dict()}, "student")

        self.assertEqual(list(selected), list(student))
        self.assertTrue(torch.equal(
            selected["head.head1.2.weight"],
            student["head.head1.2.weight"]))
        self.assertEqual(owner, "student")

    def test_rejects_missing_teacher(self):
        with self.assertRaisesRegex(ValueError, "does not contain an EMA teacher"):
            select_model_state({"student": state_dict(), "teacher": None}, "teacher")

    def test_accepts_legacy_plain_and_state_dict_checkpoints(self):
        plain = state_dict(prefix="module.")
        selected_plain, owner_plain = select_model_state(plain, "teacher")
        selected_wrapped, owner_wrapped = select_model_state(
            {"state_dict": plain}, "teacher")

        self.assertEqual(owner_plain, "legacy")
        self.assertEqual(owner_wrapped, "state_dict")
        self.assertNotIn("module.head.head1.2.weight", selected_plain)
        self.assertIn("head.head1.2.weight", selected_plain)
        self.assertIn("head.head1.2.weight", selected_wrapped)

    def test_infers_and_validates_semantic_class_count(self):
        selected, _ = select_model_state(state_dict(num_classes=7), "teacher")

        self.assertEqual(infer_semantic_num_classes(selected), 7)
        validate_checkpoint_num_classes(selected, 7, "SECOND")
        with self.assertRaisesRegex(ValueError, "7 semantic classes.*Landsat expects 5"):
            validate_checkpoint_num_classes(selected, 5, "Landsat")


if __name__ == "__main__":
    unittest.main()
