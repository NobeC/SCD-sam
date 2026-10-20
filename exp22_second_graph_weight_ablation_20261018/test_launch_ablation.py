import unittest

import launch_ablation


def normalized_weight(arguments):
    result = list(arguments)
    index = result.index("--graph-pixel-pseudo-weight") + 1
    result[index] = "<WEIGHT>"
    return result


class AblationLauncherTests(unittest.TestCase):
    def test_rejects_unknown_branch(self):
        with self.assertRaisesRegex(ValueError, "unknown branch"):
            launch_ablation.validate_branch("other", 0.02)

    def test_rejects_negative_weight(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            launch_ablation.validate_branch("graph02", -0.01)

    def test_branch_commands_only_differ_by_weight(self):
        control = launch_ablation.build_arguments("control", 0.0)
        graph02 = launch_ablation.build_arguments("graph02", 0.02)

        self.assertEqual(normalized_weight(control), normalized_weight(graph02))
        self.assertIn("--resume", control)
        self.assertIn("--epochs", control)
        self.assertEqual(control[control.index("--epochs") + 1], "70")
        self.assertEqual(control[control.index("--dataset") + 1], "second")
        self.assertEqual(control[control.index("--num-classes") + 1], "7")
        self.assertEqual(
            control[control.index("--graph-pseudo-semantic-mode") + 1],
            "complement",
        )

    def test_smoke_steps_are_opt_in(self):
        production = launch_ablation.build_arguments("graph02", 0.02)
        smoke = launch_ablation.build_arguments("graph02", 0.02, smoke_steps=1)

        self.assertNotIn("--smoke-steps", production)
        self.assertEqual(smoke[smoke.index("--smoke-steps") + 1], "1")


if __name__ == "__main__":
    unittest.main()
