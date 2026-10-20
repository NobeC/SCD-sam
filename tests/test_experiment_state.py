import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.experiment_state import build_training_state, restore_training_state


class TinyClassifier(nn.Linear):
    def __init__(self, num_classes):
        super().__init__(2, num_classes)
        self.num_classes = num_classes


def _objects(num_classes=3, teacher=True):
    student = TinyClassifier(num_classes)
    ema_teacher = TinyClassifier(num_classes) if teacher else None
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return student, ema_teacher, optimizer, scheduler


def test_checkpoint_round_trip_restores_teacher_and_pseudo_state():
    student, teacher, optimizer, scheduler = _objects()
    precision = torch.tensor([0.0, 0.7, 0.4])
    optimizer.zero_grad()
    student(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()

    payload = build_training_state(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=40,
        optimizer_step=123,
        class_precision=precision,
        best_metrics={"Fscd": 0.7, "Sek": 0.3},
        run_config={
            "dataset": "SECOND",
            "num_classes": 3,
            "seed": 1,
            "mode": "pseudo",
            "pseudo_start_epoch": 30,
        },
    )

    new_student, new_teacher, new_optimizer, new_scheduler = _objects()
    resumed = restore_training_state(
        payload,
        new_student,
        new_teacher,
        new_optimizer,
        new_scheduler,
        active_run_config={"dataset": "SECOND", "num_classes": 3},
    )

    assert resumed.epoch == 40
    assert resumed.optimizer_step == 123
    assert torch.equal(resumed.class_precision, precision)
    assert resumed.run_config["dataset"] == "SECOND"
    assert resumed.best_metrics == {"Fscd": 0.7, "Sek": 0.3}
    assert new_student.state_dict().keys() == student.state_dict().keys()
    assert all(
        torch.equal(new_student.state_dict()[key], value)
        for key, value in student.state_dict().items()
    )
    assert all(
        torch.equal(new_teacher.state_dict()[key], value)
        for key, value in teacher.state_dict().items()
    )
    assert new_optimizer.state_dict()["state"]
    assert new_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_restore_rejects_dataset_or_class_mismatch_before_loading_weights():
    student, teacher, optimizer, scheduler = _objects()
    payload = build_training_state(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        optimizer_step=2,
        class_precision=torch.ones(3),
        best_metrics={},
        run_config={"dataset": "SECOND", "num_classes": 3, "seed": 1, "mode": "ema"},
    )
    new_student, new_teacher, new_optimizer, new_scheduler = _objects()
    original = {key: value.clone() for key, value in new_student.state_dict().items()}

    with pytest.raises(ValueError, match="dataset"):
        restore_training_state(
            payload,
            new_student,
            new_teacher,
            new_optimizer,
            new_scheduler,
            active_run_config={"dataset": "Landsat", "num_classes": 3},
        )
    with pytest.raises(ValueError, match="num_classes"):
        restore_training_state(
            payload,
            new_student,
            new_teacher,
            new_optimizer,
            new_scheduler,
            active_run_config={"dataset": "SECOND", "num_classes": 5},
        )

    assert all(
        torch.equal(new_student.state_dict()[key], value)
        for key, value in original.items()
    )


def test_restore_recovers_python_numpy_and_torch_rng_states():
    student, teacher, optimizer, scheduler = _objects()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    payload = build_training_state(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        optimizer_step=0,
        class_precision=torch.ones(3),
        best_metrics={},
        run_config={"dataset": "SECOND", "num_classes": 3, "seed": 7, "mode": "pseudo"},
    )
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restore_training_state(
        payload,
        student,
        teacher,
        optimizer,
        scheduler,
        active_run_config={"dataset": "SECOND", "num_classes": 3},
    )

    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_optional_teacher_and_scheduler_are_supported():
    student, _, optimizer, _ = _objects(teacher=False)
    payload = build_training_state(
        student=student,
        teacher=None,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        optimizer_step=4,
        class_precision=torch.ones(3),
        best_metrics={"Fscd": 0.2},
        run_config={"dataset": "SECOND", "num_classes": 3, "seed": 1, "mode": "supervised"},
    )
    new_student, _, new_optimizer, _ = _objects(teacher=False)

    resumed = restore_training_state(
        payload,
        new_student,
        None,
        new_optimizer,
        None,
        active_run_config={"dataset": "SECOND", "num_classes": 3},
    )

    assert resumed.epoch == 3
