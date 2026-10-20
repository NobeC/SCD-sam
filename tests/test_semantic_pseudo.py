import torch
import torch.nn as nn

from utils.semantic_pseudo import (
    build_semantic_regions,
    select_class_balanced_pseudo,
    semantic_pseudo_loss,
    scheduled_keep_ratios,
    supervised_semantic_loss,
    update_precision_ema,
)


def test_changed_pixels_are_union_of_both_semantic_labels():
    labels_a = torch.tensor([[[0, 1], [0, 0]]])
    labels_b = torch.tensor([[[0, 0], [2, 0]]])

    changed, unchanged = build_semantic_regions(labels_a, labels_b)

    assert torch.equal(
        changed, torch.tensor([[[False, True], [True, False]]])
    )
    assert torch.equal(unchanged, ~changed)


def test_supervised_semantic_loss_ignores_unchanged_zero_labels():
    logits_a = torch.randn(1, 4, 2, 2, requires_grad=True)
    logits_b = torch.randn(1, 4, 2, 2, requires_grad=True)
    labels_a = torch.tensor([[[0, 2], [0, 0]]])
    labels_b = torch.tensor([[[0, 0], [3, 0]]])

    loss = supervised_semantic_loss(logits_a, logits_b, labels_a, labels_b)

    expected = (
        torch.nn.functional.cross_entropy(
            logits_a.permute(0, 2, 3, 1)[labels_a > 0],
            labels_a[labels_a > 0],
        )
        + torch.nn.functional.cross_entropy(
            logits_b.permute(0, 2, 3, 1)[labels_b > 0],
            labels_b[labels_b > 0],
        )
    ) * 0.5
    assert torch.allclose(loss, expected)


def test_supervised_semantic_loss_is_finite_zero_without_changed_pixels():
    logits_a = torch.randn(1, 4, 2, 2, requires_grad=True)
    logits_b = torch.randn(1, 4, 2, 2, requires_grad=True)
    labels_a = torch.zeros(1, 2, 2, dtype=torch.long)
    labels_b = torch.zeros(1, 2, 2, dtype=torch.long)

    loss = supervised_semantic_loss(logits_a, logits_b, labels_a, labels_b)

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits_a.grad is not None
    assert logits_b.grad is not None


def test_pseudo_targets_never_leave_candidate_region():
    logits = torch.tensor(
        [[
            [[8.0, 8.0], [8.0, 8.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]]
    )
    candidates = torch.tensor([[[True, False], [False, True]]])

    selection = select_class_balanced_pseudo(
        logits, candidates, torch.tensor([1.0, 1.0, 1.0])
    )

    assert torch.equal(selection.reliable_mask, candidates)
    assert torch.all(selection.targets[~candidates] == 255)


def test_tied_entropy_keeps_exact_requested_count():
    logits = torch.zeros(1, 3, 1, 4)
    candidates = torch.ones(1, 1, 4, dtype=torch.bool)

    selection = select_class_balanced_pseudo(
        logits, candidates, torch.tensor([0.5, 0.5, 0.5])
    )

    assert int(selection.reliable_mask.sum()) == 2


def test_empty_pseudo_targets_return_finite_zero_loss():
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    targets = torch.full((1, 2, 2), 255, dtype=torch.long)

    loss = semantic_pseudo_loss(logits, targets)

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_temporal_logits_generate_independent_targets():
    candidates = torch.ones(1, 1, 1, dtype=torch.bool)
    keep = torch.ones(3)
    logits_a = torch.tensor([[[[0.0]], [[9.0]], [[0.0]]]])
    logits_b = torch.tensor([[[[0.0]], [[0.0]], [[9.0]]]])

    target_a = select_class_balanced_pseudo(logits_a, candidates, keep).targets
    target_b = select_class_balanced_pseudo(logits_b, candidates, keep).targets

    assert target_a.item() == 1
    assert target_b.item() == 2


def test_absent_class_keeps_previous_precision():
    previous = torch.tensor([0.0, 0.8, 0.4])
    true_positive = torch.tensor([0.0, 2.0, 0.0])
    predicted_positive = torch.tensor([0.0, 5.0, 0.0])

    updated = update_precision_ema(
        previous, true_positive, predicted_positive, momentum=0.75
    )

    assert torch.allclose(updated, torch.tensor([0.0, 0.7, 0.4]))
    assert updated[2].item() == previous[2].item()


def test_keep_ratio_is_zero_before_pseudo_start():
    ratios = scheduled_keep_ratios(
        epoch=29, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([0.0, 0.8, 0.5]), min_ratio=0.0,
    )

    assert torch.equal(ratios, torch.zeros_like(ratios))


def test_background_keep_ratio_is_always_zero():
    ratios = scheduled_keep_ratios(
        epoch=60, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([1.0, 0.8, 0.5]), min_ratio=0.0,
    )

    assert ratios[0].item() == 0.0


def test_keep_ratio_mid_epoch_interpolates_and_scales_by_precision():
    ratios = scheduled_keep_ratios(
        epoch=65, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([1.0, 0.8, 0.5]), min_ratio=0.0,
    )

    assert torch.allclose(ratios, torch.tensor([0.0, 0.052, 0.0325]))


def test_keep_ratio_applies_minimum_clamp_to_semantic_classes():
    ratios = scheduled_keep_ratios(
        epoch=65, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([1.0, 0.8, 0.1]), min_ratio=0.02,
    )

    assert torch.allclose(ratios, torch.tensor([0.0, 0.052, 0.02]))


def test_keep_ratio_clamps_progress_at_end_and_ratio_at_one():
    ratios = scheduled_keep_ratios(
        epoch=130, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([20.0, 20.0, 0.5]), min_ratio=0.0,
    )

    assert torch.allclose(ratios, torch.tensor([0.0, 1.0, 0.05]))


def test_ema_updates_parameters_and_float_buffers():
    from utils.semantic_pseudo import update_ema_teacher

    student = nn.BatchNorm2d(1)
    teacher = nn.BatchNorm2d(1)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.fill_(0.0)
        student.running_mean.fill_(4.0)
        teacher.running_mean.fill_(0.0)

    update_ema_teacher(
        student, teacher, decay=0.999, optimizer_step=1000
    )

    assert torch.allclose(teacher.weight, torch.tensor([0.002]))
    assert torch.allclose(teacher.running_mean, torch.tensor([0.004]))


def test_ema_copies_non_floating_buffers():
    from utils.semantic_pseudo import update_ema_teacher

    student = nn.BatchNorm2d(1)
    teacher = nn.BatchNorm2d(1)
    with torch.no_grad():
        student.num_batches_tracked.fill_(7)
        teacher.num_batches_tracked.zero_()

    update_ema_teacher(
        student, teacher, decay=0.999, optimizer_step=1
    )

    assert torch.equal(
        teacher.num_batches_tracked, student.num_batches_tracked
    )
