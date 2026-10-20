# SSCLNet-Style Pseudo-Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve reproducible SCD-SAM baselines for Landsat and SECOND, then add SSCLNet-style EMA semantic pseudo-label supervision exclusively on unchanged regions and evaluate it with paired three-seed experiments.

**Architecture:** Move dataset selection and pseudo-label policy into small testable modules while retaining the existing SCD-SAM model and supervised losses. `train.py` remains the orchestration entry point, but delegates dataset validation, region-mask construction, class-balanced pseudo-label selection, EMA updates, checkpoint state, and experiment metadata to focused helpers.

**Tech Stack:** Python 3, PyTorch 1.12+, NumPy, pytest, tensorboardX; no new runtime dependency.

## Global Constraints

- Preserve the existing SCD-SAM backbone, heads, binary change loss, ChangeSimilarity loss, optimizer behavior, learning-rate schedule, input size, and gradient accumulation in baseline mode.
- Landsat uses five classes and `datasets/Landsat/`; SECOND uses seven classes and `datasets/SECOND/`.
- The binary change mask is available supervision; semantic pseudo labels are permitted only on unchanged pixels.
- EMA decay is exactly `0.999` for the designed experiment.
- Pseudo-label features are opt-in and must not change baseline results when disabled.
- Final evidence uses seeds `0`, `1`, and `2`, reports mean ± standard deviation, and requires at least `+0.5` percentage points in both Fscd and Sek with at least two of three paired seeds improving.
- Do not add Prototype Prompt, GCN, superpixel, or graph-propagation changes in this plan.

---

## File Structure

- Create `configs/datasets.py`: immutable Landsat/SECOND metadata and consistency validation.
- Create `utils/semantic_pseudo.py`: region-mask, class-balanced selection, precision update, pseudo loss, and EMA helpers.
- Create `utils/experiment_state.py`: checkpoint payload construction/restoration and run metadata.
- Modify `datasets/RS_ST.py`: consume a selected dataset configuration instead of hardcoded comment switches.
- Modify `train.py`: parse dataset/mode arguments and orchestrate the tested helpers.
- Create `tools/summarize_pseudo_experiments.py`: paired three-seed metric aggregation.
- Create `tests/test_dataset_config.py`.
- Create `tests/test_semantic_pseudo.py`.
- Create `tests/test_experiment_state.py`.
- Create `tests/test_train_modes.py`.
- Create `tests/test_experiment_summary.py`.

---

### Task 1: Explicit Dataset Configuration and Mismatch Guards

**Files:**
- Create: `configs/datasets.py`
- Modify: `datasets/RS_ST.py`
- Modify: `train.py`
- Test: `tests/test_dataset_config.py`

**Interfaces:**
- Produces: `DatasetConfig`, `get_dataset_config(name: str) -> DatasetConfig`, and `validate_dataset_model_config(config, num_classes, output_root) -> None`.
- Consumed by: `datasets/RS_ST.py`, `train.py`, checkpoint validation, and experiment metadata.

- [ ] **Step 1: Write the failing dataset selection test**

```python
# tests/test_dataset_config.py
import pytest

from configs.datasets import get_dataset_config, validate_dataset_model_config


def test_landsat_config_is_self_consistent(tmp_path):
    config = get_dataset_config("landsat")
    assert config.name == "Landsat"
    assert config.num_classes == 5
    assert config.root == "datasets/Landsat/"
    assert len(config.classes) == 5
    assert len(config.colormap) == 5
    validate_dataset_model_config(config, num_classes=5, output_root=tmp_path / "Landsat")


def test_second_config_is_self_consistent(tmp_path):
    config = get_dataset_config("second")
    assert config.name == "SECOND"
    assert config.num_classes == 7
    assert config.root == "datasets/SECOND/"
    assert len(config.classes) == 7
    assert len(config.colormap) == 7
    validate_dataset_model_config(config, num_classes=7, output_root=tmp_path / "SECOND")


def test_dataset_mismatch_fails_before_training(tmp_path):
    config = get_dataset_config("second")
    with pytest.raises(ValueError, match="num_classes"):
        validate_dataset_model_config(config, num_classes=5, output_root=tmp_path / "SECOND")
    with pytest.raises(ValueError, match="output"):
        validate_dataset_model_config(config, num_classes=7, output_root=tmp_path / "Landsat")
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/test_dataset_config.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'configs.datasets'`.

- [ ] **Step 3: Implement the minimal immutable configurations**

```python
# configs/datasets.py
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    num_classes: int
    classes: Tuple[str, ...]
    colormap: Tuple[Tuple[int, int, int], ...]


_CONFIGS = {
    "landsat": DatasetConfig(
        name="Landsat",
        root="datasets/Landsat/",
        num_classes=5,
        classes=("No change", "Farmland", "Desert", "Building", "Water"),
        colormap=((255, 255, 255), (0, 155, 0), (255, 165, 0),
                  (230, 30, 100), (0, 170, 240)),
    ),
    "second": DatasetConfig(
        name="SECOND",
        root="datasets/SECOND/",
        num_classes=7,
        classes=("unchanged", "water", "ground", "low vegetation",
                 "tree", "building", "sports field"),
        colormap=((255, 255, 255), (0, 0, 255), (128, 128, 128),
                  (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0)),
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in _CONFIGS:
        raise ValueError("Unsupported dataset: {}".format(name))
    return _CONFIGS[key]


def validate_dataset_model_config(config, num_classes, output_root):
    if num_classes != config.num_classes:
        raise ValueError(
            "num_classes={} does not match {}={}".format(
                num_classes, config.name, config.num_classes))
    if Path(output_root).name != config.name:
        raise ValueError(
            "output directory {!s} does not match dataset {}".format(
                output_root, config.name))
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
pytest -q tests/test_dataset_config.py
```

Expected: `3 passed`.

- [ ] **Step 5: Integrate the selected configuration**

Modify `datasets/RS_ST.py` so dataset metadata is configured through one function:

```python
from configs.datasets import get_dataset_config

_CONFIG = get_dataset_config(os.environ.get("SCD_DATASET", "landsat"))
num_classes = _CONFIG.num_classes
ST_COLORMAP = [list(color) for color in _CONFIG.colormap]
ST_CLASSES = list(_CONFIG.classes)
root = _CONFIG.root
```

Add `--dataset {landsat,second}` to `train.py`. Set `SCD_DATASET` before importing `datasets.RS_ST`, or refactor the dataset constructor to receive `DatasetConfig` directly if import ordering makes the environment approach unsafe. Prefer the constructor approach if it can be introduced without altering sample tensors.

- [ ] **Step 6: Add and run an import-level integration test**

```python
def test_selected_config_controls_dataset_metadata(monkeypatch):
    monkeypatch.setenv("SCD_DATASET", "second")
    import importlib
    import datasets.RS_ST as rs
    rs = importlib.reload(rs)
    assert rs.num_classes == 7
    assert rs.root == "datasets/SECOND/"
```

Run:

```bash
pytest -q tests/test_dataset_config.py
```

Expected: `4 passed`.

- [ ] **Step 7: Commit**

```bash
git add configs/datasets.py datasets/RS_ST.py train.py tests/test_dataset_config.py
git commit -m "feat: add explicit Landsat and SECOND configs"
```

---

### Task 2: SSCLNet Region Masks and Class-Balanced Pseudo Targets

**Files:**
- Create: `utils/semantic_pseudo.py`
- Modify: `utils/pseudo_label.py` only to re-export compatibility names or remove duplicated logic after tests pass
- Test: `tests/test_semantic_pseudo.py`

**Interfaces:**
- Produces:
  - `build_semantic_regions(labels_a, labels_b) -> (changed_mask, unchanged_mask)`
  - `select_class_balanced_pseudo(logits, candidate_mask, keep_ratios, ignore_index=255) -> PseudoSelection`
  - `semantic_pseudo_loss(student_logits, pseudo_targets, ignore_index=255) -> Tensor`
- `PseudoSelection` contains `targets`, `reliable_mask`, `entropy`, and `retained_per_class`.

- [ ] **Step 1: Write failing tests for the SSCLNet region definition**

```python
# tests/test_semantic_pseudo.py
import torch

from utils.semantic_pseudo import build_semantic_regions


def test_changed_pixels_are_union_of_both_semantic_labels():
    labels_a = torch.tensor([[[0, 1], [0, 0]]])
    labels_b = torch.tensor([[[0, 0], [2, 0]]])
    changed, unchanged = build_semantic_regions(labels_a, labels_b)
    assert torch.equal(changed, torch.tensor([[[False, True], [True, False]]]))
    assert torch.equal(unchanged, ~changed)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_semantic_pseudo.py::test_changed_pixels_are_union_of_both_semantic_labels
```

Expected: import fails because `utils.semantic_pseudo` does not exist.

- [ ] **Step 3: Implement the minimal mask helper**

```python
def build_semantic_regions(labels_a, labels_b):
    changed = (labels_a > 0) | (labels_b > 0)
    return changed, ~changed
```

- [ ] **Step 4: Verify GREEN**

Run the same test. Expected: `1 passed`.

- [ ] **Step 5: Write failing tests for pseudo-label isolation and exact retained counts**

```python
from utils.semantic_pseudo import select_class_balanced_pseudo


def test_pseudo_targets_never_leave_candidate_region():
    logits = torch.tensor([[
        [[8.0, 8.0], [8.0, 8.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]])
    candidates = torch.tensor([[[True, False], [False, True]]])
    selection = select_class_balanced_pseudo(
        logits, candidates, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(selection.reliable_mask, candidates)
    assert torch.all(selection.targets[~candidates] == 255)


def test_tied_entropy_keeps_exact_requested_count():
    logits = torch.zeros(1, 3, 1, 4)
    candidates = torch.ones(1, 1, 4, dtype=torch.bool)
    selection = select_class_balanced_pseudo(
        logits, candidates, torch.tensor([0.5, 0.5, 0.5]))
    assert int(selection.reliable_mask.sum()) == 2
```

- [ ] **Step 6: Verify RED**

Run:

```bash
pytest -q tests/test_semantic_pseudo.py
```

Expected: the new tests fail because `select_class_balanced_pseudo` is missing.

- [ ] **Step 7: Implement rank-based class selection**

Implement `PseudoSelection` as a dataclass. Compute probabilities, class predictions, and normalized entropy. For each predicted class, obtain candidate indices, calculate:

```python
keep_count = int(round(class_indices.numel() * float(keep_ratios[cls])))
keep_count = min(max(keep_count, 0), class_indices.numel())
ordered = class_indices[torch.argsort(class_entropy, stable=True)]
selected = ordered[:keep_count]
```

Initialize targets to `ignore_index`; assign predicted classes only at selected indices. This rank-based operation is required so entropy ties cannot exceed the requested count.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
pytest -q tests/test_semantic_pseudo.py
```

Expected: all current tests pass.

- [ ] **Step 9: Write failing tests for empty candidates and independent A/B targets**

```python
from utils.semantic_pseudo import semantic_pseudo_loss


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
```

- [ ] **Step 10: Verify RED, implement minimal zero-safe loss, and verify GREEN**

Run before implementation:

```bash
pytest -q tests/test_semantic_pseudo.py
```

Expected: failure because `semantic_pseudo_loss` is missing.

Minimal implementation:

```python
def semantic_pseudo_loss(student_logits, pseudo_targets, ignore_index=255):
    if not (pseudo_targets != ignore_index).any():
        return student_logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(
        student_logits, pseudo_targets, ignore_index=ignore_index)
```

Run again. Expected: all tests pass.

- [ ] **Step 11: Commit**

```bash
git add utils/semantic_pseudo.py utils/pseudo_label.py tests/test_semantic_pseudo.py
git commit -m "feat: add SSCLNet semantic pseudo targets"
```

---

### Task 3: Precision State, Keep-Ratio Schedule, and EMA Updates

**Files:**
- Modify: `utils/semantic_pseudo.py`
- Test: `tests/test_semantic_pseudo.py`

**Interfaces:**
- Produces:
  - `update_precision_ema(previous, true_positive, predicted_positive, momentum) -> Tensor`
  - `scheduled_keep_ratios(epoch, start_epoch, total_epochs, start_ratio, end_ratio, precision, min_ratio) -> Tensor`
  - `update_ema_teacher(student, teacher, decay, optimizer_step) -> None`

- [ ] **Step 1: Write failing tests for absent-class precision**

```python
from utils.semantic_pseudo import update_precision_ema


def test_absent_class_keeps_previous_precision():
    previous = torch.tensor([0.0, 0.8, 0.4])
    tp = torch.tensor([0.0, 4.0, 0.0])
    predicted = torch.tensor([0.0, 5.0, 0.0])
    updated = update_precision_ema(previous, tp, predicted, momentum=0.9)
    assert updated[0].item() == 0.0
    assert torch.isclose(updated[2], previous[2])
    assert updated[1] < previous[1]
```

- [ ] **Step 2: Verify RED, implement the minimal update, verify GREEN**

The update applies EMA only where `predicted_positive > 0`:

```python
observed = true_positive / predicted_positive.clamp_min(1.0)
has_evidence = predicted_positive > 0
updated = previous.clone()
updated[has_evidence] = (
    momentum * previous[has_evidence]
    + (1.0 - momentum) * observed[has_evidence]
)
```

Run:

```bash
pytest -q tests/test_semantic_pseudo.py::test_absent_class_keeps_previous_precision
```

Expected: pass.

- [ ] **Step 3: Write failing tests for warm-up and dataset-safe weights**

```python
from utils.semantic_pseudo import scheduled_keep_ratios


def test_keep_ratio_is_zero_before_pseudo_start():
    ratios = scheduled_keep_ratios(
        epoch=29, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([0.0, 0.8, 0.5]), min_ratio=0.0)
    assert torch.equal(ratios, torch.zeros_like(ratios))


def test_background_keep_ratio_is_always_zero():
    ratios = scheduled_keep_ratios(
        epoch=60, start_epoch=30, total_epochs=100,
        start_ratio=0.03, end_ratio=0.10,
        precision=torch.tensor([1.0, 0.8, 0.5]), min_ratio=0.0)
    assert ratios[0].item() == 0.0
```

Implement linear interpolation followed by precision scaling and clamping. Verify all tests pass.

- [ ] **Step 4: Write failing EMA tests**

```python
import torch.nn as nn
from utils.semantic_pseudo import update_ema_teacher


def test_ema_updates_parameters_and_float_buffers():
    student = nn.BatchNorm2d(1)
    teacher = nn.BatchNorm2d(1)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.fill_(0.0)
        student.running_mean.fill_(4.0)
        teacher.running_mean.fill_(0.0)
    update_ema_teacher(student, teacher, decay=0.999, optimizer_step=1000)
    assert torch.allclose(teacher.weight, torch.tensor([0.002]))
    assert torch.allclose(teacher.running_mean, torch.tensor([0.004]))


def test_ema_is_not_called_for_skipped_step():
    # The orchestration API must require a confirmed optimizer step.
    # This test will be completed in Task 5 against train-step orchestration.
    pass
```

- [ ] **Step 5: Verify RED, implement EMA, and verify GREEN**

Use state-dict keys so parameters and buffers remain aligned. Apply the requested fixed decay `0.999`; do not substitute the current `0.99` default. Copy non-floating buffers.

Run:

```bash
pytest -q tests/test_semantic_pseudo.py
```

Expected: all non-placeholder tests pass. Replace the temporary skipped-step test with the Task 5 integration test before Task 5 is committed; no `pass` test may remain in the final suite.

- [ ] **Step 6: Commit**

```bash
git add utils/semantic_pseudo.py tests/test_semantic_pseudo.py
git commit -m "feat: add precision schedule and EMA teacher updates"
```

---

### Task 4: Complete Checkpoint and Resume State

**Files:**
- Create: `utils/experiment_state.py`
- Modify: `train.py`
- Test: `tests/test_experiment_state.py`

**Interfaces:**
- Produces:
  - `build_training_state(...) -> dict`
  - `restore_training_state(payload, student, teacher, optimizer, scheduler) -> ResumeState`
- State includes dataset, seed, mode, epoch, optimizer step, class precision, best metrics, and pseudo configuration.

- [ ] **Step 1: Write the failing round-trip test**

```python
# tests/test_experiment_state.py
import torch
import torch.nn as nn

from utils.experiment_state import build_training_state, restore_training_state


def test_checkpoint_round_trip_restores_teacher_and_pseudo_state():
    student = nn.Linear(2, 2)
    teacher = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    precision = torch.tensor([0.0, 0.7, 0.4])

    payload = build_training_state(
        student=student, teacher=teacher, optimizer=optimizer,
        scheduler=scheduler, epoch=40, optimizer_step=123,
        class_precision=precision,
        best_metrics={"Fscd": 0.7, "Sek": 0.3},
        run_config={"dataset": "SECOND", "seed": 1, "mode": "pseudo"})

    new_student = nn.Linear(2, 2)
    new_teacher = nn.Linear(2, 2)
    new_optimizer = torch.optim.SGD(new_student.parameters(), lr=0.1)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
    resumed = restore_training_state(
        payload, new_student, new_teacher, new_optimizer, new_scheduler)

    assert resumed.epoch == 40
    assert resumed.optimizer_step == 123
    assert torch.equal(resumed.class_precision, precision)
    assert resumed.run_config["dataset"] == "SECOND"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_experiment_state.py
```

Expected: import fails because `utils.experiment_state` does not exist.

- [ ] **Step 3: Implement minimal complete state helpers**

Use a frozen `ResumeState` dataclass. Store both models, optimizer, scheduler, class precision, best metrics, run configuration, and CPU/CUDA RNG states. Validate that checkpoint dataset and class count match the active configuration before loading weights.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest -q tests/test_experiment_state.py
```

Expected: pass.

- [ ] **Step 5: Integrate state helpers into `train.py`**

Replace model-only resume handling with `restore_training_state`. Save a complete payload for best and periodic checkpoints. Filename format:

```text
{dataset}_{mode}_seed{seed}_epoch{epoch:03d}_Fscd{fscd:.4f}.pth
```

- [ ] **Step 6: Run unit and syntax verification**

```bash
pytest -q tests/test_experiment_state.py tests/test_dataset_config.py
python -m py_compile train.py utils/experiment_state.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 7: Commit**

```bash
git add utils/experiment_state.py train.py tests/test_experiment_state.py
git commit -m "feat: save complete pseudo-label training state"
```

---

### Task 5: Training-Mode Integration Without Baseline Drift

**Files:**
- Modify: `train.py`
- Modify: `utils/semantic_pseudo.py`
- Test: `tests/test_train_modes.py`

**Interfaces:**
- Produces: `compute_training_losses(...) -> LossBundle` and `apply_optimizer_step(...) -> bool`.
- Baseline mode excludes teacher forward and returns an exact zero pseudo term.
- Pseudo mode uses unchanged-region targets only after `pseudo_start_epoch`.

- [ ] **Step 1: Write a failing baseline-equivalence test**

```python
# tests/test_train_modes.py
from types import SimpleNamespace
import torch

from train import compute_training_losses


def test_disabled_pseudo_mode_preserves_supervised_total():
    sem_sup = torch.tensor(1.25, requires_grad=True)
    change = torch.tensor(0.50, requires_grad=True)
    similarity = torch.tensor(0.25, requires_grad=True)
    losses = compute_training_losses(
        sem_sup=sem_sup, change=change, similarity=similarity,
        pseudo=torch.tensor(99.0), lambda_u=0.0, pseudo_enabled=False)
    assert losses.pseudo.item() == 0.0
    assert losses.total.item() == 2.0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_train_modes.py::test_disabled_pseudo_mode_preserves_supervised_total
```

Expected: import or attribute failure because the helper is missing.

- [ ] **Step 3: Implement the minimal `LossBundle` helper**

```python
@dataclass(frozen=True)
class LossBundle:
    semantic_supervised: torch.Tensor
    change: torch.Tensor
    similarity: torch.Tensor
    pseudo: torch.Tensor
    weighted_pseudo: torch.Tensor
    total: torch.Tensor
```

When pseudo mode is disabled, derive zero from `sem_sup.sum() * 0.0` and do not use the supplied pseudo value.

- [ ] **Step 4: Verify GREEN**

Run the same test. Expected: pass.

- [ ] **Step 5: Write failing pseudo-region integration tests**

```python
def test_changed_pixels_are_ignored_by_pseudo_loss():
    # Construct one changed pixel and one unchanged pixel.
    # Assert the changed pseudo target is 255 and only the unchanged
    # student logit receives a pseudo-loss gradient.
    ...


def test_before_start_epoch_pseudo_weight_is_zero():
    ...


def test_valid_step_updates_student_then_teacher():
    ...


def test_nonfinite_step_updates_neither_teacher_nor_scheduler():
    ...
```

Replace each ellipsis before running the test with explicit 1×C×1×2 tensors, cloned model parameters, and exact assertions. Do not mock optimizer or scheduler behavior; use `torch.optim.SGD` and `StepLR` on a one-layer real model.

- [ ] **Step 6: Verify RED**

Run:

```bash
pytest -q tests/test_train_modes.py
```

Expected: each new behavior test fails because orchestration has not yet been extracted.

- [ ] **Step 7: Integrate the tested helpers into `train.py`**

Training order:

```text
student forward
supervised losses
teacher forward under no_grad only when enabled and epoch >= start
changed/unchanged masks from binary supervision
teacher-change consistency mask
independent A/B class-balanced pseudo targets
zero-safe A/B pseudo loss
combined loss and backward
finite-gradient check
optimizer step
scheduler step
EMA update
```

EMA and scheduler execute only when the optimizer step succeeds. Change the CLI default EMA decay to `0.999`. Keep pseudo mode disabled by default.

- [ ] **Step 8: Verify GREEN and baseline imports**

Run:

```bash
pytest -q tests/test_train_modes.py tests/test_semantic_pseudo.py
python -m py_compile train.py
```

Expected: all pass.

- [ ] **Step 9: Add persistent diagnostics**

Log these scalar tags:

```text
train/loss_sem_sup
train/loss_change
train/loss_similarity
train/loss_sem_pseudo
train/lambda_sem_u
train/weighted_sem_pseudo
train/pseudo_ratio_total
train/pseudo_ratio_class_{i}
train/teacher_precision_class_{i}
train/teacher_student_agreement
train/skipped_nonfinite_steps
```

Write run configuration as JSON next to TensorBoard logs and into every checkpoint.

- [ ] **Step 10: Commit**

```bash
git add train.py utils/semantic_pseudo.py tests/test_train_modes.py
git commit -m "feat: integrate SSCLNet pseudo supervision into training"
```

---

### Task 6: Paired Three-Seed Experiment Runner and Summary

**Files:**
- Create: `tools/summarize_pseudo_experiments.py`
- Create: `configs/experiments/landsat_baseline.json`
- Create: `configs/experiments/landsat_pseudo.json`
- Create: `configs/experiments/second_baseline.json`
- Create: `configs/experiments/second_pseudo.json`
- Test: `tests/test_experiment_summary.py`

**Interfaces:**
- Produces:
  - `summarize_pairs(records) -> dict`
  - CLI that reads JSON result records and writes a Markdown/JSON summary.

- [ ] **Step 1: Write the failing paired-summary test**

```python
# tests/test_experiment_summary.py
from tools.summarize_pseudo_experiments import summarize_pairs


def test_summary_reports_paired_mean_std_and_success():
    records = [
        {"seed": 0, "mode": "baseline", "Fscd": 70.0, "Sek": 32.0},
        {"seed": 0, "mode": "pseudo", "Fscd": 70.8, "Sek": 32.7},
        {"seed": 1, "mode": "baseline", "Fscd": 70.2, "Sek": 32.1},
        {"seed": 1, "mode": "pseudo", "Fscd": 70.9, "Sek": 32.8},
        {"seed": 2, "mode": "baseline", "Fscd": 69.9, "Sek": 31.9},
        {"seed": 2, "mode": "pseudo", "Fscd": 70.5, "Sek": 32.5},
    ]
    summary = summarize_pairs(records)
    assert summary["paired_seed_count"] == 3
    assert summary["Fscd"]["mean_delta"] >= 0.5
    assert summary["Sek"]["mean_delta"] >= 0.5
    assert summary["Fscd"]["improved_seed_count"] == 3
    assert summary["success"] is True
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_experiment_summary.py
```

Expected: import fails because the summary tool does not exist.

- [ ] **Step 3: Implement exact paired aggregation**

Match records by seed, reject missing or duplicate mode/seed pairs, calculate population mean and sample standard deviation, count positive deltas, and apply:

```python
success = (
    fscd_mean_delta >= 0.5
    and sek_mean_delta >= 0.5
    and fscd_improved_seed_count >= 2
    and sek_improved_seed_count >= 2
)
```

Include mIoU/OA regression checks when those fields are supplied.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest -q tests/test_experiment_summary.py
```

Expected: pass.

- [ ] **Step 5: Add fixed experiment configurations**

Each JSON file contains dataset, mode, seeds `[0, 1, 2]`, epochs, optimizer settings copied from the frozen baseline, `pseudo_start_epoch`, `ema_decay`, `lambda_sem_u`, keep-ratio schedule, output directory, and configuration identifier.

Baseline configurations set:

```json
{
  "use_ema_teacher": false,
  "use_semantic_pseudo": false,
  "lambda_sem_u": 0.0
}
```

Pseudo configurations set `ema_decay` to `0.999`. Landsat and SECOND use separately selected `lambda_sem_u`; do not share a value merely for symmetry.

- [ ] **Step 6: Verify configuration parsing**

Extend `tests/test_experiment_summary.py` to load all four JSON files and assert dataset, mode, seed list, and pseudo flags are consistent.

Run:

```bash
pytest -q tests/test_experiment_summary.py tests/test_dataset_config.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tools/summarize_pseudo_experiments.py configs/experiments tests/test_experiment_summary.py
git commit -m "feat: add paired pseudo-label experiment summaries"
```

---

### Task 7: End-to-End Verification and Experiment Execution

**Files:**
- Modify: `README.md` or create `docs/experiments/ssclnet-pseudo-label.md`
- Test: all test files

**Interfaces:**
- Consumes all prior tasks.
- Produces reproducible commands and result records for Landsat and SECOND.

- [ ] **Step 1: Run the complete CPU test suite**

```bash
pytest -q \
  tests/test_dataset_config.py \
  tests/test_semantic_pseudo.py \
  tests/test_experiment_state.py \
  tests/test_train_modes.py \
  tests/test_experiment_summary.py
```

Expected: all tests pass with no warnings caused by project code.

- [ ] **Step 2: Run syntax checks**

```bash
python -m py_compile \
  train.py \
  configs/datasets.py \
  utils/semantic_pseudo.py \
  utils/experiment_state.py \
  tools/summarize_pseudo_experiments.py
```

Expected: exit zero.

- [ ] **Step 3: Run one-batch smoke tests**

Add a tested `--max-train-batches 1 --max-val-batches 1` development option that defaults to unlimited and does not alter normal runs.

Run baseline smoke tests:

```bash
python train.py --dataset landsat --seed 0 \
  --no-use-ema-teacher --no-use-semantic-pseudo \
  --epochs 1 --max-train-batches 1 --max-val-batches 1

python train.py --dataset second --seed 0 \
  --no-use-ema-teacher --no-use-semantic-pseudo \
  --epochs 1 --max-train-batches 1 --max-val-batches 1
```

Run pseudo smoke tests with `--pseudo-start-epoch 0` so the branch executes:

```bash
python train.py --dataset landsat --seed 0 \
  --use-ema-teacher --use-semantic-pseudo \
  --ema-decay 0.999 --pseudo-start-epoch 0 \
  --lambda-sem-u 0.1 \
  --epochs 1 --max-train-batches 1 --max-val-batches 1

python train.py --dataset second --seed 0 \
  --use-ema-teacher --use-semantic-pseudo \
  --ema-decay 0.999 --pseudo-start-epoch 0 \
  --lambda-sem-u 0.03 \
  --epochs 1 --max-train-batches 1 --max-val-batches 1
```

Expected: correct dataset/class count is printed; baseline logs zero pseudo contribution; pseudo runs execute teacher inference; all losses are finite; checkpoints contain complete state.

- [ ] **Step 4: Reproduce frozen supervised baselines**

Run the established paper-compatible hyperparameters for Landsat and SECOND with pseudo features disabled. Compare the new run against the stored baseline, including best Fscd, Sek, mIoU, OA, loss curves, optimizer steps, and learning-rate curve.

Do not begin full pseudo experiments until:

- Landsat remains at its established near-paper result.
- SECOND reaches the original-paper neighborhood or the remaining gap is explicitly documented and accepted.

- [ ] **Step 5: Select one pseudo configuration per dataset**

Use seed `0` only:

```text
Landsat lambda_u: 0.1, 0.2, 0.3
SECOND lambda_u: 0.01, 0.03, 0.05, 0.1
```

Keep EMA decay `0.999` and the approved keep-ratio/warm-up schedule fixed. Select based on validation Fscd/Sek plus pseudo precision and coverage, not Fscd alone.

- [ ] **Step 6: Run paired three-seed experiments**

For each dataset and seed `0`, `1`, `2`, run:

```text
baseline: EMA off, pseudo off
pseudo: selected config, identical seed and baseline settings
```

Store one machine-readable result JSON per run with best and final metrics.

- [ ] **Step 7: Generate the final comparison**

```bash
python tools/summarize_pseudo_experiments.py \
  --input results/experiments/Landsat \
  --output results/experiments/Landsat/summary

python tools/summarize_pseudo_experiments.py \
  --input results/experiments/SECOND \
  --output results/experiments/SECOND/summary
```

Expected: paired per-seed deltas, mean ± standard deviation, improved-seed counts, mIoU/OA regression status, and an explicit pass/fail result for each dataset.

- [ ] **Step 8: Document exact commands and results**

Create `docs/experiments/ssclnet-pseudo-label.md` containing environment, commit SHA, dataset versions, checkpoint provenance, commands, configuration hashes, per-seed metrics, aggregated metrics, pseudo-label diagnostics, runtime, and GPU memory.

- [ ] **Step 9: Final verification**

Run:

```bash
git status --short
pytest -q tests
python -m py_compile train.py
```

Expected: only intended experiment outputs or documentation remain uncommitted; tests pass; syntax check passes.

- [ ] **Step 10: Commit**

```bash
git add docs/experiments/ssclnet-pseudo-label.md
git commit -m "docs: record SSCLNet pseudo-label experiments"
```
