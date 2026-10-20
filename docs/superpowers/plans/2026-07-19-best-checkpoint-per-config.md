# Best Checkpoint per Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain exactly one highest-Fscd checkpoint per parameter configuration and encode parameters plus all validation metrics in its filename.

**Architecture:** Add a focused `utils/checkpoint_manager.py` module for stable configuration identities, metric-bearing filenames, atomic saves, and same-configuration cleanup. Keep training responsible only for deciding whether Fscd improved and passing a consistent best-metric record to the manager.

**Tech Stack:** Python 3.8, PyTorch, pathlib, pytest.

## Global Constraints

- Fscd is the sole model-selection metric and must improve strictly.
- A failed new save must leave the previous best checkpoint intact.
- Cleanup must never cross parameter configuration or seed boundaries.
- All filename metrics and checkpoint `best_metrics` come from one best epoch.

---

### Task 1: Checkpoint naming and atomic replacement

**Files:**
- Create: `utils/checkpoint_manager.py`
- Create: `tests/test_checkpoint_manager.py`

**Interfaces:**
- Produces: `configuration_id(run_config) -> str`
- Produces: `checkpoint_filename(run_config, epoch, metrics) -> str`
- Produces: `save_best_checkpoint(directory, run_config, epoch, metrics, payload, save_fn=torch.save) -> pathlib.Path`

- [ ] Write failing tests for stable parameter formatting, complete metrics, same-config replacement, cross-config isolation, and save-failure preservation.
- [ ] Run `pytest -q tests/test_checkpoint_manager.py` and confirm failures.
- [ ] Implement filename generation and atomic replacement using a temporary sibling path followed by `os.replace`.
- [ ] Run `pytest -q tests/test_checkpoint_manager.py` and confirm all tests pass.

### Task 2: Training integration and metric consistency

**Files:**
- Modify: `train.py`
- Modify: `tests/test_checkpoint_manager.py`

**Interfaces:**
- Consumes: `save_best_checkpoint(...)` from Task 1.

- [ ] Add a failing source-level regression test asserting periodic checkpoint saving is removed and all best metrics update together.
- [ ] Run the focused test and confirm failure.
- [ ] Track best epoch, mIoU, Sek, Fscd, OA, and validation loss together; call the manager only on strict Fscd improvement.
- [ ] Run `pytest -q tests/test_checkpoint_manager.py tests/test_experiment_state.py`.
- [ ] Run `python -m py_compile train.py utils/checkpoint_manager.py`.

### Task 3: Final verification

**Files:**
- Verify only.

- [ ] Run the focused tests and syntax compilation with fresh output.
- [ ] Inspect `git diff -- train.py utils/checkpoint_manager.py tests/test_checkpoint_manager.py` for scope and accidental changes.
