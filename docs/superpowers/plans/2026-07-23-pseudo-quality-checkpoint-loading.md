# Pseudo-quality Checkpoint Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably load teacher, student, and legacy checkpoints in the pseudo-label visualization script with explicit dataset validation.

**Architecture:** Put pure checkpoint-selection and validation helpers in `utils/pseudo_checkpoint.py`, then connect them to `inspect_pseudo_quality.py`. Select the dataset before importing dataset-dependent modules.

**Tech Stack:** Python, PyTorch, unittest, argparse.

## Global Constraints

- Do not modify training behavior or model architecture.
- Default structured-checkpoint visualization to the EMA teacher.
- Reject dataset/class-count mismatch and incomplete model loads.

---

### Task 1: Checkpoint selection and validation

**Files:**
- Create: `utils/pseudo_checkpoint.py`
- Create: `tests/test_pseudo_checkpoint.py`

**Interfaces:**
- Produces: `select_model_state(checkpoint, owner) -> (state_dict, selected_owner)`
- Produces: `infer_semantic_num_classes(state_dict) -> Optional[int]`
- Produces: `validate_checkpoint_num_classes(state_dict, expected, dataset) -> None`

- [ ] Write tests for teacher/student selection, legacy mappings, absent teacher, and class mismatch.
- [ ] Run the tests and verify they fail because the module does not exist.
- [ ] Implement the three pure helper functions.
- [ ] Run the focused tests and verify they pass.

### Task 2: Strict visualization-script integration

**Files:**
- Modify: `inspect_pseudo_quality.py`
- Test: `tests/test_pseudo_checkpoint.py`

**Interfaces:**
- Consumes: the Task 1 helper functions.
- Produces: CLI arguments `--dataset` and `--checkpoint-model`.

- [ ] Add an early dataset parser and set `SCD_DATASET` before importing `RS_ST`.
- [ ] Replace hardcoded `DATA_NAME` with the resolved dataset configuration.
- [ ] Replace permissive checkpoint loading with requested-state selection, class validation, and strict loading.
- [ ] Register `--dataset` and `--checkpoint-model`.
- [ ] Run focused unit tests and Python syntax compilation.
- [ ] Smoke-test state extraction from the reported Landsat structured checkpoint without running full inference.
