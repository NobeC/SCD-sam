# Stage4 AMP Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep FPN stage4 convolution and BatchNorm finite under AMP and reject contaminated training snapshots before resume.

**Architecture:** Add a narrow FP32 execution boundary inside `FPNNeck` only for `stage4_Conv1`. Add reusable finite-state validation in the existing experiment-state restore boundary before any model mutation.

**Tech Stack:** Python 3.8, PyTorch, standard-library unittest.

## Global Constraints

- Preserve AMP behavior outside `neck.stage4_Conv1`.
- Preserve the Landsat and SECOND model architecture and checkpoint schema.
- Reject invalid checkpoints before loading any student or teacher tensor.
- Name the first invalid state tensor in the exception.

---

### Task 1: FP32 stage4 boundary

**Files:**
- Modify: `models/Decoder/FPN.py`
- Create: `tests/test_fpn_amp_stability_unittest.py`

**Interfaces:**
- Produces: `FPNNeck._forward_stage4(fa4, fb4) -> Tensor`.

- [ ] Write a failing test that calls `_forward_stage4` inside autocast and records FP32 input at `stage4_Conv1`.
- [ ] Run the test and confirm it fails because `_forward_stage4` is absent.
- [ ] Implement `_forward_stage4` with autocast disabled and FP32 concatenated input; call it from `forward`.
- [ ] Run the focused test and confirm it passes.

### Task 2: Finite checkpoint validation

**Files:**
- Modify: `utils/experiment_state.py`
- Create: `tests/test_experiment_state_finite_unittest.py`

**Interfaces:**
- Produces: `_validate_finite_state_dict(state_dict, owner) -> None`.

- [ ] Write failing tests for student NaN and teacher Inf rejection before model mutation.
- [ ] Run tests and confirm the validator behavior is absent.
- [ ] Validate student and teacher floating tensors before any `load_state_dict` call.
- [ ] Run focused tests and existing experiment-state tests where the environment permits.

### Task 3: Verification

**Files:**
- Verify only.

- [ ] Run both unittest modules with the CUDA/PyTorch environment.
- [ ] Run `python -m py_compile models/Decoder/FPN.py utils/experiment_state.py train.py`.
- [ ] Inspect the focused git diff for unrelated changes.
