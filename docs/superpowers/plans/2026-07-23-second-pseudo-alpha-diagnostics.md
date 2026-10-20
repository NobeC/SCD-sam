# SECOND Pseudo-Alpha Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Print valid class-wise entropy-sorted precision and unlabeled coverage for a fixed pseudo-alpha sweep during SECOND pseudo-label training.

**Architecture:** Add a reusable tensor-only diagnostic accumulator to `utils/pseudo_label.py`. The training loop updates it from teacher outputs, valid changed labels, and actual unchanged pseudo-label candidates, then prints one per-class table at epoch end.

**Tech Stack:** Python, PyTorch, unittest

## Global Constraints

- Keep `lambda_sem_u` and `pseudo_change_thr` unchanged.
- Sweep ratios `0.03, 0.05, 0.10, 0.15, 0.20, 0.30`.
- Semantic precision uses only labels 1–6.
- Unchanged regions report coverage, not semantic accuracy.
- Preserve Landsat behavior and all checkpoint formats.

---

### Task 1: Entropy-Sorted Ratio Statistics

**Files:**
- Modify: `utils/pseudo_label.py`
- Create: `tests/test_pseudo_alpha_diagnostics.py`

**Interfaces:**
- Consumes: teacher logits, semantic labels, valid-GT mask, unlabeled candidate mask, ratios
- Produces: additive per-class counts and formatted precision/coverage summaries

- [ ] **Step 1: Write deterministic failing tests**

Test that lower-entropy predictions are selected first, measured precision uses
only valid GT pixels, and unlabeled pixels affect coverage but never precision.

- [ ] **Step 2: Verify tests fail because the diagnostic API is absent**

Run:

```bash
python -m unittest discover -s tests -p 'test_pseudo_alpha_diagnostics.py' -v
```

- [ ] **Step 3: Implement tensor-only accumulator**

Add initialization, per-batch accumulation, and epoch-summary functions. Use
stable ascending `torch.argsort` per predicted class and exact integer keep
counts matching the configured ratio.

- [ ] **Step 4: Verify focused tests pass**

Run the test command from Step 2 and expect all cases to pass.

### Task 2: Training Log Integration

**Files:**
- Modify: `train.py`
- Test: `tests/test_pseudo_alpha_diagnostics.py`

**Interfaces:**
- Consumes: the Task 1 accumulator API
- Produces: one `[pseudo-alpha sweep]` table per active pseudo-label epoch

- [ ] **Step 1: Initialize epoch diagnostics**

Create the accumulator for ratios `(0.03, 0.05, 0.10, 0.15, 0.20, 0.30)`.

- [ ] **Step 2: Accumulate A/B teacher outputs**

Use labels `>0` as the measured-precision mask and the same unchanged candidate
mask used by pseudo-label training as the unlabeled-coverage mask.

- [ ] **Step 3: Print sorted class-wise summary**

At epoch end print candidate counts, entropy statistics, ratio, accepted count,
measured TP/FP/precision, and unlabeled coverage for classes 1–6.

- [ ] **Step 4: Run regression verification**

Run:

```bash
python -m unittest discover -s tests -p 'test_pseudo_alpha_diagnostics.py' -v
python -m unittest discover -s tests -p 'test_semantic_pseudo.py' -v
python -m py_compile train.py utils/pseudo_label.py
```

Expected: diagnostic tests pass and both Python files compile successfully.

