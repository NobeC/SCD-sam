# Fusion4 Attention AMP Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent FP16 overflow in `fusion4` attention from producing non-finite model outputs.

**Architecture:** Retain AMP for projections and the rest of SCD-SAM, while calculating attention scores, Softmax probabilities, and context aggregation in FP32. Convert the context back to the projection dtype before the existing output layer.

**Tech Stack:** Python, PyTorch, unittest/pytest

## Global Constraints

- Preserve the existing backbone and model parameter structure.
- Preserve Landsat and SECOND behavior.
- Do not add external dependencies.
- Keep AMP enabled outside the narrow attention boundary.

---

### Task 1: Add FP32 attention regression coverage

**Files:**
- Create: `tests/test_dfi_attention_amp_stability.py`
- Modify: `models/Encoder/DFI.py:91-121`

**Interfaces:**
- Consumes: `models.Encoder.DFI.Attention.forward(x1, x2)`
- Produces: finite output with the same dtype and shape as `x1`

- [ ] **Step 1: Write the failing test**

Create an `Attention` instance, replace its projections with identities, pass
large finite FP16 inputs whose dot product overflows FP16, and assert that the
result is finite, FP16, and shape-preserving.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_dfi_attention_amp_stability.py
```

Expected: failure because the existing FP16 query-key multiplication produces
non-finite Softmax output.

- [ ] **Step 3: Write minimal implementation**

In `Attention.forward`, calculate score multiplication, scaling, Softmax, and
context multiplication using `.float()`. Convert the context tensor back with
`.to(mixed_value_layer.dtype)` immediately before the output projection.

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
pytest -q tests/test_dfi_attention_amp_stability.py tests/test_fpn_amp_stability_unittest.py
python -m py_compile models/Encoder/DFI.py
```

Expected: all tests pass and compilation exits successfully.

- [ ] **Step 5: Review the final diff**

Run:

```bash
git diff -- models/Encoder/DFI.py tests/test_dfi_attention_amp_stability.py
```

Confirm that only the narrow attention stability boundary and its regression
test changed production behavior.

