# SCD-SAM Training Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Landsat graph+pseudo training time without changing gradients, sample coverage, active losses, or checkpoint format.

**Architecture:** The student optionally returns detached raw MobileSAM features so graph Stage1 can reuse them. Small tested helpers configure multiprocessing and accumulate exact training metrics on-device; pseudo-only diagnostics are sampled independently from the training loss.

**Tech Stack:** Python 3.8, PyTorch, unittest, CUDA AMP.

## Global Constraints

- Keep the current final-window gradient divisor and `drop_last=False`.
- Keep separate autocast scopes and all explicit FP32 numerical-stability paths.
- Do not change loss weights, optimizer/EMA update order, validation metrics, or checkpoint payloads.
- Preserve fallback graph computation for standalone callers.
- Work directly in the user-authorized project; do not commit the existing dirty/untracked workspace.

---

### Task 1: Reuse Student MobileSAM Features

**Files:**
- Modify: `main_model.py`
- Modify: `utils/graph_stage1_training.py`
- Modify: `train.py`
- Test: `tests/test_graph_stage1_training_unittest.py`

**Interfaces:**
- `ChangeDetection.forward(xa, xb, return_graph_features=False)` returns the existing three logits by default; when true it returns `(logits, (stage_a, neck_a, stage_b, neck_b))`.
- `GraphStage1TrainingRuntime.compute(..., graph_data, backbone_features=None)` accepts detached cached features and otherwise uses `feature_provider`.

- [ ] Add a failing runtime test that supplies four cached tensors, replaces the fallback backbone with a raising module, calls `compute`, and asserts finite differentiable graph losses.
- [ ] Run `python -m unittest discover -s tests -p test_graph_stage1_training_unittest.py -v`; expect the new keyword-argument test to fail.
- [ ] Add the optional return contract to `ChangeDetection.forward`, detach raw stage-3/neck tensors before channel adaptation, accept cached tensors in the graph runtime, and wire `train.py` to request/pass them only when graph Stage1 is active.
- [ ] Re-run the focused test and expect all cases to pass.

### Task 2: Optimize Loader and Host/Device Dataflow

**Files:**
- Create: `utils/training_performance.py`
- Modify: `train.py`
- Create: `tests/test_training_performance_unittest.py`

**Interfaces:**
- `loader_worker_options(num_workers: int) -> dict` returns `{}` for zero workers and persistent/prefetch options for positive workers.
- Training retains `audit_labels_A/B` on CPU and passes them to graph compute while CUDA copies use `non_blocking=True`.

- [ ] Add failing tests asserting zero-worker and four-worker loader options.
- [ ] Run the focused test; expect import failure because the helper does not exist.
- [ ] Implement the helper, merge it into both DataLoader constructors, preserve `drop_last=False`, add nonblocking CUDA copies, and retain CPU audit labels before transfer.
- [ ] Re-run the focused and graph runtime tests; expect all cases to pass.

### Task 3: Defer Training Metric Synchronization

**Files:**
- Modify: `utils/training_performance.py`
- Modify: `train.py`
- Modify: `tests/test_training_performance_unittest.py`

**Interfaces:**
- `TrainingMetricAccumulator(names, device)` accumulates last/sum/count tensors and exposes `snapshot()` only at log or epoch boundaries.
- `change_accuracy(outputs_a, outputs_b, change_logits, labels_a, labels_b)` returns an exact scalar tensor matching the existing fixed-shape per-sample mean.

- [ ] Add failing CPU tests comparing `change_accuracy` with the legacy NumPy calculation and checking exact accumulator last/average values.
- [ ] Run the focused test; expect missing symbols.
- [ ] Implement the helpers and replace per-iteration training `.cpu().numpy()` conversions with CUDA accumulation. Preserve exact epoch `acc_meter.avg` semantics and synchronize snapshots only for logging/end-of-epoch checkpoint metrics.
- [ ] Replace training and validation change-mask construction with the equivalent `change_logits > 0` operation; retain validation NumPy collection.
- [ ] Re-run focused tests and training-related unit tests.

### Task 4: Sample Pseudo Diagnostics and Use Inference Mode

**Files:**
- Modify: `train.py`
- Modify: `utils/training_performance.py`
- Modify: `tests/test_training_performance_unittest.py`

**Interfaces:**
- `is_sample_step(iteration, interval, include_first=True) -> bool` controls diagnostics without changing pseudo loss execution.
- CLI options `--pseudo-visualization-interval` (default 200) and `--pseudo-diagnostic-interval` (default 50); zero disables the associated sampling path.

- [ ] Add failing cadence tests for first step, periodic steps, disabled intervals, and invalid negative intervals.
- [ ] Run the focused test; expect missing cadence helper.
- [ ] Implement cadence validation and CLI arguments, sample alpha diagnostics and visualization at their independent intervals, and replace teacher `no_grad` with `inference_mode`.
- [ ] Re-run the focused pseudo/checkpoint tests.

### Task 5: Verification and Benchmark

**Files:**
- Verify: all modified files

- [ ] Run `python -m py_compile train.py main_model.py utils/graph_stage1_training.py utils/training_performance.py` and require exit code 0.
- [ ] Run `python -m unittest discover -s tests -p '*_unittest.py' -v` and require zero failures.
- [ ] Run `python tests/smoke_graph_stage1_real.py` and require finite loss/backward/optimizer output.
- [ ] Run a short identical-input CUDA timing comparison for fallback versus cached graph features with warm-up and synchronization; report measured results without extrapolating total epoch speed.
- [ ] Inspect the final diff to confirm accumulation divisor, `drop_last`, loss weights, AMP exclusions, and checkpoint schema are unchanged.

