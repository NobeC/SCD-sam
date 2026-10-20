# SAM Graph Stage1 Training Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the completed P0/P1 SAM graph components to the existing training loop behind a disabled-by-default switch.

**Architecture:** A focused runtime in `utils/graph_stage1_training.py` owns graph-only modules, optimizer, losses, and checkpoint state without registering or saving a duplicate backbone. `train.py` wraps the base dataset only when graph training is enabled and combines detached-backbone graph losses with the existing supervised loss while preserving the original command behavior.

**Tech Stack:** Python 3.8, PyTorch, NumPy, standard-library unittest.

## Global Constraints

- Preserve the default Landsat and SECOND training behavior.
- Do not alter validation or inference.
- Keep graph feature extraction detached from the backbone in Stage1.
- Run graph pooling, prototype logits, and EdgeNet loss in FP32.
- Resume old non-graph checkpoints with fresh graph state; require compatible graph state when present.

---

### Task 1: Graph Stage1 training runtime

**Files:**
- Create: `utils/graph_stage1_training.py`
- Test: `tests/test_graph_stage1_training_unittest.py`

**Interfaces:**
- Produces `GraphStage1TrainingRuntime` with `compute`, `parameters`, `zero_grad`, `state_dict`, and `load_state_dict`.
- Consumes the existing feature provider, temporal projector, audit function, and Stage1 coordinator.

- [ ] Write a failing test for finite loss/backward, queue state round-trip, and old-checkpoint compatibility.
- [ ] Run the focused unittest and confirm the missing-module failure.
- [ ] Implement the minimum runtime and checkpoint schema.
- [ ] Run the focused and all P1 unittests.

### Task 2: Opt-in integration in `train.py`

**Files:**
- Modify: `train.py`
- Test: `tests/test_train_graph_stage1_integration_unittest.py`

**Interfaces:**
- Adds `--use-sam-graph-stage1`, SAM cache/audit configuration, graph loss weights, graph optimizer configuration, and graph checkpoint data.
- Default false path retains the original four-item dataset batches and model/checkpoint behavior.

- [ ] Write a failing source-level integration test covering flags, wrapper construction, graph losses, optimizer stepping, logging, and checkpoint state.
- [ ] Run the focused test and confirm failure.
- [ ] Patch `train.py` with the smallest guarded integration.
- [ ] Run syntax checks, focused tests, full unittests, and a real MobileSAM one-step smoke test when data/cache availability permits.

