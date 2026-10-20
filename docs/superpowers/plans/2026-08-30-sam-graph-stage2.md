# SAM Graph Stage2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimal batch-local SAM graph Stage2 that uses the trained Stage1 EdgeNet and prototypes to veto EMA-teacher pseudo labels and backpropagate accepted pixel pseudo-label CE into the student.

**Architecture:** Current-batch GT/U nodes form sparse propagation, spatial, temporal, and self relations. A one-layer relation-aware GAT refines current nodes, existing EMA prototypes classify them, hard mask reprojection creates graph pixel predictions, and graph predictions veto teacher candidates before student CE. Stage2 is opt-in, excluded from validation/inference, and restores Stage1 v1 checkpoints with fresh Stage2 state.

**Tech Stack:** Python 3.8, PyTorch 1.12+ APIs, unittest, existing MobileSAM cache and graph modules; no PyTorch Geometric or new dependencies.

## Global Constraints

- Keep Stage1 behavior and old model forward returns unchanged when Stage2 is disabled.
- Class zero is never a graph prototype or graph pseudo target.
- IGNORE nodes never enter the graph; temporal edges require U/U state.
- EdgeNet tensor shapes remain unchanged and pseudo pairs never supervise EdgeNet.
- Teacher inference and pseudo targets are detached; Stage2 pixel CE must update student logits/backbone.
- All attention, reliability logs, segment softmax, KL, and accumulators run in FP32.
- Empty graphs and empty accepted pseudo sets return connected finite zero losses.
- Validation and inference never execute Stage2.

---

### Task 1: Batch Sparse Graph Builder

**Files:**
- Create: `models/graph/graph_builder.py`
- Create: `tests/test_graph_stage2_builder_unittest.py`

**Interfaces:**
- Consumes: `TemporalNodeOutput`, `SamNodeAudit`, the existing `EdgeNet`.
- Produces: `SparseGraphEdges(source, destination, relation, reliability, propagation_count, spatial_count, temporal_count)` and `build_stage2_edges(node_outputs, audits, edge_net, local_topk, edge_threshold, device)`.

- [ ] **Step 1: Write failing tests** for bidirectional GT-U Top-K, U-U center-distance Top-K, U/U-only temporal edges, ignored-node exclusion, deterministic ties, and mandatory self loops.
- [ ] **Step 2: Run tests to verify RED** with:
  `python -m unittest tests.test_graph_stage2_builder_unittest -v`
  Expected: import failure for `models.graph.graph_builder`.
- [ ] **Step 3: Implement flattened current-node metadata and deterministic candidates.** Keep an index map back to `(sample, time, mask_index)`. Call EdgeNet only for GT-U candidates with relation ID zero, filter `sigmoid(logit) >= edge_threshold`, use bounded reliability `1.0` for self/temporal and a distance-derived `[eps,1]` prior for spatial.
- [ ] **Step 4: Run builder tests to GREEN** and existing Stage1 edge tests.
- [ ] **Step 5: Commit** builder and tests.

### Task 2: Relation-Aware GAT

**Files:**
- Create: `models/graph/relation_gat.py`
- Create: `tests/test_relation_gat_unittest.py`

**Interfaces:**
- Consumes: current nodes `[N,D]`, `SparseGraphEdges`, relation IDs `propagation=0`, `spatial=1`, `temporal=2`, `self=3`.
- Produces: `RelationAwareGAT.forward(nodes, source, destination, relation, reliability) -> [N,D]`.

- [ ] **Step 1: Write failing tests** for empty nodes, one node/self loop, per-destination normalization, invalid indices, CUDA autocast finiteness, and nonzero input/module gradients.
- [ ] **Step 2: Run tests to verify RED.**
- [ ] **Step 3: Implement one-layer four-head GAT.** Use Q/K/V projections, relation embeddings, `beta * log(reliability.clamp_min(1e-6))`, a deterministic FP32 segment softmax grouped by destination/head, FP32 `index_add_` aggregation, output projection, dropout, residual, and LayerNorm. Reject non-finite inputs/outputs.
- [ ] **Step 4: Run focused tests to GREEN.**
- [ ] **Step 5: Commit** GAT and tests.

### Task 3: Hard Reprojection and Teacher Veto

**Files:**
- Create: `models/graph/pseudo_fusion.py`
- Create: `tests/test_graph_pseudo_fusion_unittest.py`

**Interfaces:**
- Produces: `hard_reproject_u_nodes(masks, states, node_pred, node_conf, node_valid, ignore_index=255)` returning pixel prediction/confidence/valid; `teacher_graph_veto(...)` returning detached target, valid mask, and diagnostics; `graph_pixel_pseudo_loss(logits_a, logits_b, target, valid, ignore_index=255)`.

- [ ] **Step 1: Write failing tests** for overlap priority `(confidence, smaller area, stable index)`, GT/IGNORE exclusion, teacher agreement acceptance, graph conflict veto, graph-invalid fallback with stricter threshold, changed-pixel leakage rejection, class-zero rejection, and connected zero loss.
- [ ] **Step 2: Run tests to verify RED.**
- [ ] **Step 3: Implement hard reprojection.** Iterate stable mask order after sorting by confidence descending and area/index ascending, updating only where the candidate outranks the recorded owner.
- [ ] **Step 4: Implement teacher veto.** Require A/B teacher agreement, both base thresholds, unchanged gate, GT-defined unchanged pixels, and semantic class >0. Effective graph predictions veto conflicts; absent graph permits fallback only when both teacher confidences pass `teacher_fallback_threshold`. Assert zero accepted changed pixels.
- [ ] **Step 5: Implement detached CE.** Build A/B ignore-index targets from the accepted class target and average only nonempty terms so gradients reach student logits.
- [ ] **Step 6: Run focused tests to GREEN and commit.**

### Task 4: Stage2 Orchestration

**Files:**
- Create: `models/graph/stage2.py`
- Create: `tests/test_graph_stage2_unittest.py`

**Interfaces:**
- Produces: `Stage2Result` containing graph A/B probs, pred/conf/valid, GT/temporal losses, node/edge counts, mean reliability, graph GT accuracy, and per-sample node metadata.
- Produces: `SamGraphStage2(stage1, node_dim, heads, dropout, local_topk, edge_threshold, edge_bias_beta).forward(node_outputs, audits)`.

- [ ] **Step 1: Write failing tests** for prototype class-zero/uninitialized masking, finite graph GT loss, empty-GT connected zero, U/U JS consistency, graph output split back to A/B masks, and no second prototype bank.
- [ ] **Step 2: Run tests to verify RED.**
- [ ] **Step 3: Implement Stage2.** Build sparse edges, run GAT, classify only when at least one prototype is initialized, avoid all-negative-infinity softmax by marking invalid nodes before softmax, compute GT CE and symmetric KL/JS in FP32, and preserve differentiability of current node inputs.
- [ ] **Step 4: Run Stage2 and Stage1 tests to GREEN and commit.**

### Task 5: Runtime, CLI, and Checkpoints

**Files:**
- Modify: `utils/graph_stage1_training.py`
- Modify: `main_model.py`
- Modify: `tests/test_graph_stage1_training_unittest.py`
- Create: `tests/test_graph_stage2_checkpoint_unittest.py`

**Interfaces:**
- Extend `GraphStage1TrainingRuntime.compute(..., stage2_enabled=False)` to return a combined runtime result without changing disabled behavior.
- Extend graph state to format version 2 while retaining key `sam_graph_stage1`.

- [ ] **Step 1: Write failing CLI tests** for all required Stage2 flags/defaults and invalid combinations.
- [ ] **Step 2: Write failing checkpoint tests** proving v1 restores Stage1 exactly and reports fresh Stage2, while v2 round-trips GAT/optimizer/config/counters.
- [ ] **Step 3: Add opt-in CLI arguments** in `add_graph_stage1_arguments`: Stage2 enable/start, Top-K, edge threshold/beta, heads/dropout, temp/pixel weights, pseudo ramp, graph confidence, teacher fallback threshold.
- [ ] **Step 4: Preserve forward compatibility.** Add an optional graph-feature detach control to `ChangeDetection.forward`; default keeps the existing return and detached tensors, Stage2 training requests live graph features only.
- [ ] **Step 5: Extend runtime.** Instantiate Stage2 only when enabled, add its parameters to the existing graph optimizer, use detached node copies for Stage1 and live nodes for Stage2, and expose Stage2 result plus masks/audits for fusion.
- [ ] **Step 6: Upgrade checkpoint state.** Version 1 restores old projector/Stage1/optimizer/queue and initializes Stage2 fresh; version 2 restores Stage2 strictly after num-class/node/head checks. Handle optimizer state incompatibility from added parameters by rebuilding fresh graph optimizer state only for the v1 upgrade while preserving Stage1 tensors.
- [ ] **Step 7: Run runtime/checkpoint tests to GREEN and commit.**

### Task 6: Training Integration and Diagnostics

**Files:**
- Modify: `train.py`
- Modify: `tests/test_training_performance_unittest.py` or create `tests/test_graph_stage2_training_unittest.py`

**Interfaces:**
- Stage2 runs only when `use_sam_graph_stage2` and `curr_epoch >= graph_stage2_start_epoch`.
- `graph_pixel_pseudo_weight(curr_epoch)` ramps from zero to the CLI target over `graph_pseudo_ramp_epochs`.

- [ ] **Step 1: Write failing integration tests** for disabled regression, stage boundary, ramp values, no validation call, and graph pixel CE producing nonzero student semantic/backbone gradient in a tiny differentiable model.
- [ ] **Step 2: Run tests to verify RED.**
- [ ] **Step 3: Integrate the forward path.** Request live graph features only during active Stage2; run teacher inference under `inference_mode`; compute Stage1 and Stage2 results; reproject U predictions; veto teacher candidates; add independently normalized graph GT, temporal, and pixel pseudo losses.
- [ ] **Step 4: Integrate one atomic optimizer boundary.** Combined backward, unscale both optimizers, clip/check both grad norms, skip both optimizers on any non-finite state, then step both and update EMA teacher.
- [ ] **Step 5: Add diagnostics** to console/TensorBoard: state/edge counts, mean reliability, graph GT accuracy/confidence, graph valid pixels, teacher candidate/veto/fallback/accepted ratios, class histogram, three Stage2 losses, grad norms, and skip count.
- [ ] **Step 6: Run focused and full unittest suites to GREEN and commit.**

### Task 7: Smoke Verification

**Files:**
- Create or update: `tests/test_graph_stage2_smoke_unittest.py`

- [ ] **Step 1: Run syntax compilation** for every created/modified Python file.
- [ ] **Step 2: Run all `*_unittest.py` tests** and record the passing count.
- [ ] **Step 3: Run a synthetic Stage2 forward/backward/optimizer step** and assert finite losses/gradients, nonzero pixel-pseudo student gradient, and zero changed-pixel leakage.
- [ ] **Step 4: Load one real SAM cache sample** from the selected dataset and run graph construction/reprojection.
- [ ] **Step 5: Run one real training batch forward/backward/optimizer step** on an available GPU, without validation or checkpoint overwrite.
- [ ] **Step 6: Load a real Stage1 v1 checkpoint** and verify a Stage2-enabled runtime restores Stage1 and initializes Stage2 fresh.
- [ ] **Step 7: Report only changed files, implemented/excluded scope, tests, smoke result, blockers, and the recommended training command.
