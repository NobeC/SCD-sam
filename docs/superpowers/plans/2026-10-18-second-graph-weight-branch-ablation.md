# SECOND Graph-Weight Branch Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a controlled SECOND ablation from the epoch-54 checkpoint with graph pixel pseudo-label weights 0.00, 0.02, and 0.05, preserving all other state and settings.

**Architecture:** Create an isolated experiment snapshot with a branch-aware launcher. The launcher validates checkpoint/GPU/output safety, constructs one canonical command whose only branch-dependent training value is `graph_pixel_pseudo_weight`, records a manifest, and launches each branch detached into a private output tree. GPU 0 and GPU 2 run the first two branches concurrently; the third branch is represented by a frozen queued command for later launch.

**Tech Stack:** Python 3.8, PyTorch, argparse, subprocess, pytest, NVIDIA CLI, existing SCD-SAM training/checkpoint utilities.

## Global Constraints

- Dataset is SECOND with exactly 7 semantic classes.
- Resume checkpoint is `exp21_second_graph_pseudo_complement_alpha01_03_20261016/output/checkpoints/SECOND/SECOND_pseudo_graph_stage2_seed42_epoch054_Fscd0.5961.pth`.
- Resume at epoch 55 and stop after epoch 69 with `--epochs 70`.
- Preserve existing experiments and Landsat behavior.
- The only training-value difference among branches is `graph_pixel_pseudo_weight`.
- Never overwrite an existing branch output or PID marker.

---

### Task 1: Branch-Aware Launcher and Tests

**Files:**
- Create: `exp22_second_graph_weight_ablation_20261018/launch_ablation.py`
- Create: `exp22_second_graph_weight_ablation_20261018/test_launch_ablation.py`
- Copy unchanged snapshot: `exp22_second_graph_weight_ablation_20261018/train_ablation.py`
- Copy unchanged snapshot: `exp22_second_graph_weight_ablation_20261018/graph_pseudo_complement.py`

**Interfaces:**
- Consumes: epoch-54 complete checkpoint and existing exp21 training implementation.
- Produces: `build_arguments(branch, weight) -> list[str]`, `validate_branch(branch, weight)`, and a CLI accepting `--branch`, `--weight`, `--gpu`, and `--smoke-steps`.

- [ ] **Step 1: Write failing launcher tests**

Test that the launcher rejects unknown branch names and negative weights, and that commands for weights 0.00 and 0.02 are identical after normalizing the weight argument and branch-private paths. Assert `--resume`, `--epochs 70`, `--dataset second`, `--num-classes 7`, and complement mode are present.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
/home/homenew/yang/anaconda3/envs/scd-sam-cuda12/bin/python -m pytest exp22_second_graph_weight_ablation_20261018/test_launch_ablation.py -q
```

Expected: failure because `launch_ablation.py` does not exist.

- [ ] **Step 3: Implement the launcher and source snapshot**

Copy the exp21 training and complement files byte-for-byte. Implement output protection, idle-GPU checking, checkpoint existence checking, detached launch, exact command recording, PID recording, and JSON manifest recording. For smoke mode, use a caller-provided temporary run directory and `--smoke-steps 1`.

- [ ] **Step 4: Run launcher unit tests**

Run the pytest command from Step 2.

Expected: all tests pass.

- [ ] **Step 5: Commit launcher and tests**

```bash
git add exp22_second_graph_weight_ablation_20261018
git commit -m "exp: add SECOND graph-weight ablation launcher"
```

### Task 2: Checkpoint and Smoke Preflight

**Files:**
- Read: epoch-54 checkpoint
- Temporary output: `/tmp/second_graph_weight_smoke_*`

**Interfaces:**
- Consumes: Task 1 launcher and snapshot.
- Produces: evidence that resume state and requested graph weight reach training correctly.

- [ ] **Step 1: Inspect checkpoint metadata**

Load the checkpoint on CPU and assert `epoch == 55`, dataset is SECOND, semantic head count is 7, and complete student/teacher/optimizer/scheduler/graph state is present.

- [ ] **Step 2: Run one-batch smoke test on an idle GPU**

Launch branch `smoke02` with weight 0.02, `--smoke-steps 1`, and a temporary run directory.

- [ ] **Step 3: Verify smoke evidence**

Confirm the log contains the resume checkpoint path, epoch 55, restored graph state, SECOND data counts, and no traceback. Remove only the temporary smoke directory after successful verification.

### Task 3: Production Launch and Monitoring Handoff

**Files:**
- Create: `exp22_second_graph_weight_ablation_20261018/control/*`
- Create: `exp22_second_graph_weight_ablation_20261018/graph02/*`
- Create: `exp22_second_graph_weight_ablation_20261018/queue_graph05.json`

**Interfaces:**
- Consumes: verified launcher and idle physical GPUs.
- Produces: two live processes plus an immutable queued specification for the third branch.

- [ ] **Step 1: Recheck GPU availability**

Require GPU 0 and GPU 2 to have at most 100 MiB memory use, at most 5% utilization, and no compute process.

- [ ] **Step 2: Launch control on GPU 0**

Use branch `control`, weight 0.00, full run through epoch 69.

- [ ] **Step 3: Launch graph02 on GPU 2**

Use branch `graph02`, weight 0.02, full run through epoch 69.

- [ ] **Step 4: Verify both processes**

Confirm both PIDs are alive, both logs have started, commands differ only in branch-private paths, GPU assignment, and graph pseudo-label weight, and neither log contains a traceback.

- [ ] **Step 5: Freeze graph05 queue entry**

Write its weight 0.05, checkpoint, canonical command hash, and policy to launch on the first idle GPU after either active branch completes. Do not start it while only two approved idle GPUs are available.

### Task 4: Result Extraction Contract

**Files:**
- Create: `exp22_second_graph_weight_ablation_20261018/summarize_results.py`
- Create: `exp22_second_graph_weight_ablation_20261018/test_summarize_results.py`

**Interfaces:**
- Consumes: branch `train.log` files.
- Produces: per-epoch CSV/JSON and comparison containing best Fscd, best Sek, epochs 65–69 means, and non-finite skip counts.

- [ ] **Step 1: Write failing parser tests using compact synthetic logs**

Cover complete epochs, incomplete final epochs, best-metric selection, mean calculation, and non-finite warning counts.

- [ ] **Step 2: Implement the minimal parser and comparison writer**

Parse existing validation line format without changing training code.

- [ ] **Step 3: Run parser and launcher test suites**

Expected: all tests pass.

- [ ] **Step 4: Commit result tooling**

```bash
git add exp22_second_graph_weight_ablation_20261018/summarize_results.py exp22_second_graph_weight_ablation_20261018/test_summarize_results.py
git commit -m "exp: add SECOND ablation result summarizer"
```
