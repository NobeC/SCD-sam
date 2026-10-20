# Landsat Graph Stage2 Short Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run three isolated Landsat continuations from the same epoch-59 checkpoint through epoch 74 to distinguish Stage2, graph-confirmed, and teacher-fallback effects.

**Architecture:** Each group runs from an isolated lightweight experiment directory containing symlinks to shared source/data and private output directories. A and B run concurrently on physical GPUs 0 and 3; a queue process starts C on GPU 3 only after B exits successfully.

**Tech Stack:** Bash, CUDA_VISIBLE_DEVICES, Python 3.8/PyTorch training entry point, NVIDIA RTX 3090, existing SCD-SAM checkpoint format.

## Global Constraints

- Resume every group from `exp5_landsat_gt_only_gc_20260920/checkpoints/Landsat/Landsat_pseudo_graph_stage2_seed42_epoch059_Fscd0.8074.pth`.
- Run epochs 60 through 74 by passing `--epochs 75`.
- Preserve all active exp5 and exp6 processes and outputs.
- Use physical GPUs 0 and 3 only.
- Keep every option identical to exp5 except the named A/B/C ablation.
- A retains a configured Stage2 runtime for checkpoint compatibility but sets `--graph-stage2-start-epoch 75`, so Stage2 never executes during the short run.
- C changes only `--teacher-fallback-threshold` from `0.95` to `0.999` relative to B.

---

### Task 1: Create isolated experiment roots

**Files:**
- Create: `ablation_landsat_stage2_20260921/A_stage1_control/`
- Create: `ablation_landsat_stage2_20260921/B_stage2_current/`
- Create: `ablation_landsat_stage2_20260921/C_stage2_graph_confirmed/`

**Interfaces:**
- Consumes: shared source, datasets, SAM cache, weights, and epoch-59 checkpoint.
- Produces: three independent working directories whose `working_path` redirects checkpoints, logs, results, and visualizations.

- [ ] **Step 1: Verify source checkpoint and idle GPUs**

Run:

```bash
test -f exp5_landsat_gt_only_gc_20260920/checkpoints/Landsat/Landsat_pseudo_graph_stage2_seed42_epoch059_Fscd0.8074.pth
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Expected: checkpoint test exits 0; GPUs 0 and 3 show negligible memory and utilization.

- [ ] **Step 2: Create roots, private output folders, and shared symlinks**

For each of `A_stage1_control`, `B_stage2_current`, and `C_stage2_graph_confirmed`, create `checkpoints`, `logs`, `results`, and `visualizations`, then symlink `MobileSAM`, `configs`, `datasets`, `main_model.py`, `models`, `sam_mask_cache`, `train.py`, `utils`, and `weights` to their absolute paths under `/home/homenew/yang/SCD/SCD-SAM/SCD`.

- [ ] **Step 3: Verify isolation**

Run:

```bash
find ablation_landsat_stage2_20260921 -maxdepth 2 -type l -printf '%p -> %l\n'
find ablation_landsat_stage2_20260921 -maxdepth 2 -type d -printf '%p\n'
```

Expected: each group has nine shared symlinks and four private output directories.

### Task 2: Launch A and B, with C queued behind B

**Files:**
- Create: `ablation_landsat_stage2_20260921/A_stage1_control/logs/train.log`
- Create: `ablation_landsat_stage2_20260921/B_stage2_current/logs/train.log`
- Create: `ablation_landsat_stage2_20260921/C_stage2_graph_confirmed/logs/train.log`
- Create: per-group `train.pid`; create `BC_queue.pid` in the ablation root.

**Interfaces:**
- Consumes: isolated roots from Task 1 and the common checkpoint.
- Produces: A on GPU 0, B followed by C on GPU 3, with durable logs and PID records.

- [ ] **Step 1: Define the common training arguments**

Use the exp5 values exactly:

```text
--dataset landsat --num-classes 5 --cuda 0 --seed 42 --epochs 75
--batch-size 2 --effective-batch-size 8 --val-batch-size 2 --input-size 512
--num-workers 8 --no-persistent-workers --timing-interval 50
--learning-rate 3.5e-4 --grad-clip 1.0 --amp --gradient-checkpointing
--cudnn-benchmark --use-ema-teacher --ema-decay 0.999
--use-semantic-pseudo --pseudo-start-epoch 30 --pseudo-rampup-epoch 10
--lambda-sem-u 0.10 --pseudo-alpha-start 0.10 --pseudo-alpha-end 0.30
--pseudo-min-ratio 0.02 --pseudo-change-thr 0.30
--pseudo-precision-init 0.80 --pseudo-precision-momentum 0.90
--use-sam-graph-stage1 --use-sam-graph-stage2 --sam-cache-root sam_mask_cache
--sam-node-min-area-ratio 0.005 --sam-node-max-area-ratio 0.10
--sam-tau-gt 0.70 --sam-tau-purity 0.85
--sam-node-min-mask-pixels 4 --sam-node-min-changed-pixels 4
--lambda-graph-gt 0.10 --lambda-edge-gt 0.10
--graph-local-topk 4 --graph-edge-threshold 0.70 --graph-edge-bias-beta 1.0
--graph-temp-weight 0.02 --graph-pixel-pseudo-weight 0.05
--graph-pseudo-ramp-epochs 20 --graph-confidence-threshold 0.70
--graph-teacher-threshold 0.90
```

Every run also receives the absolute common `--resume` path.

- [ ] **Step 2: Launch A on physical GPU 0**

Append `--graph-stage2-start-epoch 75 --teacher-fallback-threshold 0.95`. Run under `nohup` from A's directory with `CUDA_VISIBLE_DEVICES=0`, redirect both streams to `logs/train.log`, and write the process PID to `train.pid`.

- [ ] **Step 3: Launch the B-then-C queue on physical GPU 3**

Run one durable background shell with `CUDA_VISIBLE_DEVICES=3`. It launches B using `--graph-stage2-start-epoch 60 --teacher-fallback-threshold 0.95`, waits for B, and only on exit code 0 launches C using `--graph-stage2-start-epoch 60 --teacher-fallback-threshold 0.999`. Each Python PID is written to its group's `train.pid`; the queue-shell PID is written to `BC_queue.pid`.

- [ ] **Step 4: Confirm processes and command lines**

Run:

```bash
ps -o pid,ppid,state,etimes,time,pcpu,pmem,args -p "$(cat ablation_landsat_stage2_20260921/A_stage1_control/train.pid)" "$(cat ablation_landsat_stage2_20260921/B_stage2_current/train.pid)"
```

Expected: A and B Python processes exist; A has Stage2 start 75, B has Stage2 start 60, and both use the same checkpoint and all other common arguments.

### Task 3: Verify forward progress and resource isolation

**Files:**
- Inspect: A and B `logs/train.log`
- Inspect: A and B output directories

**Interfaces:**
- Consumes: live A/B processes.
- Produces: evidence that both jobs restored epoch 60, reached training iterations, use separate GPUs, and write isolated outputs.

- [ ] **Step 1: Wait for initialization without blocking longer than 60 seconds per poll**

Poll both logs until each contains `Resumed complete training state`, followed by at least one `[throughput]` line or an explicit failure traceback.

- [ ] **Step 2: Validate checkpoint restoration**

Expected in both logs:

```text
Resumed complete training state from ...epoch059_Fscd0.8074.pth at epoch 60
SAM graph Stage1 state restored from checkpoint.
```

- [ ] **Step 3: Validate GPU activity**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
```

Expected: physical GPUs 0 and 3 have allocated memory and nonzero compute activity; existing GPUs 1 and 2 remain owned by exp5 and exp6.

- [ ] **Step 4: Validate A/B semantic difference**

After the first sampled graph log, A must have no `[sam-graph-stage2]` line because Stage2 starts at 75. B must emit `[sam-graph-stage2]` with weight ramping from zero at epoch 60.

- [ ] **Step 5: Report paths and monitoring commands**

Report the A/B PIDs, queue PID, current GPU memory/utilization, and clickable log paths. State that C is queued and will launch only after B exits successfully.

### Task 4: Completion comparison

**Files:**
- Inspect: all three `logs/train.log`
- Inspect: all three `checkpoints/Landsat/`

**Interfaces:**
- Consumes: completed A/B/C runs.
- Produces: matched-epoch comparison and a Stage2/fallback conclusion.

- [ ] **Step 1: Confirm all jobs ended normally**

Each log must contain `Training finished.` and validation through epoch 74. If not, report the exact final log line and exit condition.

- [ ] **Step 2: Extract comparable metrics**

For each group, extract best and epoch-74 Fscd, IoU, Sek, accuracy, non-finite skip count, and median sampled step time.

- [ ] **Step 3: Interpret against the approved decision table**

Apply the B-vs-A and C-vs-A rules from the design specification. Do not claim a graph benefit solely from teacher-fallback accepted pixels.
