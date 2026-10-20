# SECOND Training Speed and Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fixed-size SECOND Graph Stage1 training faster and observable, verify it fits on an RTX 3090 without checkpointing, then safely resume the existing run.

**Architecture:** Keep performance policy in small testable helpers in `utils/training_performance.py`; `train.py` only parses options, applies the selected CUDA backend mode, constructs loaders, and records sampled timings. Preserve all model, loss, accumulation, validation, and checkpoint behavior.

**Tech Stack:** Python 3.8, PyTorch, unittest, CUDA AMP.

## Global Constraints

- SECOND remains seven classes and resumes the epoch-14 Graph Stage1 checkpoint.
- Preserve physical batch 4/effective batch 8 unless the smoke run proves OOM.
- Preserve AMP, optimizer, scheduler, loss weights, graph behavior, metrics, and checkpoint schema.
- Do not add `torch.compile`, channels-last, graph rewrites, or pseudo-label changes.
- Revalidate PID 3290166 before terminating it; write the replacement run to a timestamped log.

---

### Task 1: Testable Performance Policy

**Files:**
- Modify: `tests/test_training_performance_unittest.py`
- Modify: `utils/training_performance.py`

**Interfaces:**
- `loader_worker_options(num_workers: int, persistent_workers: bool = True) -> dict`
- `configure_cudnn(deterministic: bool) -> dict[str, bool]`
- `IntervalTimer(interval: int)` with `start_data()`, `start_compute()`, and `finish_step(iteration, synchronize=None)` returning a sampled timing dictionary or `None`.

- [ ] Add failing tests that persistence can be disabled, cuDNN modes are mutually correct, invalid timing intervals fail, unsampled timing returns `None`, and a sampled timing returns nonnegative `data_seconds`, `compute_seconds`, and `step_seconds`.
- [ ] Run `python -m unittest tests.test_training_performance_unittest -v`; expect failures because the interfaces are missing.
- [ ] Implement the minimal helpers. `configure_cudnn(False)` sets benchmark true/deterministic false; `configure_cudnn(True)` does the inverse. `IntervalTimer` uses `time.perf_counter` and invokes the optional synchronization callback only on sampled steps.
- [ ] Re-run the focused tests and require all cases to pass.

### Task 2: Integrate CLI, Loader Safety, and Timing

**Files:**
- Modify: `train.py`
- Test: `tests/test_training_cli_unittest.py`

**Interfaces:**
- CLI defaults: checkpointing disabled, cuDNN benchmark enabled, persistent workers disabled, timing interval 50.
- CLI overrides: `--gradient-checkpointing`, `--deterministic`, `--persistent-workers`, `--timing-interval`.

- [ ] Add a subprocess CLI test for `python train.py --help` asserting the new switches are documented; verify the default-policy constants through exported parser/helper construction without initializing training.
- [ ] Run the focused test and confirm it fails on missing parser/default behavior.
- [ ] Refactor parser construction into `build_argument_parser()`, call `configure_cudnn(opt.deterministic)` from `main`, pass persistence into both loader option calls, and add sampled data/compute timing around each train iteration. Log timing to stdout and TensorBoard only at the requested interval.
- [ ] Run CLI and performance tests and require all cases to pass.

### Task 3: Static and CUDA Verification

**Files:**
- Verify: `train.py`, `utils/training_performance.py`, focused tests.

- [ ] Run `python -m py_compile train.py utils/training_performance.py tests/test_training_performance_unittest.py tests/test_training_cli_unittest.py` and require exit code 0.
- [ ] Run `python -m unittest discover -s tests -p '*_unittest.py' -v` and record pass/fail totals.
- [ ] Run a bounded SECOND Graph Stage1 smoke benchmark without checkpointing, persistence, validation, or checkpoint writes; require several optimizer steps, finite losses, and report peak allocated/reserved CUDA memory and mean step time.
- [ ] If batch 4 OOMs, repeat at batch 2/effective batch 8 and select that configuration; otherwise retain batch 4.

### Task 4: Safe Process Replacement

**Files:**
- Create at runtime: `logs/SECOND/restarts/<UTC timestamp>.log`

- [ ] Re-read `/proc/3290166/cmdline`, confirm it is the stale SECOND Graph Stage1 run, and archive its exact command and checkpoint.
- [ ] Send SIGTERM only to PID 3290166, wait briefly, and verify the PID exited and GPU 0 memory was released; do not affect the GPU 3 process.
- [ ] Start the verified configuration from `checkpoints/SECOND/SECOND_supervised_graph_stage1_seed42_epoch014_Fscd0.5351.pth` with unbuffered output redirected to the timestamped log.
- [ ] Verify the new PID remains alive, its log advances through multiple steps, GPU utilization is nonzero during compute, and report the PID/log path/selected memory and timing settings.

