# SECOND Training Speed and Restart Design

## Goal

Reduce the wall-clock step time of the active SECOND SAM Graph Stage1 training,
prevent an indefinitely idle training process, and resume from the existing
epoch-14 checkpoint without changing model objectives or checkpoint format.

## Current Evidence

- The active command trains SECOND with physical batch size 4, effective batch
  size 8, AMP, eight loader workers, and SAM Graph Stage1.
- Gradient checkpointing is explicitly enabled.
- GPU 0 holds about 11.7 GiB of 24 GiB, but stayed at 0% utilization throughout
  a ten-second sample.
- The training process waits in `futex_wait_queue`, and its DataLoader worker
  children are no longer present. This run is not making useful GPU progress.
- The existing August throughput work already reuses student MobileSAM features,
  uses pinned/non-blocking transfers, defers metric synchronization, and samples
  pseudo-label diagnostics. Those optimizations must be preserved.

## Selected Approach

Use a conservative, measurable optimization rather than changing the network or
losses:

1. Make memory-saving gradient checkpointing opt-in instead of the default.
2. Add a cuDNN benchmark switch suitable for fixed 512x512 inputs, while keeping
   deterministic mode available as an explicit reproducibility choice.
3. Make persistent DataLoader workers configurable so a restart can use ordinary
   workers if persistent-worker shutdown or IPC is unstable.
4. Add low-frequency phase timing that reports data wait and training compute
   time without synchronizing CUDA on every step.
5. Run a short smoke/throughput test first. Only after it demonstrates forward
   progress and acceptable memory use should the stale process be terminated and
   training resumed from the epoch-14 checkpoint.

## CLI and Behavior

- `--gradient-checkpointing` remains supported, but the default becomes disabled.
- Add mutually exclusive `--cudnn-benchmark` and `--deterministic` modes. The
  throughput-oriented default uses cuDNN benchmark because training uses fixed
  spatial dimensions. The deterministic option restores deterministic cuDNN
  behavior for exact-reproduction experiments.
- Add `--persistent-workers` and `--no-persistent-workers`. Persistence is only
  passed to DataLoader when `num_workers > 0`.
- Add `--timing-interval N`; zero disables timing. Timing synchronizes CUDA only
  at the selected interval and reports averaged data/compute duration, preventing
  instrumentation from materially slowing the hot path.

## Preserved Invariants

- Dataset remains SECOND with seven semantic classes.
- Physical batch size remains 4 and effective batch size remains 8 unless memory
  measurement proves batch 4 cannot fit without checkpointing.
- AMP, optimizer, scheduler, accumulation divisor, all losses and their weights,
  graph Stage1 behavior, validation metrics, and checkpoint payload remain
  unchanged.
- Existing Landsat behavior remains available; deterministic reproduction can be
  selected explicitly.
- No `torch.compile`, channels-last conversion, graph algorithm rewrite, or
  pseudo-label change is included in this iteration.

## Failure Handling and Restart

- The smoke run must complete several optimizer steps and report peak allocated
  and reserved CUDA memory plus average step time.
- If disabling checkpointing causes CUDA OOM at batch 4, retry batch 2 with the
  same effective batch size 8; do not silently re-enable checkpointing.
- Do not kill the current process until the replacement configuration passes the
  smoke run. Termination targets only PID 3290166 after re-validating its command.
- Resume from
  `checkpoints/SECOND/SECOND_supervised_graph_stage1_seed42_epoch014_Fscd0.5351.pth`.
- The production restart writes stdout and stderr to a timestamped log so future
  stalls can be diagnosed without relying on a terminal buffer.

## Verification

1. Unit-test worker-option combinations and timing cadence/helpers.
2. Run Python compilation and focused training-performance tests.
3. Run a bounded CUDA smoke benchmark using SECOND Graph Stage1.
4. Verify peak memory remains below the 24 GiB device limit and GPU utilization
   becomes nonzero during compute.
5. Re-check the final command, checkpoint, PID, and log path before restarting.

