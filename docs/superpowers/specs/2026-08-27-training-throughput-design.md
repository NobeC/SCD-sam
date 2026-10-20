# SCD-SAM Training Throughput Design

## Goal

Reduce Landsat training time without changing gradient scaling, sample coverage,
loss definitions, validation metrics, checkpoint compatibility, or the existing
AMP stability fixes.

## Approaches Considered

1. **Safe dataflow optimization (selected).** Reuse MobileSAM features already
   produced by the student, keep training statistics on the GPU until logging,
   overlap host-to-device copies with loading, and sample expensive diagnostics.
   This removes redundant work while preserving the active graph and pseudo-label
   objectives.
2. **Disable graph Stage1 during pseudo training.** This gives the largest simple
   speedup, but changes the active objective and does not exercise the graph
   branch, so it remains a command-level experimental option rather than the
   implementation default.
3. **Deep graph vectorization.** Vectorize mask pooling, audits, and edge sampling.
   This can improve throughput further but changes several numerically sensitive
   paths and is deferred until the safe changes are benchmarked.

## Design

### Reuse Student MobileSAM Features

`ChangeDetection.forward` accepts an opt-in `return_graph_features` flag. The
normal return contract stays unchanged. When enabled, it additionally returns
detached raw MobileSAM stage-3 and neck tensors for A and B. `train.py` requests
them only when graph Stage1 is enabled and passes them into
`GraphStage1TrainingRuntime.compute`. The runtime uses supplied tensors and falls
back to its current backbone provider for tests and standalone callers.

This removes the graph branch's second A/B MobileSAM forward without allowing
graph losses to update the backbone, matching the existing `detach=True`
semantics.

### Data Loading and Transfers

DataLoader construction enables `persistent_workers=True` and
`prefetch_factor=2` only when `num_workers > 0`. It preserves
`drop_last=False`. CUDA transfers use `non_blocking=True` with pinned DataLoader
memory. Training retains the original CPU label tensors for graph node audits,
avoiding an immediate GPU-to-CPU round trip.

### Training Metrics

Training predictions and accuracy remain on CUDA. Exact correct/total statistics
and detached loss sums are accumulated as tensors and synchronized only at log
boundaries and epoch end. Validation continues to collect NumPy predictions
because `SCDD_eval_all` consumes them. Change masks use the equivalent logit test
`out_change > 0`, avoiding sigmoid and transferring boolean predictions.

### Pseudo-label Overhead

Teacher forward uses `torch.inference_mode()`. Pseudo visualization and the
six-ratio alpha sweep become independently configurable sampling operations with
conservative defaults. The actual pseudo-label selection and loss still run on
every eligible iteration.

### Numerical and Training Invariants

- Keep the current per-window gradient divisor, including the final incomplete
  accumulation window.
- Keep `drop_last=False` and train on all 1908 Landsat samples.
- Keep separate autocast scopes and explicit FP32 prototype/EdgeNet paths.
- Keep all loss weights, optimizer steps, EMA updates, and checkpoint payloads
  unchanged.

## Error Handling

- Loader-only multiprocessing options are omitted when `num_workers == 0`.
- Cached graph feature shapes and finiteness are validated without per-feature
  Python CUDA synchronization in the hot path; the existing aggregate
  non-finite-loss guard remains authoritative.
- Standalone graph callers that do not supply cached features continue to work.

## Tests and Verification

1. Unit test that supplied graph features bypass the backbone and produce a
   finite, differentiable graph loss.
2. Unit test conditional DataLoader kwargs for zero and nonzero workers.
3. Unit test exact CUDA/CPU-equivalent training accuracy accumulation.
4. Unit test pseudo diagnostic cadence.
5. Run each test red then green, the full unit suite, Python compilation, and the
   real MobileSAM graph AMP smoke test.
6. Run a short benchmark with identical checkpoint, seed, batch size, and active
   losses before making throughput claims.

