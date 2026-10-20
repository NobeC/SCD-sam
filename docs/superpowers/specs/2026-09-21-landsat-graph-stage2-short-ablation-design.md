# Landsat Graph Stage2 Short Ablation Design

## Goal

Determine whether the current Landsat Graph Stage2 improvement comes from graph-confirmed pseudo labels or primarily from the EMA-teacher fallback path, without disturbing the two active full-length experiments.

## Common Starting Point

All three runs resume from the same pre-Stage2 checkpoint:

`exp5_landsat_gt_only_gc_20260920/checkpoints/Landsat/Landsat_pseudo_graph_stage2_seed42_epoch059_Fscd0.8074.pth`

The checkpoint is the exp5 epoch-59 state. Stage2 starts at epoch 60. Each run resumes complete training state and stops after epoch 74 by setting `--epochs 75`.

Except for the single ablated setting described below, every dataset, model, optimizer, scheduler, EMA, semantic-pseudo, SAM cache, Stage1 graph, data-loader, AMP, and gradient-accumulation option matches exp5.

## Experiment Groups

### A: Stage1-only control

Disable Graph Stage2 while retaining Graph Stage1, EMA teacher, and semantic pseudo labels. This measures the continuation trajectory without Stage2.

### B: Current exp5 Stage2

Enable Stage2 with the current exp5 configuration, including `--teacher-fallback-threshold 0.95`. This is the positive treatment and reproduces the current Stage2 behavior from the shared checkpoint.

### C: Graph-confirmation-focused Stage2

Use the same Stage2 configuration as group B, changing only `--teacher-fallback-threshold 0.999`. This suppresses most teacher-only fallback while preserving teacher candidates confirmed by valid, non-conflicting graph evidence.

## Scheduling and Isolation

- Use currently idle physical GPUs 0 and 3.
- Start A and B concurrently on separate GPUs.
- Start C on whichever of those GPUs becomes free first.
- Give every group its own experiment directory, checkpoint directory, result directory, TensorBoard directory, PID file, and stdout/stderr log.
- Do not terminate, modify, or reuse outputs from the active exp5 and exp6 full runs.

## Monitoring and Failure Handling

- Confirm each process reaches multiple optimizer steps and produces throughput output.
- Record CUDA memory use and GPU utilization after startup.
- If a run encounters CUDA OOM or exits before producing training iterations, preserve its log and report the failure; do not silently change batch size or enable a different configuration.
- If non-finite skips grow rapidly, retain the run for diagnosis but do not reinterpret skipped updates as useful training.
- Group C starts only after a GPU is genuinely free and the prior process exit status is known.

## Comparison

Compare all groups at matched epochs using:

- best and final Fscd, IoU, Sek, and accuracy;
- non-finite skip count;
- step time;
- graph-valid, teacher-candidate, fallback, and accepted ratios;
- graph and student gradient diagnostics on optimizer-update steps.

Interpretation:

- B above A and C above A: graph-confirmed supervision is useful.
- B above A but C not above A: most benefit comes from teacher fallback.
- B not above A: current Stage2 does not improve the shared baseline.
- C has high precision but negligible coverage: improve graph coverage before increasing graph-loss weights.

## Success Criteria

The ablation is operationally successful when all three runs complete epochs 60 through 74 from the identical checkpoint with isolated outputs and comparable validation records. It is scientifically positive only if a Stage2 group exceeds A by more than ordinary epoch-to-epoch validation noise and does not incur materially worse numerical instability.
