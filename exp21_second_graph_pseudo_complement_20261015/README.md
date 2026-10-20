# SECOND original pseudo-label configuration with full graph

Run: exp7_second_original_fullgraph_20260922

Reuse the original pseudo-label settings: seed 42, 100 epochs, learning rate
0.00035, EMA 0.999, lambda_sem_u 0.2, pseudo start 30, ramp 10, alpha 0.1 to
0.3, minimum ratio 0.02. Add the full graph configuration used by the earlier
100-epoch SECOND graph run. Start from the same pretrained initialization,
without resuming the finished baseline.

Historical execution settings are incomplete. Explicitly use micro-batch 2,
effective batch 8, validation batch 2, input size 512, 4 workers, AMP enabled,
gradient checkpointing disabled, persistent workers enabled, and gradient
clipping at 1.0, matching the established SECOND graph execution settings.

Create an isolated source snapshot. Preserve current numerical protections.
Preflight reproduced FP16 embedding projection overflow and empty-supervision
reduction overflow in the shared source. The new snapshot additionally computes
the DFI embedding projection and differentiable zero-loss reductions in FP32.
Validation must retry an invalid AMP batch in FP32 and reject persistent
non-finite outputs instead of dropping validation samples or repairing model
state while scoring. Record this evaluation change and source hashes.

Only retain the verified highest validation Fscd checkpoint after normal
completion. Keep candidates when training or evaluation fails.
