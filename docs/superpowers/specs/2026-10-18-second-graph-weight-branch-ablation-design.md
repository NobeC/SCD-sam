# SECOND Graph Pseudo-Label Weight Branch Ablation Design

## Goal

Determine whether complementary graph pseudo-label supervision improves SECOND validation metrics when all branches resume from the same epoch-54 training state.

## Experimental Design

Resume all branches from:

`exp21_second_graph_pseudo_complement_alpha01_03_20261016/output/checkpoints/SECOND/SECOND_pseudo_graph_stage2_seed42_epoch054_Fscd0.5961.pth`

Run three branches through epoch 69, so validation results cover epochs 55 through 69:

| Branch | `graph_pixel_pseudo_weight` | Physical GPU |
|---|---:|---:|
| control | 0.00 | 0 |
| graph02 | 0.02 | 2 |
| graph05 | 0.05 | queued on the first GPU that becomes idle |

The checkpoint restores student, EMA teacher, graph modules, optimizers, scheduler, class precision, random-number state, and the next epoch index. Each branch must use a private output directory and a private source snapshot. Existing experiments, checkpoints, and Landsat behavior must remain unchanged.

## Controlled Configuration

Apart from `graph_pixel_pseudo_weight`, every branch uses the exp21 configuration:

- Dataset: SECOND, 7 classes
- Seed: 42
- Epoch range: resume at 55 and stop after epoch 69 (`--epochs 70`)
- Physical batch size: 2; effective batch size: 8
- Input size: 512; validation batch size: 2
- Learning rate and scheduler state: restored from the epoch-54 checkpoint
- EMA decay: 0.99
- Semantic pseudo-label weight: 0.10
- Semantic pseudo-label schedule: start 40, ramp 5, alpha 0.10 to 0.30
- Graph Stage1 node loss: 0.10; edge loss: 0.10
- Graph Stage2 start: epoch 55; graph ramp: 10 epochs
- Graph temporal weight: 0.02
- Complementary semantic/graph fusion
- AMP enabled, gradient checkpointing disabled

The control branch keeps Graph Stage1, Graph Stage2 propagation, and all graph diagnostics active. Only its pixel-level graph pseudo-label loss coefficient is zero. This isolates the effect of the graph pseudo-label gradient while keeping data loading and graph computation comparable.

## Launch and Isolation

Create one experiment root containing a frozen training script, complement helper, launcher, manifest, logs, and separate output directories for each branch. The launcher must:

1. Refuse to overwrite an existing branch output or PID marker.
2. Verify the requested physical GPU is idle before launch.
3. Record the exact command, source hashes, checkpoint path, branch weight, GPU, and process ID.
4. Launch detached processes with unbuffered logging.
5. Support launching `graph05` later without changing its configuration.

## Preflight Verification

Before production launch:

1. Verify the epoch-54 checkpoint exists.
2. Verify its stored next epoch is 55 and its dataset/class count are SECOND/7.
3. Run existing complement unit tests.
4. Run one smoke batch for one nonzero-weight branch in a temporary output directory.
5. Confirm the smoke log reports SECOND, seven semantic classes, restored epoch 55, restored graph state, and the requested graph weight.

Temporary smoke artifacts must not be mixed with production outputs.

## Evaluation and Decision Rule

For every completed epoch, collect Fscd, mIoU, Sek, OA, validation loss, learning rate, pseudo-label ratios, graph class counts, and non-finite skip counts.

Primary comparison:

- Compare each nonzero branch against the control at the same epoch.
- Report best Fscd through epoch 69 and the mean over epochs 65–69.
- A branch is a promising gain only if both Fscd and Sek exceed the control, without increased non-finite failures.

This single-seed short ablation is a screening experiment, not final evidence. A promising branch must subsequently be reproduced with at least three seeds before claiming a reliable gain.

## Failure Handling

- Do not silently restart from a different checkpoint.
- Keep logs and checkpoints when a branch fails.
- If a branch stops, record its last completed epoch and error before relaunching.
- Do not reuse an output directory for a replacement run.

## Out of Scope

This experiment does not add class balancing, per-class precision gates, change SAM area thresholds, modify the model, or alter Landsat settings. Those changes require separate ablations after the graph-weight effect is isolated.
