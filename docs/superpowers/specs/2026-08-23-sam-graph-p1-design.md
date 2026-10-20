# SAM Graph P1 Representation Learning Design

## Scope

P1 implements the train-only representation-learning components needed for graph Stage 1. It consumes the P0 shared mid-scale masks and GT/U/ignore audit. It does not modify `ChangeDetection`, `train.py`, validation, inference, or the existing checkpoint payload.

P1 includes stable MobileSAM stage/neck feature extraction, mask region pooling, 128-dimensional temporal node projection, GT-only EMA prototypes, GT-only supervised EdgeNet, a class-balanced detached memory queue, and a Stage 1 loss coordinator.

P1 excludes GAT, unlabeled propagation, temporal KL, node-to-pixel projection, teacher veto, and semantic pseudo labels.

## Isolation and feature extraction

`SamGraphFeatureProvider` is an independent module that receives the existing MobileSAM backbone and calls it for image A and image B. It consumes stable outputs already returned by the encoder:

- stage3: index 2, `[B, 320, 32, 32]` for a 512-pixel input;
- encoder neck: index 4, `[B, 256, 16, 16]`.

The default path detaches both feature maps before region pooling. This protects the working Landsat backbone, avoids retaining a second backbone computation graph, and keeps graph losses from changing the main network in P1.

This approach performs an additional backbone forward when the graph branch is enabled. It is accepted for P1 because it preserves the original main-model interface. A future optimized integration may expose already-computed features from the main forward behind an opt-in flag.

Exact Block5/9/12 hooks are prohibited.

## Region pooling and geometry

Each sample owns a variable-length boolean mask tensor `[N,H,W]`. Masks are mapped independently to stage3 and neck resolution with nearest-neighbor interpolation. A mapped mask containing no valid feature cell is marked invalid and excluded.

For every valid mask and feature level, pooling computes FP32 masked mean and masked max. Max pooling uses a finite masked fill and validates that the mask is nonempty before reduction.

Geometry contains five normalized FP32 values:

- center x;
- center y;
- bounding-box width;
- bounding-box height;
- area ratio.

All values are normalized to `[0,1]` using the source mask dimensions.

## Temporal node projector

Stage3 mean/max and neck mean/max are concatenated into a single-time feature `f_t`. For each shared mask, the projector builds:

```text
[f_t, abs(f_A-f_B), f_A*f_B, geometry, time_embedding, scale_embedding]
```

P1 supports the mid scale only but retains a three-slot scale embedding for future fine/mid/coarse extension. A and B use different learnable time embeddings. The feature is processed by LayerNorm and a two-layer MLP to node dimension 128.

The projector returns aligned `h_A`, `h_B`, and a validity mask. A mask must be valid at both selected feature levels to participate.

## EMA prototypes and graph classification

`EMAPrototypeBank` registers these buffers:

- `prototypes`: `[num_classes,128]`, FP32;
- `initialized`: `[num_classes]`, boolean.

Class 0 is permanently invalid and is never updated or predicted. Updates consume detached, normalized GT node embeddings only. Each present semantic class produces one batch mean and updates its slot with normalized EMA momentum, default `0.99`.

Graph logits use FP32 cosine similarity divided by a configurable positive temperature. Uninitialized class logits are masked. If no semantic prototype is initialized or no GT node is eligible, prototype CE returns a zero connected to the node computation graph.

Prototype slots map directly to dataset class IDs, so Landsat uses five slots and SECOND uses seven without hardcoding.

## Class-balanced GT memory queue

`ClassBalancedNodeQueue` maintains a bounded FIFO queue for every semantic class `1..num_classes-1`. Enqueued embeddings are normalized, detached, moved to CPU, and limited to a configurable per-class capacity.

The queue never stores class 0, unlabeled nodes, ignore nodes, or pseudo labels. Sampling transfers selected embeddings to the active node device. Empty and rare-class queues are valid states.

## GT-only EdgeNet

`EdgeNet` predicts same-semantic reliability from:

```text
[h_i, h_j, abs(h_i-h_j), h_i*h_j, relation_embedding]
```

P1 has one GT-GT relation ID but keeps the relation embedding interface extensible. The network is a two-layer MLP producing logits; training uses `binary_cross_entropy_with_logits` in FP32 rather than applying sigmoid before BCE.

Pair construction uses current GT nodes plus detached queue nodes. It selects balanced same-class positives and cross-class negatives. When available, a configurable portion of negatives are hard negatives with high cosine similarity. Pair targets can only be produced from true GT class IDs.

If either positive or negative pairs are unavailable, the sampler uses the available side without inventing targets. If no pair exists, edge loss is a zero connected to current node embeddings and EdgeNet parameters.

## Stage 1 coordinator

`SamGraphStage1` owns the node projector, prototype bank, EdgeNet, and memory queue. It consumes pooled A/B features and P0 audit states/classes for a list of samples.

For each temporal node whose state is GT, it:

1. selects the corresponding projected embedding;
2. updates prototypes with detached embeddings;
3. computes prototype classification loss where valid;
4. constructs GT-only EdgeNet pairs using the current batch and queue;
5. computes balanced edge BCE;
6. enqueues current detached GT embeddings after pair construction.

It returns:

- `loss_graph_gt`;
- `loss_edge_gt`;
- GT node count;
- positive and negative pair counts;
- initialized prototype mask.

The coordinator does not combine these values with the main loss in P1. The future training integration will apply `lambda_graph_gt` and `lambda_edge`.

## Numerical and error handling

- Mask interpolation, pooling accumulation, cosine similarity, prototype normalization, and EdgeNet BCE use FP32.
- Every normalization clamps its norm by an epsilon.
- Empty masks, empty nodes, absent GT classes, single-node batches, no positive pairs, and no negative pairs return finite results.
- All zero losses remain connected to graph module parameters or node embeddings.
- Queue tensors remain detached CPU tensors and never retain an old computation graph.
- Input shape, channel count, class ID, device, and finite-value validation fail with explicit errors.

## Testing

Tests follow red-green TDD and cover:

- stable stage3/neck selection and default detach;
- nearest mask mapping and invalid small masks;
- exact mean/max pooling and geometry;
- aligned A/B 128-dimensional output;
- dynamic Landsat/SECOND prototype slots;
- class-0 exclusion and per-class initialization;
- finite logits and empty prototype loss;
- FIFO queue capacity and detached CPU storage;
- GT-only balanced pair construction and hard negatives;
- EdgeNet connected zero loss;
- Stage 1 outputs for normal, empty-GT, one-class, and single-node inputs;
- absence of changes to `ChangeDetection.forward` and its state dict.

## Deferred work

Training-loop integration, graph optimizer/checkpoint persistence, GAT, GT-U/U-U edges, temporal consistency, graph pixel probabilities, EMA-teacher veto, semantic pseudo labels, multi-scale masks, exact block features, and main-forward feature reuse are deferred to P2 or a dedicated integration step.
