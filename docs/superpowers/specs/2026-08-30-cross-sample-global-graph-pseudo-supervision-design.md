# Cross-Sample Global Graph Propagation and Pseudo Supervision Design

## Goal

Extend the completed SAM graph Stage1 into an online cross-sample representation branch that propagates semantic evidence from reliable GT nodes to unlabeled SAM nodes. Convert accepted graph predictions into pixel pseudo labels and use their semantic cross-entropy to update the student backbone.

The implementation must preserve the current backbone, supervised losses, EMA pseudo path, Stage1 checkpoint compatibility, AMP stability, and explicit Landsat/SECOND dataset switching.

## Scope

This version uses a mini-batch graph plus a bounded cross-batch memory bank. It does not construct a full-dataset graph or require PyTorch Geometric. Historical memory nodes are detached anchors; only current-batch nodes retain gradients.

Included:

- GT, unlabeled, and ignored SAM node separation using the existing audit result.
- Within-sample spatial/feature edges, temporal correspondence edges, GT-to-unlabeled edges, and current-to-memory global Top-K edges.
- A one-layer, four-head relation-aware sparse GAT implemented with PyTorch tensor operations.
- Prototype classification of propagated nodes.
- Hard graph-to-pixel reprojection with deterministic overlap resolution.
- Cross-space and cross-time pseudo-label agreement.
- Pixel pseudo-label loss that updates the student semantic path and backbone.
- Node pseudo loss, temporal consistency loss, diagnostics, CLI controls, and resumable state.

Excluded:

- Full-dataset offline graph optimization.
- Learnable soft pixel reprojection.
- Replacing the existing EMA semantic pseudo-label path.
- External graph libraries.

## Existing Contracts to Preserve

- `TemporalNodeProjector` produces paired A/B nodes with shape `[N, node_dim]` for each sample.
- `SamGraphStage1` owns the trained prototype projection, EMA prototypes, EdgeNet, and class-balanced GT queue.
- `sam_graph_stage1` format-version-1 checkpoints must load without mutation of their existing tensors.
- When global propagation is disabled, training behavior must remain unchanged.
- Dataset class zero remains the unlabeled/unchanged class and is never a graph prototype target.

## Architecture

### 1. Current-batch node set

For every shared SAM mask, create A and B nodes with the existing projector. The node audit assigns each temporal node one of:

- `GT`: reliable class in `[1, num_classes)`.
- `U`: valid node without a semantic GT class.
- `IGNORE`: mixed, unstable, invalid, or filtered node.

Ignored nodes do not receive predictions or enter the memory bank. GT nodes supervise Stage1 and graph classification. U nodes receive propagated predictions.

Stage1 receives detached raw node inputs so its established GT-only behavior remains stable. The propagation branch receives the same current-batch node tensors without detaching their backbone inputs.

### 2. Cross-sample memory bank

Use a bounded class-balanced FIFO bank. Each entry stores:

- normalized detached node embedding;
- detached semantic probability;
- predicted or GT class;
- confidence;
- sample identity and time index.

GT anchors are always eligible. U nodes enter only after graph/classification agreement and the configured confidence threshold. A per-class capacity prevents dominant classes from occupying the bank. The bank lives on CPU and snapshots only the candidates required on the active GPU.

The bank is updated after a successful optimizer step, never before pseudo labels for that step are computed.

### 3. Candidate edges

Build directed sparse edges into each current node:

| Relation | Endpoints | Candidate rule |
| --- | --- | --- |
| spatial | same sample and time | geometry-center Top-K among valid nodes |
| feature | same sample and time | cosine Top-K among valid nodes |
| temporal | paired A/B node from the same shared mask | one-to-one, U only |
| propagation | current GT/U combinations | feature Top-K, excluding ignored nodes |
| global | current node to memory node | cosine Top-K, excluding the same sample identity |

Duplicate edges are coalesced by `(source, target, relation)`. Self loops are added inside the GAT. Candidate construction is deterministic for tied scores.

The trained Stage1 EdgeNet remains unchanged with its single relation embedding so format-version-1 weights load strictly. It predicts semantic reliability for feature, propagation, and global candidates. Spatial and temporal priors provide their own bounded base weights. Relation identity is modeled by the GAT, not by changing Stage1 EdgeNet dimensions.

### 4. Relation-aware sparse GAT

Implement one residual layer with node dimension 128, four heads, relation-specific key/value projections, dropout 0.1, and LayerNorm. For an edge `j -> i`, attention combines:

- projected query/key compatibility;
- `log(clamped_edge_reliability)`;
- a learned relation bias.

Softmax normalization is performed independently over incoming edges for each destination and head. Similarity, attention normalization, and aggregation run in FP32 under AMP. Memory values are constants; current node values remain differentiable.

The output is a residual refinement of current nodes only. Memory nodes act as keys/values and are not updated by backpropagation.

### 5. Graph prediction

Classify propagated current nodes against the existing EMA prototype bank after the trained prototype projection. Class zero and uninitialized prototypes are masked. Produce separate A/B node probability tensors and confidence values.

GT-node graph cross-entropy keeps the GAT aligned with known classes. Accepted U-node pseudo labels optionally supervise graph nodes with a smaller, separately normalized loss. Temporal KL/JS consistency is computed only for paired U nodes that pass validity checks.

### 6. Graph-to-pixel reprojection

For each time, assign a node probability to every pixel covered by an eligible U mask. SAM masks may overlap, so each pixel takes the distribution from the node with the highest graph confidence; ties are resolved by smaller mask area and then stable mask index. Pixels outside accepted masks remain ignore.

This hard mapping keeps SAM boundaries and prevents an average over nested masks from diluting confident predictions.

### 7. Cross-space, cross-time pseudo-label fusion

Use detached teacher classification probabilities when EMA teacher is enabled; otherwise use detached student probabilities. Within the GT-defined unchanged region, accept a pseudo label only when:

1. graph A agrees with classification B;
2. graph B agrees with classification A;
3. graph A and graph B agree;
4. graph confidence and classification confidence exceed their thresholds;
5. the graph node is not GT or ignored.

Accepted class IDs must be in `[1, num_classes)`. Changed pixels keep GT supervision. Unaccepted unchanged pixels remain ignore. The precedence is `GT > accepted graph pseudo > ignore`.

Pseudo targets and acceptance masks are detached. Cross-entropy is then evaluated on the current student A/B pixel logits. This loss is the direct path by which graph pseudo labels update the semantic heads and backbone.

## Losses and Gradient Boundaries

The total loss becomes:

`L = Lsupervised + Lema_pseudo + Lstage1 + Lgraph_gt + Lgraph_temp + Lgraph_node_pseudo + Lgraph_pixel_pseudo`.

Rules:

- Stage1 GT/prototype/edge inputs remain detached from the backbone.
- GAT GT loss, temporal loss, and graph pixel pseudo loss may update current backbone features.
- Memory-bank tensors and generated pseudo targets are always detached.
- GT and pseudo losses are normalized independently.
- Graph pixel pseudo weight uses an epoch ramp and starts smaller than supervised semantic loss.
- Empty node/edge/accepted-pixel cases return connected finite zero losses.

Recommended initial weights:

- graph GT: `0.1`;
- edge GT: `0.1`;
- graph temporal: `0.02`;
- graph node pseudo: `0.02`;
- graph pixel pseudo: `0.05` for SECOND and `0.1` for Landsat;
- graph pseudo ramp: 10 epochs.

## Runtime and CLI

Add an opt-in `--use-sam-global-graph` flag. It requires `--use-sam-graph-stage1` and a valid SAM cache. Additional controls cover:

- global start epoch;
- local/global Top-K;
- memory capacity and confidence threshold;
- graph/classification confidence thresholds;
- temporal, node-pseudo, and pixel-pseudo loss weights;
- pseudo ramp length;
- GAT heads and dropout.

Defaults leave the feature disabled. Landsat and SECOND share the implementation, while recommended loss/confidence values are supplied by command line rather than hardcoded dataset branches.

## Optimizer Sequencing

Propagation modules belong to the graph optimizer. A training iteration performs one combined backward pass. Before stepping:

- unscale both optimizers;
- clip student and graph gradients separately;
- reject the entire update on any non-finite loss or gradient.

After both optimizer steps succeed:

- update EMA teacher;
- enqueue detached accepted anchors into the global memory.

Gradient accumulation must stage pending bank entries and commit them only at the accumulation boundary.

## Checkpoint Compatibility

Upgrade graph runtime state to format version 2 with propagation module, GAT optimizer state, global memory, and counters.

- Version 1 Stage1 checkpoint: restore projector, Stage1, optimizer, and GT queue; initialize GAT/global memory fresh and report this explicitly.
- Version 2 checkpoint: restore all state strictly after dataset, class count, node dimension, and head-count checks.
- Propagation disabled: ignore the additional v2 state without affecting the student path.

## Diagnostics

Log:

- current GT/U/ignored node counts;
- local, temporal, and global edge counts;
- global memory size by class;
- graph confidence and graph/class agreement rate;
- accepted node and pixel pseudo ratios;
- graph GT, temporal, node pseudo, and pixel pseudo losses;
- student and graph gradient norms;
- non-finite update skips.

These values must be available in console output and TensorBoard.

## Error Handling

Reject:

- missing or mismatched SAM cache metadata;
- graph flag combinations that omit Stage1;
- invalid Top-K, capacities, confidence thresholds, or loss weights;
- checkpoint dataset/class/node/head mismatches;
- non-finite node, edge, attention, probability, loss, or gradient tensors.

An empty memory bank, missing class prototype, or a batch without U nodes is valid and produces finite connected zero auxiliary losses.

## Verification

Unit tests must prove:

- global memory is detached, CPU resident, class balanced, FIFO, and resumable;
- global Top-K excludes the current sample and is deterministic;
- each relation produces valid sparse edges without ignored nodes;
- GAT output is finite under CUDA autocast and gradients reach current nodes but not memory nodes;
- overlap reprojection follows confidence/area/index precedence;
- pseudo fusion enforces all agreement and threshold conditions;
- graph pixel pseudo loss creates nonzero backbone gradients;
- empty graphs produce connected finite zero losses;
- Stage1 v1 checkpoints initialize propagation state without corrupting trained Stage1 tensors;
- v2 checkpoints round-trip all new modules and memory;
- propagation-disabled training retains existing behavior.

Integration smoke tests use a tiny synthetic batch and one real cached sample. Completion requires a finite forward/backward/optimizer step with nonzero graph-pseudo backbone gradient and successful resume from an existing Stage1 checkpoint.

## Implementation Boundaries

Keep responsibilities separated:

- `models/graph/global_memory.py`: detached cross-sample memory.
- `models/graph/graph_builder.py`: deterministic sparse candidate edges.
- `models/graph/relation_gat.py`: FP32-safe relation-aware aggregation.
- `models/graph/pseudo_fusion.py`: reprojection and agreement filtering.
- `models/graph/stage2.py`: propagation orchestration and result contract.
- `utils/graph_stage1_training.py`: extend the existing runtime/checkpoint owner without duplicating training orchestration.
- `train.py`: CLI validation, loss composition, metrics, optimizer sequencing, and pending-bank commit only.

No unrelated backbone, validation, dataset-label, or MobileSAM cache changes are part of this implementation.
