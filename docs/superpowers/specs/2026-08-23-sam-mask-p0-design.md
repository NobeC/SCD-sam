# SAM Mask P0 Data Pipeline Design

## Scope

P0 establishes the data and compatibility boundary required by the future graph branch. It does not generate SAM masks, add graph neural networks, or modify `ChangeDetection`.

The implementation must preserve the current Landsat default and SECOND dataset switch. Existing dataset callers continue receiving the original four tensors unless SAM graph data is explicitly enabled.

## Cache contract

Each source image pair has one cache file:

```text
sam_mask_cache/<Dataset>/<split>/<sample_id>.npz
```

Required arrays:

- `masks_A`: `[N_A, H, W]`, boolean or `uint8` binary masks.
- `masks_B`: `[N_B, H, W]`, boolean or `uint8` binary masks.
- `image_shape`: `[H, W]`.
- `format_version`: scalar integer, initially `1`.

Optional metadata may include a generator name and generator configuration string. Loading must not execute pickled Python objects: `numpy.load(..., allow_pickle=False)` is required.

Missing files, unsupported versions, non-binary masks, rank mismatches, and image-shape mismatches raise an error containing the sample ID and cache path. The graph-data path must never silently fall back to empty masks.

## Shared-mask preparation

The loader concatenates A and B masks, removes empty or undersized masks, and deduplicates them deterministically. Two masks are duplicates when either their IoU exceeds the configured IoU threshold or their containment ratio exceeds the configured containment threshold. When duplicates compete, the smaller mask is retained for containment duplicates and the first stable item is retained for equivalent-area masks.

P0 then filters by configurable area ratio. The default MVP range is mid-scale masks with area ratio from `0.01` through `0.05`. Filtering and deduplication operate at source resolution before feature mapping.

The output is a variable-length boolean tensor `[N, H, W]`. A valid cache is allowed to produce zero masks after filtering; this is an observable empty sample, not a cache failure.

## Dataset API and backward compatibility

`datasets.RS_ST.Data` gains optional graph-data settings. With graph data disabled, `__getitem__` returns exactly:

```text
(image_A, image_B, label_A, label_B)
```

With graph data enabled, it returns:

```text
(image_A, image_B, label_A, label_B, graph_data)
```

where `graph_data` contains the stable `sample_id` and `shared_masks`. The sample ID is the image filename stem. Validation and inference keep graph data disabled.

Because samples contain different mask counts, P0 provides a graph-aware collate function. It stacks the four dense tensors but preserves graph data as a per-sample list. Existing `DataLoader` behavior is unchanged when graph data is disabled.

## Synchronized augmentation

Rotation and flip decisions are sampled once and applied identically to image A, image B, both labels, and every shared mask. Masks use discrete NumPy rotation/flip operations and remain boolean; no bilinear interpolation is permitted.

The augmented masks must retain the same spatial shape as the augmented labels. Empty masks created by future crop support are removed after transformation. P0 supports the existing 90-degree rotation and flip path only; arbitrary crop support is deferred until the training dataset actually enables cropping.

## GT, unlabeled, and ignore audit

P0 provides a pure function that classifies each shared mask independently for time A and time B using:

- `changed = (label_A != 0) | (label_B != 0)`;
- configurable `tau_gt`, `tau_purity`, `tau_u`;
- configurable `min_mask_pixels` and `min_changed_pixels`.

Each temporal node receives one state: GT, unlabeled, or ignore. GT labels must be in `1..num_classes-1`; class 0 is never a GT semantic node. The function returns per-node states, class IDs, change ratios, purity, and changed-pixel counts.

An aggregation utility reports per sample and per class:

- total shared masks;
- GT/U/ignore temporal-node counts;
- GT nodes per semantic class;
- images with no GT nodes;
- potential positive and negative GT pair counts;
- changed-pixel coverage by any shared mask and by reliable GT masks.

Statistics must handle empty masks and absent classes without NaN.

## Graph and checkpoint isolation boundary

P0 introduces configuration/data types only. It does not register graph modules inside `ChangeDetection`, alter its forward result, or add graph parameters to the existing optimizer.

Future graph training state will live under optional top-level checkpoint fields such as `graph` and `graph_optimizer`. The current format remains loadable when those fields are absent. P0 documents and tests this boundary but does not implement EdgeNet, GAT, prototypes, or graph checkpoint payloads.

## Error handling and numerical rules

- Cache validation occurs before augmentation.
- All ratios use a denominator clamped to at least one.
- Empty input returns correctly shaped empty tensors and zero integer statistics.
- Masks are boolean on CPU in the dataset layer.
- Dataset code does not move masks to CUDA.
- Unknown label IDs outside `[0, num_classes-1]` raise an error in the audit function.

## Testing strategy

Tests cover cache validation, deterministic union/deduplication, area filtering, synchronized augmentation, legacy four-item dataset behavior, graph-aware five-item behavior and collate, GT/U/ignore assignment, empty sets, class/pair statistics, Landsat/SECOND dynamic class counts, and unchanged `ChangeDetection`/checkpoint boundaries.

## Deferred work

SAM mask generation, exact MobileSAM block features, region pooling, projectors, memory banks, EdgeNet, GAT, prototypes, pixel projection, teacher veto, multi-scale nodes, and graph optimizer/checkpoint persistence are P1 or later.
