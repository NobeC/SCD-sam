# HRSCD Split and Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, leakage-free HRSCD dataset with 117/29/145 original-image pairs and aligned 512x512 PNG patches.

**Architecture:** Read the five official ZIP archives directly, pair their sorted TIFF members using the same positional convention as TorchRS, assign original pairs to fixed seed-42 splits, and tile each split independently. Persist both original-pair and patch-level manifests so every output can be traced back to one source pair and crop coordinate.

**Tech Stack:** Python 3.8, standard library (`zipfile`, `csv`, `random`, `pathlib`), NumPy, OpenCV, pytest.

## Global Constraints

- Split original image pairs before tiling.
- Use exactly train=117, val=29, test=145 and random seed 42.
- Use 512x512 crops and include a final edge-anchored crop on dimensions not divisible by 512.
- Output `A`, `B`, `label_A`, `label_B`, and `change` with identical zero-padded PNG names.
- Never use test data for training statistics, prototypes, thresholds, or pseudo-labels.
- Preserve the five downloaded archives unchanged.

---

### Task 1: Deterministic split and tiling primitives

**Files:**
- Create: `tools/prepare_hrscd.py`
- Create: `tests/test_prepare_hrscd.py`

**Interfaces:**
- Produces: `make_split_indices(total, train, val, test, seed)` and `tile_starts(length, tile_size)`.

- [ ] Write failing tests for deterministic counts, disjoint original indices, and complete edge coverage.
- [ ] Run `pytest -q tests/test_prepare_hrscd.py` and verify failure because the module is missing.
- [ ] Implement the minimal pure functions.
- [ ] Run `pytest -q tests/test_prepare_hrscd.py` and verify all tests pass.

### Task 2: Archive pairing, validation, and conversion

**Files:**
- Modify: `tools/prepare_hrscd.py`
- Modify: `tests/test_prepare_hrscd.py`

**Interfaces:**
- Consumes: the five ZIP paths and deterministic split/tile primitives.
- Produces: `prepare_dataset(raw_root, output_root, seed=42, tile_size=512)`.

- [ ] Add failing tests for positional archive pairing, mismatched archive counts, and aligned output names.
- [ ] Run the targeted tests and verify expected failures.
- [ ] Implement ZIP decoding, shape/range validation, PNG writing, and CSV/JSON manifests.
- [ ] Run all tests and syntax compilation.

### Task 3: Generate and audit HRSCD

**Files:**
- Generate: `datasets/HRSCD/{train,val,test}/{A,B,label_A,label_B,change}`
- Generate: `datasets/HRSCD/original_pairs.csv`
- Generate: `datasets/HRSCD/patches.csv`
- Generate: `datasets/HRSCD/dataset_info.json`

**Interfaces:**
- Consumes: `datasets/HRSCD_raw/HRSCD/*.zip`.
- Produces: the immutable split manifests and aligned 512x512 dataset.

- [ ] Run the converter with seed 42.
- [ ] Verify original counts 117/29/145 and zero overlap.
- [ ] Verify all five modalities have identical patch counts and filenames per split.
- [ ] Decode representative outputs and verify 512x512 dimensions and valid label ranges.
- [ ] Report output size, patch counts, and manifest paths.

