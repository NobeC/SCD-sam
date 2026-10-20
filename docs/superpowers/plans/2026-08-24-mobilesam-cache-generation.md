# MobileSAM Offline Cache Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate stable MobileSAM A/B masks and quality-deduplicated shared caches offline.

**Architecture:** A shared safe NPZ v2 codec owns packbits and deduplication. Three tools generate, inspect, and audit caches; the existing v1 loader gains backward-compatible v2 reading.

**Tech Stack:** Python 3.8, PyTorch, MobileSAM, NumPy, OpenCV, unittest.

## Global Constraints

- Do not modify `train.py`, main model, validation, or graph objectives.
- Never use GT during cache generation.
- Import complete MobileSAM only in `tools/generate_sam_cache.py`.
- Preserve every valid fine/mid/coarse mask in A/B storage.

### Task 1: Cache v2 codec and deduplication

- [ ] Write failing codec tests.
- [ ] Implement pack/unpack, scale tags, quality-first dedup, atomic save/load.
- [ ] Add v2 compatibility to `utils/sam_mask_cache.py`.
- [ ] Run codec and existing P0 tests.

### Task 2: Offline generator

- [ ] Write failing filename-pair tests.
- [ ] Implement CLI, raw RGB validation, MobileSAM AMG, resume/error logging/statistics.
- [ ] Run CLI help and one-sample real generation.

### Task 3: Inspection and audit

- [ ] Implement deterministic visual overlays and aggregate cache audit.
- [ ] Run inspection/audit on generated pilot cache.
- [ ] Run full unittests and syntax checks.
