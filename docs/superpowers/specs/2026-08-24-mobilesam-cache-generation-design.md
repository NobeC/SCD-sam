# MobileSAM Offline Cache Generation Design

Generate Landsat A/B masks offline with the complete `mobile_sam` package. Keep model generation isolated to `tools/generate_sam_cache.py`; do not modify training, model forward, validation, or graph logic.

Cache v2 uses safe compressed NPZ with packbits masks, aligned SAM metadata, deterministic quality-first A/B union deduplication, fine/mid/coarse scale tags, and shared source/index references. `utils/sam_mask_cache.py` remains backward-compatible with v1 and selects mid masks only when training loads the cache.

Generation reads original RGB uint8 images, validates exact A/B filenames and shapes, writes atomically, skips existing files unless overwrite is requested, records per-image failures, and reports aggregate counts/timing/bytes. Inspection and audit tools never invoke MobileSAM.
