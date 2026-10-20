# Landsat historical best graph configuration: weight ablations

Date: 2026-09-30. Requested: three fresh 100-epoch, seed42 experiments.

The historical configuration achieved validation Fscd 83.4612880027% at epoch94.
Its run_config is independently confirmed in the remaining epoch099 checkpoint
Landsat_pseudo_graph_stage2_seed42_epoch099_Fscd0.8317.pth. That file stores the
historical best Fscd, but its IoU/Sek belong to epoch99. The epoch94 same-epoch
metrics are preserved in the 20260919 inventory, configuration group E12.

| Run | GPU | lambda_graph_gt | graph_pixel_pseudo_weight | Only ablation |
| --- | --- | --- | --- | --- |
| Reconstructed historical reference | - | 0.10 | 0.05 | Reference settings, not an additional new run |
| exp14_landsat_best_graphgt02_20260930 | 1 | 0.20 | 0.05 | Graph GT weight |
| exp15_landsat_best_pixel0_20260930 | 2 | 0.10 | 0.00 | Graph pixel weight |
| exp16_landsat_best_pixel1_20260930 | 3 | 0.10 | 1.00 | Graph pixel weight |

All three use lambda_sem_u=0.50, alpha=0.50 to 0.80, EMA=0.999,
pseudo start=30, pseudo ramp=5, min ratio=0.05; full graph Stage1+Stage2,
Stage2 start=55, TopK=8, edge threshold=0.50, graph ramp=10,
lambda_edge_gt=0.10 and graph_temp_weight=0.02.

Unarchived historical settings are explicitly standardized across the three new
runs using exp8's existing Landsat execution settings: input512, batch2,
effective batch8, validation batch2, LR3.5e-4, 8workers, AMP and gradient
checkpointing on, persistent workers off, grad clip1; SAM node area0.001-0.05,
GT threshold0.80, purity0.90, graph/teacher/fallback thresholds0.70/0.90/0.95.
All 67 training source files are copied and hash-verified against exp8. The
SECOND-only numerical patches are not introduced. This is a replay of recorded
hyperparameters under pinned code and execution settings, not a guarantee of
bitwise historical reproduction. Historical Fscd83.4613% is a reference; no new
unchanged reference run is launched by this request. One seed is not a global
optimality claim. Differences between the GT-weight variant and a pixel-weight
variant change two weights; assess each against the reconstructed reference.

lambda_graph_gt multiplies both Stage1 and Stage2 GT node losses. A pixel loss
weight of zero leaves graph GT supervision, edge supervision and temporal loss
enabled; graph prediction/selection diagnostics still run. It is not a no-graph
baseline. The pixel weight follows the common graph ramp starting at epoch55.

Fresh training uses the same pretrained MobileSAM weights and dataset/cache
links, never the historical trained checkpoint. The historical checkpoint is
read on CPU only to audit configuration. Each run retains launch.json,
command.txt, train.pid, source hashes, pretrained hashes, resolved dataset paths,
the baseline command and exactly one variant weight difference.

No automatic checkpoint deletion is started. If validation excludes any sample,
has non-finite outputs/loss, or reports no finite predictions, invalidate the
score and final comparison pending complete reevaluation. Do not silently change
the running configuration. Isolated AMP gradient skips followed by recovery are
recorded; sustained failures are reported. Existing SECOND training on GPU0 is
outside this launcher's mutation scope.
