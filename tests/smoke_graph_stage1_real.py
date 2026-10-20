"""One-step synthetic smoke test for the real MobileSAM graph Stage1 path."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.Encoder.mobilesam import build_sam_vit_t
from utils.graph_stage1_training import GraphStage1TrainingRuntime
from utils.sam_node_audit import SamNodeAuditConfig


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real MobileSAM smoke test")
    device = torch.device("cuda:0")
    backbone = build_sam_vit_t(use_checkpoint=False).to(device).eval()
    runtime = GraphStage1TrainingRuntime(
        backbone,
        num_classes=3,
        audit_config=SamNodeAuditConfig(
            tau_gt=0.8,
            tau_purity=0.9,
            tau_u=0.95,
            min_mask_pixels=4,
            min_changed_pixels=4,
        ),
        max_pairs=16,
    )
    images_a = torch.randn(1, 3, 512, 512, device=device)
    images_b = torch.randn(1, 3, 512, 512, device=device)
    labels_a = torch.ones(1, 512, 512, dtype=torch.long, device=device)
    labels_b = torch.full((1, 512, 512), 2, dtype=torch.long, device=device)
    masks = torch.zeros(2, 512, 512, dtype=torch.bool)
    masks[0, :256] = True
    masks[1, 256:] = True
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = runtime.compute(
            images_a,
            images_b,
            labels_a,
            labels_b,
            [{"sample_id": "synthetic", "shared_masks": masks}],
        )
    loss = result.loss_graph_gt + result.loss_edge_gt
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite graph loss")
    loss.backward()
    runtime.optimizer.step()
    print(
        "real_mobile_sam_graph_amp_smoke=ok loss={:.6f} gt_nodes={} positive={} negative={}".format(
            float(loss.detach()),
            result.gt_node_count,
            result.positive_pair_count,
            result.negative_pair_count,
        )
    )


if __name__ == "__main__":
    main()
