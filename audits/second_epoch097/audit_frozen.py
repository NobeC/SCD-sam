"""Frozen SECOND pseudo-label audit; writes only audit reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SCD_DATASET"] = "second"
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from main_model import ChangeDetection
from datasets import RS_ST as RS
from models.graph.region_features import SamGraphFeatureProvider
from models.graph.node_projector import TemporalNodeProjector
from models.graph.stage1 import SamGraphStage1
from models.graph.stage2 import SamGraphStage2
from models.graph.pseudo_fusion import reproject_stage2_batch, teacher_graph_veto
from utils.sam_mask_cache import load_shared_sam_masks, SamMaskCacheConfig
from utils.sam_node_audit import audit_sam_nodes, SamNodeAuditConfig
from utils.pseudo_label import generate_entropy_pseudo_label
import utils.sam_mask_cache as mask_cache_module
from fast_masks import prepare_shared_masks

mask_cache_module.prepare_shared_masks = prepare_shared_masks


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Stats:
    def __init__(self):
        self.total = 0
        self.gt_total = 0
        self.groups = {}
        self.nodes = {}

    def add(self, name, mask, pred, gt):
        item = self.groups.setdefault(name, {"pixels": 0, "predicted_classes": np.zeros(7, np.int64),
                                             "confusion": np.zeros((7, 7), np.int64)})
        p = pred[mask].cpu().numpy().astype(np.int64)
        item["pixels"] += len(p)
        item["predicted_classes"] += np.bincount(p, minlength=7)
        valid = mask & (gt > 0) & (gt < 7)
        codes = (gt[valid] * 7 + pred[valid]).cpu().numpy().astype(np.int64)
        item["confusion"] += np.bincount(codes, minlength=49).reshape(7, 7)

    def report(self):
        result = {"pixel_time_denominator": self.total, "valid_gt_pixel_times": self.gt_total,
                  "node_counts": self.nodes, "groups": {}}
        for name, item in self.groups.items():
            cm = item["confusion"]
            n = int(cm.sum())
            result["groups"][name] = {
                "pixels": item["pixels"], "coverage": item["pixels"] / max(self.total, 1),
                "predicted_classes": item["predicted_classes"].tolist(),
                "valid_gt_pixels": n, "valid_gt_coverage": n / max(self.gt_total, 1),
                "accuracy": float(np.trace(cm) / n) if n else None,
                "confusion_gt_rows_pred_columns": cm.tolist(),
                "class_accuracy": [float(cm[k, k] / cm[k].sum()) if cm[k].sum() else None
                                   for k in range(7)]}
        return result


def read_image(path, side):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(RS.normalize_image(rgb, side).transpose(2, 0, 1)).float()


def read_label(path):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    label = np.full(rgb.shape[:2], -1, np.int64)
    for k, color in enumerate(RS.ST_COLORMAP):
        label[np.all(rgb == color, axis=2)] = k
    if (label < 0).any():
        raise ValueError("Unknown label color: " + str(path))
    return torch.from_numpy(label)


def compare(stats, teacher_mask, teacher_pred, graph_mask, graph_pred, gt, prefix):
    masks = {"agree": teacher_mask & graph_mask & (teacher_pred == graph_pred),
             "conflict": teacher_mask & graph_mask & (teacher_pred != graph_pred),
             "graph_only": graph_mask & ~teacher_mask,
             "teacher_only": teacher_mask & ~graph_mask}
    for category, mask in masks.items():
        stats.add(prefix + "/" + category + "/teacher", mask, teacher_pred, gt)
        stats.add(prefix + "/" + category + "/graph", mask, graph_pred, gt)
    assert sum(int(m.sum()) for m in masks.values()) == int((teacher_mask | graph_mask).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "report.json")
    args = parser.parse_args()
    print("Loading frozen checkpoint", flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = ROOT / "checkpoints/SECOND/SECOND_pseudo_graph_stage2_seed42_epoch097_Fscd0.6112.pth"
    initial_stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    c = torch.load(checkpoint, map_location="cpu")
    cfg, gc = c["run_config"], c["sam_graph_stage1"]["config"]
    assert cfg["dataset"] == "SECOND" and cfg["num_classes"] == 7
    opt = SimpleNamespace(backbone="msam_96", neck="fpn+drop", head="fcn",
                          pretrain="", input_size=512, num_classes=7)
    teacher = ChangeDetection(opt).to(args.device).eval()
    teacher.load_state_dict(c["teacher"], strict=True)
    # Only the student encoder supplies graph features; avoid a second full model.
    from models.Encoder.mobilesam import build_sam_vit_t
    backbone = build_sam_vit_t().to(args.device).eval()
    encoder_state = {k[len("backbone."):]: v for k, v in c["student"].items()
                     if k.startswith("backbone.")}
    backbone.load_state_dict(encoder_state, strict=True)
    provider = SamGraphFeatureProvider(backbone).eval()
    projector = TemporalNodeProjector(gc["stage_channels"], gc["neck_channels"], gc["node_dim"]).to(args.device).eval()
    stage1 = SamGraphStage1(7, gc["node_dim"], queue_capacity=gc["queue_capacity"]).eval()
    stage2 = SamGraphStage2(7, gc["node_dim"], heads=gc["graph_gat_heads"],
                           dropout=gc["graph_gat_dropout"], local_topk=gc["graph_local_topk"],
                           edge_threshold=gc["graph_edge_threshold"],
                           edge_bias_beta=gc["graph_edge_bias_beta"]).eval()
    for key, module in (("projector", projector), ("stage1", stage1), ("stage2", stage2)):
        module.load_state_dict(c["sam_graph_stage1"][key], strict=True)
    modules = (teacher, backbone, projector, stage1, stage2)
    for module in modules:
        module.requires_grad_(False)
    versions = [(t, t._version) for module in modules for t in module.state_dict().values()]
    frozen_graph = {k: v.clone() for k, v in stage1.state_dict().items()}
    gamma = ((0.03 + (0.1 - 0.03) * (97 - 40) / (100 - 40)) * c["class_precision"]).clamp(0, 0.1).to(args.device)
    gamma[0] = 0
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "filename_epoch": 97, "resume_epoch": c["epoch"], "run_config": cfg,
                "graph_config": gc, "gamma": gamma.tolist(), "batch_size": args.batch_size,
                "assumptions_not_in_checkpoint": {"epochs": 100, "teacher_threshold": 0.9,
                    "fallback_threshold": 0.95, "graph_confidence_threshold": 0.7,
                    "mask_area_ratio": [0.001, 0.05], "node_audit": vars(SamNodeAuditConfig())},
                "classes": RS.ST_CLASSES, "modes": {
                    "native": "GT used ONLY for training-style node eligibility and pseudo-region gating; no prototype updates",
                    "hidden_gt": "All input labels zero; true labels used only for scoring; changed-region results are diagnostic, not training yield"}}
    metadata["semantic_probe"] = "No change or temporal gating; separate classifier-quality diagnostic, not training pseudo-label yield"
    metadata["execution"] = {"network_device": args.device, "graph_device": "cpu",
                             "torch_version": torch.__version__, "precision": "float32",
                             "mask_deduplication": "exact packed-bit equivalent; reference output verified on three samples"}
    sources = [Path("main_model.py"), Path("datasets/RS_ST.py"), Path("models/head/FCN.py")]
    sources += list(Path("models/graph").glob("*.py"))
    sources += [Path("utils/pseudo_label.py"), Path("utils/sam_node_audit.py"), Path("utils/sam_mask_cache.py")]
    metadata["source_sha256"] = {str(p): digest(ROOT / p) for p in sources}
    metadata["audit_source_sha256"] = {p.name: digest(p) for p in
                                      (Path(__file__), Path(__file__).with_name("fast_masks.py"))}
    files = sorted((ROOT / "datasets/SECOND/val/im1").glob("*.png"))
    for folder in ("im2", "label1", "label2"):
        assert {p.name for p in files} == {p.name for p in (ROOT / "datasets/SECOND/val" / folder).glob("*.png")}
    if args.limit:
        files = files[:args.limit]
    metadata["samples"] = [p.stem for p in files]
    results = {mode: Stats() for mode in ("native", "hidden_gt")}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(files), args.batch_size):
            batch_started = time.monotonic()
            batch_files = files[start:start + args.batch_size]
            imgs = [torch.stack([read_image(ROOT / "datasets/SECOND/val" / folder / p.name, side)
                                for p in batch_files]).to(args.device)
                    for folder, side in (("im1", "A"), ("im2", "B"))]
            labels = [torch.stack([read_label(ROOT / "datasets/SECOND/val" / folder / p.name)
                                  for p in batch_files]).to(args.device)
                      for folder in ("label1", "label2")]
            masks = [load_shared_sam_masks(ROOT / "sam_mask_cache", "SECOND", "val", p.stem,
                                           labels[0].shape[-2:], SamMaskCacheConfig(min_area_ratio=0.001, max_area_ratio=0.05))
                     for p in batch_files]
            masks_done = time.monotonic()
            outs = teacher(*imgs)
            assert all(torch.isfinite(t).all() for t in outs)
            probs = [torch.softmax(o.float(), 1) for o in outs[:2]]
            conf, pred = zip(*(p.max(1) for p in probs))
            unchanged = outs[2].float().sigmoid().squeeze(1) < cfg["pseudo_change_thr"]
            nodes = projector(*provider(*imgs), masks)
            nodes = [replace(n, nodes_a=n.nodes_a.cpu(), nodes_b=n.nodes_b.cpu(),
                             geometry=n.geometry.cpu(), valid=n.valid.cpu()) for n in nodes]
            features_done = time.monotonic()
            for mode, stats in results.items():
                inputs = labels if mode == "native" else [torch.zeros_like(l) for l in labels]
                audits = [audit_sam_nodes(m, inputs[0][i].cpu(), inputs[1][i].cpu(), 7)
                          for i, m in enumerate(masks)]
                # Calling Stage1.forward would update prototypes and queue even in eval mode.
                graph = stage2(nodes, audits, stage1)
                pixels = reproject_stage2_batch(graph, audits, masks)
                pixels = replace(pixels, **{k: v.to(args.device) for k, v in vars(pixels).items()})
                for key in ("node_count", "valid_node_count", "edge_count", "graph_gt_count", "graph_gt_correct"):
                    stats.nodes[key] = stats.nodes.get(key, 0) + getattr(graph, key)
                gate = (inputs[0] == 0) & (inputs[1] == 0)
                candidate = gate & unchanged
                veto = teacher_graph_veto(probs[0], probs[1], outs[2], *inputs,
                    pixels.pred_a, pixels.conf_a, pixels.valid_a,
                    pixels.pred_b, pixels.conf_b, pixels.valid_b,
                    0.9, 0.95, 0.7, cfg["pseudo_change_thr"], 7)
                effective = [v & (q >= 0.7) & (p > 0) for p, q, v in
                             ((pixels.pred_a, pixels.conf_a, pixels.valid_a),
                              (pixels.pred_b, pixels.conf_b, pixels.valid_b))]
                tc = candidate & (pred[0] > 0) & (pred[0] == pred[1]) & (conf[0] >= 0.9) & (conf[1] >= 0.9)
                has_graph = effective[0] | effective[1]
                conflict = (effective[0] & (pixels.pred_a != pred[0])) | (effective[1] & (pixels.pred_b != pred[1]))
                fallback = tc & ~has_graph & (conf[0] >= 0.95) & (conf[1] >= 0.95)
                assert int(fallback.sum()) == veto.fallback_count
                assert torch.equal(veto.valid, (tc & has_graph & ~conflict) | fallback)
                for i, (gp, gv) in enumerate(((pixels.pred_a, pixels.valid_a), (pixels.pred_b, pixels.valid_b))):
                    gt = labels[i]
                    stats.total += gt.numel()
                    stats.gt_total += int((gt > 0).sum())
                    target, entropy_mask, _ = generate_entropy_pseudo_label(outs[i], candidate, 7, gamma)
                    stats.add("teacher_entropy", entropy_mask, pred[i], gt)
                    stats.add("teacher_confidence_pair", tc, pred[i], gt)
                    stats.add("graph_raw", gv, gp, gt)
                    stats.add("graph_effective", effective[i], gp, gt)
                    if mode == "hidden_gt":
                        semantic_conf = (pred[i] > 0) & (conf[i] >= 0.9)
                        _, semantic_entropy, _ = generate_entropy_pseudo_label(
                            outs[i], torch.ones_like(candidate), 7, gamma)
                        stats.add("probe_teacher_confidence", semantic_conf, pred[i], gt)
                        stats.add("probe_teacher_entropy", semantic_entropy, pred[i], gt)
                        compare(stats, semantic_conf, pred[i], effective[i], gp, gt,
                                "probe_confidence_vs_graph")
                        compare(stats, semantic_entropy, pred[i], effective[i], gp, gt,
                                "probe_entropy_vs_graph")
                    compare(stats, entropy_mask, pred[i], effective[i], gp, gt, "entropy_vs_graph")
                    compare(stats, tc, pred[i], effective[i], gp, gt, "confidence_vs_graph")
                    for name, mask in (("fallback", fallback), ("veto_conflict", tc & conflict),
                                       ("accepted_with_graph", tc & has_graph & ~conflict), ("accepted_total", veto.valid)):
                        stats.add(name, mask, pred[i], gt)
            print("AUDIT {}/{} {:.1f}s; masks {:.1f}s features {:.1f}s graph/stat {:.1f}s".format(
                min(start + args.batch_size, len(files)), len(files), time.monotonic() - started,
                masks_done - batch_started, features_done - masks_done,
                time.monotonic() - features_done), flush=True)
    assert all(t._version == version for t, version in versions), "Frozen state mutated"
    assert all(torch.equal(v, frozen_graph[k]) for k, v in stage1.state_dict().items())
    assert initial_stat == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    metadata["frozen_state_verified"] = True
    metadata["elapsed_seconds"] = time.monotonic() - started
    report = {"metadata": metadata, "results": {m: s.report() for m, s in results.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print("REPORT " + str(args.output), flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
