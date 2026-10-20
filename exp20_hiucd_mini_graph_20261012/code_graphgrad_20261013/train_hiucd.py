#!/usr/bin/env python3
"""Matched fully supervised baseline vs. SAM graph Stage-1 experiment."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.hiucd_mini import HiUCDMini

CLASSES = 10


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) and k != "geometry" else v
            for k, v in batch.items()}


def supervised_loss(outputs, batch):
    sem = []
    for logits, target in ((outputs[0].float(), batch["label_A"]),
                           (outputs[1].float(), batch["label_B"])):
        sem.append(F.cross_entropy(logits, target, ignore_index=0) if (target != 0).any()
                   else logits.sum() * 0.0)
    semantic = sum(sem) / 2.0
    logit = outputs[2][:, 0].float()
    valid = batch["change_valid"]
    target = batch["change"].float()
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pos = valid & (target > .5)
    neg = valid & ~pos
    pos_count = pos.sum().clamp_min(1).float()
    neg_count = neg.sum().clamp_min(1).float()
    binary = (.25 * (bce * pos).sum() / pos_count + .75 * (bce * neg).sum() / neg_count)
    consistency_valid = valid & (batch["label_A"] > 0) & (batch["label_B"] > 0)
    prob_a = outputs[0].float()[:, 1:].softmax(1)
    prob_b = outputs[1].float()[:, 1:].softmax(1)
    similarity = F.cosine_similarity(prob_a, prob_b, dim=1).clamp(0, 1)
    consistency_map = torch.where(batch["change"].bool(), similarity, 1.0 - similarity)
    consistency = consistency_map[consistency_valid].mean() if consistency_valid.any() else semantic * 0.0
    return semantic + binary + consistency, {"semantic": semantic, "binary": binary, "consistency": consistency}


class Metrics:
    def __init__(self):
        self.semantic = torch.zeros(CLASSES, CLASSES, dtype=torch.int64)
        self.scd = torch.zeros(CLASSES, CLASSES, dtype=torch.int64)
        self.binary = torch.zeros(2, 2, dtype=torch.int64)

    @torch.no_grad()
    def update(self, outputs, batch):
        pa = outputs[0].argmax(1).cpu()
        pb = outputs[1].argmax(1).cpu()
        change_pred = (outputs[2][:, 0] > 0).long().cpu()
        for pred, target in ((pa, batch["label_A"].cpu()), (pb, batch["label_B"].cpu())):
            valid = target > 0
            self.semantic += torch.bincount((target[valid] * CLASSES + pred[valid]),
                                            minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)
        valid_change = batch["change_valid"].cpu()
        true_change = batch["change"].cpu().long()
        self.binary += torch.bincount((true_change[valid_change] * 2 + change_pred[valid_change]),
                                      minlength=4).reshape(2, 2)
        target_scd = batch["label_B"].cpu() * true_change
        pred_scd = pb * change_pred
        valid_scd = valid_change & (~true_change.bool() | (batch["label_B"].cpu() > 0))
        self.scd += torch.bincount((target_scd[valid_scd] * CLASSES + pred_scd[valid_scd]),
                                   minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)

    def result(self):
        cm = self.semantic.double()
        union = cm.sum(0) + cm.sum(1) - cm.diag()
        iou = [float(cm[i, i] / union[i]) if union[i] else None for i in range(1, CLASSES)]
        scd = self.scd.double()
        f1s = []
        for cls in range(1, CLASSES):
            tp = scd[cls, cls]
            denom = 2 * tp + scd[cls].sum() - tp + scd[:, cls].sum() - tp
            if denom > 0:
                f1s.append(float(2 * tp / denom))
        tn, fp, fn, tp = self.binary.flatten().tolist()
        return {"Fscd": float(np.mean(f1s)) if f1s else None,
                "semantic_mIoU_present": float(np.mean([x for x in iou if x is not None])) if any(x is not None for x in iou) else None,
                "semantic_IoU_classes_1_to_9": iou,
                "change_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "change_IoU": tp / (tp + fp + fn) if tp + fp + fn else None,
                "change_confusion_TN_FP_FN_TP": [tn, fp, fn, tp],
                "semantic_confusion": self.semantic.tolist()}


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    metrics, loss_sum, sample_count = Metrics(), 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", enabled=amp):
            outputs = model(batch["A"], batch["B"])
            loss, _ = supervised_loss(outputs, batch)
        if not torch.isfinite(loss) or not all(torch.isfinite(x).all() for x in outputs):
            raise FloatingPointError("Invalid validation output/loss; score is not valid")
        metrics.update(outputs, batch)
        size = len(batch["name"])
        loss_sum += float(loss) * size
        sample_count += size
    result = metrics.result()
    result["loss"] = loss_sum / max(sample_count, 1)
    return result


def atomic_save(payload, path):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def state_hash(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--verify-graph-gradient", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--lambda-graph-gt", type=float, default=.1)
    parser.add_argument("--lambda-edge-gt", type=float, default=.1)
    parser.add_argument("--sam-tau-gt", type=float, default=.8)
    parser.add_argument("--sam-tau-purity", type=float, default=.9)
    parser.add_argument("--sam-tau-u", type=float, default=.95)
    parser.add_argument("--sam-min-area", type=float, default=.001)
    parser.add_argument("--sam-max-area", type=float, default=.05)
    parser.add_argument("--graph-node-dim", type=int, default=128)
    parser.add_argument("--graph-max-regions", type=int, default=128)
    opt = parser.parse_args()
    if opt.verify_graph_gradient and (not opt.graph or not opt.smoke_steps):
        parser.error("Gradient verification requires --graph and --smoke-steps")
    if opt.graph and not opt.cache_root:
        parser.error("--graph requires an offline --cache-root")
    if opt.effective_batch_size % opt.batch_size or opt.epochs < 1:
        parser.error("Invalid epoch or gradient-accumulation settings")
    output = Path(opt.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing non-empty output directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    split_path = Path(opt.split_json)
    manifest = json.loads(split_path.read_text())
    seed_all(opt.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    train = HiUCDMini(opt.root, split_path, "train", augment=not bool(opt.smoke_steps))
    val = HiUCDMini(opt.root, split_path, "val", augment=False)
    loader_gen = torch.Generator().manual_seed(opt.seed)
    loader = DataLoader(train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                        pin_memory=True, worker_init_fn=seed_worker, generator=loader_gen,
                        persistent_workers=False)
    validation = DataLoader(val, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True, persistent_workers=False)
    from main_model import ChangeDetection
    from types import SimpleNamespace
    model = ChangeDetection(SimpleNamespace(backbone="msam_96", neck="fpn+drop", head="fcn",
        pretrain="", num_classes=CLASSES, input_size=512, gradient_checkpointing=True)).to(device)
    init_hash = state_hash(model)
    graph = None
    if opt.graph:
        from utils.graph_stage1_training import GraphStage1TrainingRuntime
        from utils.sam_mask_cache import SamMaskCacheConfig, load_shared_sam_masks
        from utils.sam_node_audit import SamNodeAuditConfig
        cache_report = json.loads((Path(opt.cache_root) / "HiUCDMini" / "cache_manifest.json").read_text())
        if not cache_report.get("complete") or cache_report.get("split_sha256") != manifest["split_sha256"]:
            raise RuntimeError("Complete offline SAM cache for this exact split is required")
        train_ids = {item["sample_id"] for item in manifest["patches"]["train"]}
        if set(cache_report["entries"]) != train_ids:
            raise RuntimeError("SAM cache entries do not exactly match training patches")
        graph = GraphStage1TrainingRuntime(model.backbone, CLASSES, node_dim=opt.graph_node_dim,
            audit_config=SamNodeAuditConfig(tau_gt=opt.sam_tau_gt, tau_purity=opt.sam_tau_purity,
                tau_u=opt.sam_tau_u, min_mask_pixels=4, min_changed_pixels=4), use_stage2=False)
        graph.cache_loader = (load_shared_sam_masks, SamMaskCacheConfig(
            min_area_ratio=opt.sam_min_area, max_area_ratio=opt.sam_max_area))
        graph.cache_root = Path(opt.cache_root)
        graph.max_regions = opt.graph_max_regions
    optimizer = torch.optim.AdamW([
        {"params": [v for k, v in model.named_parameters() if v.requires_grad and "backbone" in k], "lr": opt.lr / 10},
        {"params": [v for k, v in model.named_parameters() if v.requires_grad and "backbone" not in k], "lr": opt.lr}],
        weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    seed_all(opt.seed)
    scaler = torch.cuda.amp.GradScaler(enabled=not opt.no_amp, init_scale=128.0)
    config = vars(opt).copy()
    config.update(num_classes=CLASSES, split_sha256=manifest["split_sha256"], initial_model_sha256=init_hash,
        train_patches=len(train), val_patches=len(val), dataset_protocol=manifest["protocol"],
        graph_policy="SAM Stage-1 GT prototype and edge supervision with backbone gradients; no EMA/pseudo-label Stage-2")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("CONFIG", json.dumps(config, sort_keys=True), flush=True)
    best, step, started = -1.0, 0, time.time()
    nonfinite_skips = 0
    accum = 1 if opt.smoke_steps else opt.effective_batch_size // opt.batch_size
    for epoch in range(opt.epochs):
        model.train()
        if graph:
            graph.train()
            graph.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(loader):
            batch = move(raw_batch, device)
            graph_data = None
            if graph:
                load_masks, cache_cfg = graph.cache_loader
                graph_data = []
                for name, geometry in zip(batch["name"], raw_batch["geometry"]):
                    masks = load_masks(graph.cache_root, "HiUCDMini", "train", name, (512, 512), cache_cfg)
                    k, horizontal, vertical = [int(v) for v in geometry]
                    masks = torch.rot90(masks, k, (1, 2))
                    if horizontal:
                        masks = masks.flip(2)
                    if vertical:
                        masks = masks.flip(1)
                    graph_data.append({"shared_masks": masks[:graph.max_regions].contiguous()})
            with torch.autocast("cuda", enabled=not opt.no_amp):
                if graph:
                    outputs, features = model(batch["A"], batch["B"], return_graph_features=True,
                                              detach_graph_features=False)
                else:
                    outputs = model(batch["A"], batch["B"])
                    features = None
                loss, parts = supervised_loss(outputs, batch)
            if graph:
                with torch.random.fork_rng(devices=[device.index or 0]):
                    torch.manual_seed(opt.seed + step)
                    from utils.sam_node_audit import audit_sam_nodes
                    masks = [item["shared_masks"] for item in graph_data]
                    audits = [audit_sam_nodes(mask, batch["label_A"][j].detach().cpu(),
                        batch["label_B"][j].detach().cpu(), CLASSES, graph.audit_config)
                        for j, mask in enumerate(masks)]
                    # The shared Stage-1 runtime detaches features; retain them in this isolated experiment.
                    nodes = graph.projector(*(x.float() for x in features), masks)
                    graph_result = graph.stage1(nodes, audits)
                graph_loss = opt.lambda_graph_gt * graph_result.loss_graph_gt + opt.lambda_edge_gt * graph_result.loss_edge_gt
                if opt.verify_graph_gradient and step == 0:
                    graph_loss.backward(retain_graph=True)
                    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
                    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                        raise AssertionError("Graph-only backbone gradients are missing or nonfinite")
                    norm = sum(float(g.float().square().sum()) for g in gradients) ** .5
                    if norm <= 0:
                        raise AssertionError("Graph-only backbone gradient norm is zero")
                    print("GRAPH_GRADIENT_VERIFIED", json.dumps({"backbone_grad_norm": norm,
                        "parameter_tensors": len(gradients)}), flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    graph.zero_grad()
                loss = loss + graph_loss
            if not torch.isfinite(loss) or not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError("Nonfinite training loss/output; stopping without invalid checkpoint")
            scaler.scale(loss / accum).backward()
            if (index + 1) % accum == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                model_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                finite_norms = bool(torch.isfinite(model_norm))
                if graph:
                    scaler.unscale_(graph.optimizer)
                    graph_norm = torch.nn.utils.clip_grad_norm_(list(graph.parameters()), 1.0)
                    finite_norms = finite_norms and bool(torch.isfinite(graph_norm))
                if finite_norms:
                    scaler.step(optimizer)
                    if graph:
                        scaler.step(graph.optimizer)
                else:
                    nonfinite_skips += 1
                    if scaler.is_enabled():
                        scaler.update(new_scale=max(scaler.get_scale() * .5, 1.0))
                if graph:
                    graph.zero_grad()
                if finite_norms:
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if index == 0 or (index + 1) % 50 == 0:
                record = {"epoch": epoch, "step": step, "batch": index + 1,
                          "loss": float(loss.detach()), "grad_norm": float(model_norm) if (index + 1) % accum == 0 else None,
                          **{k: float(v.detach()) for k, v in parts.items()}}
                if graph:
                    record.update(graph_loss=float(graph_loss.detach()), graph_gt=float(graph_result.loss_graph_gt.detach()),
                                  graph_edge=float(graph_result.loss_edge_gt.detach()), graph_gt_nodes=graph_result.gt_node_count)
                record["nonfinite_gradient_skips"] = nonfinite_skips
                print("TRAIN", json.dumps(record), flush=True)
            if opt.smoke_steps and step >= opt.smoke_steps:
                print("SMOKE_COMPLETE", flush=True)
                return
        metrics = evaluate(model, validation, device, not opt.no_amp)
        print("VAL", epoch, json.dumps(metrics), flush=True)
        score = metrics["Fscd"] if metrics["Fscd"] is not None else -1.0
        best = max(best, score)
        scheduler.step()
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "epoch": epoch + 1, "step": step, "best": best, "config": config,
                   "graph": graph.state_dict() if graph else None}
        atomic_save(payload, output / ("epoch%03d.pth" % epoch))
        atomic_save(payload, output / "last.pth")
        if score >= best:
            atomic_save(payload, output / "best.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
    print("TRAINING_COMPLETE", json.dumps({"best_Fscd": best, "elapsed_s": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()

