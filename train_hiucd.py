#!/usr/bin/env python3
"""Hi-UCD mini: fully supervised + EMA Teacher pseudo labels + SAM graph complement."""
import argparse
import copy
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
PSEUDO_IGNORE = 255


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


@torch.no_grad()
def update_ema_teacher(student, teacher, decay):
    for (k_s, v_s), (k_t, v_t) in zip(
            student.state_dict().items(), teacher.state_dict().items()):
        if v_t.dtype.is_floating_point:
            v_t.mul_(decay).add_(v_s.detach(), alpha=1.0 - decay)
        else:
            v_t.copy_(v_s)


def pseudo_ramp_weight(epoch, start_epoch, ramp_epochs, max_weight):
    if epoch < start_epoch:
        return 0.0
    progress = min(1.0, (epoch - start_epoch) / max(ramp_epochs, 1))
    return max_weight * progress


@torch.no_grad()
def entropy_pseudo_labels(probs, candidate_mask, num_classes, keep_ratio=0.7):
    """Simple entropy filtering: per-class keep lowest-entropy pixels."""
    B, C, H, W = probs.shape
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
    pred = probs.argmax(dim=1)
    pseudo = torch.full((B, H, W), PSEUDO_IGNORE, device=probs.device, dtype=torch.long)
    for cls in range(1, num_classes):
        mask = (pred == cls) & candidate_mask
        if not mask.any():
            continue
        ent = entropy[mask]
        k = max(1, int(mask.sum().item() * keep_ratio))
        thr = torch.topk(ent, k=min(k, ent.numel()), largest=False).values.max()
        pseudo[mask & (entropy <= thr)] = cls
    return pseudo


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
            raise FloatingPointError("Invalid validation output/loss")
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
    # Dataset
    parser.add_argument("--root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    # EMA Teacher
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--no-ema-teacher", action="store_true")
    # Pseudo labels
    parser.add_argument("--pseudo-start-epoch", type=int, default=10)
    parser.add_argument("--pseudo-rampup-epochs", type=int, default=5)
    parser.add_argument("--lambda-sem-u", type=float, default=0.1)
    parser.add_argument("--pseudo-change-thr", type=float, default=0.3)
    parser.add_argument("--pseudo-keep-ratio", type=float, default=0.7)
    # Graph
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--lambda-graph-gt", type=float, default=.1)
    parser.add_argument("--lambda-edge-gt", type=float, default=.1)
    parser.add_argument("--sam-tau-gt", type=float, default=.8)
    parser.add_argument("--sam-tau-purity", type=float, default=.9)
    parser.add_argument("--sam-tau-u", type=float, default=.95)
    parser.add_argument("--sam-min-area", type=float, default=.001)
    parser.add_argument("--sam-max-area", type=float, default=.05)
    parser.add_argument("--graph-node-dim", type=int, default=128)
    parser.add_argument("--graph-max-regions", type=int, default=128)
    parser.add_argument("--graph-local-topk", type=int, default=8)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.5)
    parser.add_argument("--graph-pixel-pseudo-weight", type=float, default=0.05)
    parser.add_argument("--graph-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--graph-temp-weight", type=float, default=0.02)

    opt = parser.parse_args()
    if opt.graph and not opt.cache_root:
        parser.error("--graph requires --cache-root")
    if opt.effective_batch_size % opt.batch_size or opt.epochs < 1:
        parser.error("Invalid epoch or gradient-accumulation settings")

    use_ema = not opt.no_ema_teacher

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

    train_ds = HiUCDMini(opt.root, split_path, "train", augment=not bool(opt.smoke_steps))
    val_ds = HiUCDMini(opt.root, split_path, "val", augment=False)
    loader_gen = torch.Generator().manual_seed(opt.seed)
    loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.workers, pin_memory=True,
                        worker_init_fn=seed_worker, generator=loader_gen,
                        persistent_workers=opt.workers > 0)
    validation = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.workers, pin_memory=True,
                            persistent_workers=opt.workers > 0)

    from main_model import ChangeDetection
    from types import SimpleNamespace
    model = ChangeDetection(SimpleNamespace(
        backbone="msam_96", neck="fpn+drop", head="fcn",
        pretrain="", num_classes=CLASSES, input_size=512,
        gradient_checkpointing=True)).to(device)
    init_hash = state_hash(model)

    # EMA Teacher
    teacher = None
    if use_ema:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Graph
    graph = None
    if opt.graph:
        from utils.graph_stage1_training import GraphStage1TrainingRuntime
        from utils.sam_mask_cache import SamMaskCacheConfig, load_shared_sam_masks
        from utils.sam_node_audit import SamNodeAuditConfig
        cache_report = json.loads(
            (Path(opt.cache_root) / "HiUCDMini" / "cache_manifest.json").read_text())
        if not cache_report.get("complete") or cache_report.get("split_sha256") != manifest["split_sha256"]:
            raise RuntimeError("Complete offline SAM cache for this exact split is required")
        train_ids = {item["sample_id"] for item in manifest["patches"]["train"]}
        if set(cache_report["entries"]) != train_ids:
            raise RuntimeError("SAM cache entries do not exactly match training patches")
        graph = GraphStage1TrainingRuntime(
            model.backbone, CLASSES, node_dim=opt.graph_node_dim,
            audit_config=SamNodeAuditConfig(
                tau_gt=opt.sam_tau_gt, tau_purity=opt.sam_tau_purity,
                tau_u=opt.sam_tau_u, min_mask_pixels=4, min_changed_pixels=4),
            use_stage2=True,
            graph_local_topk=opt.graph_local_topk,
            graph_edge_threshold=opt.graph_edge_threshold,
        )
        graph.cache_loader = (load_shared_sam_masks,
                              SamMaskCacheConfig(min_area_ratio=opt.sam_min_area,
                                                 max_area_ratio=opt.sam_max_area))
        graph.cache_root = Path(opt.cache_root)
        graph.max_regions = opt.graph_max_regions

        from models.graph.pseudo_fusion import teacher_graph_complement, reproject_stage2_batch

    optimizer = torch.optim.AdamW([
        {"params": [v for k, v in model.named_parameters()
                    if v.requires_grad and "backbone" in k], "lr": opt.lr / 10},
        {"params": [v for k, v in model.named_parameters()
                    if v.requires_grad and "backbone" not in k], "lr": opt.lr}],
        weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    seed_all(opt.seed)
    scaler = torch.cuda.amp.GradScaler(enabled=not opt.no_amp, init_scale=128.0)

    config = vars(opt).copy()
    config.update(num_classes=CLASSES, split_sha256=manifest["split_sha256"],
                  initial_model_sha256=init_hash,
                  train_patches=len(train_ds), val_patches=len(val_ds),
                  dataset_protocol=manifest["protocol"],
                  use_ema_teacher=use_ema)
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

        pseudo_active = use_ema and (epoch >= opt.pseudo_start_epoch)
        sem_u_weight = pseudo_ramp_weight(
            epoch, opt.pseudo_start_epoch, opt.pseudo_rampup_epochs, opt.lambda_sem_u)
        graph_pseudo_weight = pseudo_ramp_weight(
            epoch, opt.pseudo_start_epoch, opt.pseudo_rampup_epochs, opt.graph_pixel_pseudo_weight)

        for index, raw_batch in enumerate(loader):
            batch = move(raw_batch, device)
            model_norm = 0.0

            # Load SAM masks for graph and SamMaskSupplement
            graph_data = None
            if graph:
                load_masks, cache_cfg = graph.cache_loader
                graph_data = []
                for name, geometry in zip(batch["name"], batch["geometry"]):
                    masks = load_masks(graph.cache_root, "HiUCDMini", "train",
                                       name, (512, 512), cache_cfg)
                    k, horizontal, vertical = [int(v) for v in geometry]
                    masks = torch.rot90(masks, k, (1, 2))
                    if horizontal:
                        masks = masks.flip(2)
                    if vertical:
                        masks = masks.flip(1)
                    graph_data.append({"shared_masks": masks[:graph.max_regions].contiguous()})
            masks_list = ([item["shared_masks"] for item in graph_data]
                          if graph_data is not None else None)

            # Student forward
            with torch.autocast("cuda", enabled=not opt.no_amp):
                if graph:
                    outputs, features = model(batch["A"], batch["B"],
                                              return_graph_features=True,
                                              detach_graph_features=True,
                                              masks_list=masks_list)
                else:
                    outputs = model(batch["A"], batch["B"], masks_list=masks_list)
                    features = None
                loss, parts = supervised_loss(outputs, batch)

            # Graph Stage1 + Stage2 losses
            loss_graph_gt = loss_edge_gt = loss_graph_temp = outputs[0].sum() * 0.0
            graph_result = None
            if graph:
                with torch.random.fork_rng(devices=[device.index or 0]):
                    torch.manual_seed(opt.seed + step)
                    graph_result = graph.compute(
                        batch["A"], batch["B"],
                        batch["label_A"], batch["label_B"],
                        graph_data,
                        backbone_features=tuple(x.float() for x in features),
                        stage2_enabled=True)
                loss_graph_gt = graph_result.loss_graph_gt
                loss_edge_gt = graph_result.loss_edge_gt
                loss_graph_temp = graph_result.loss_graph_temp
                loss_stage2_gt = graph_result.loss_stage2_graph_gt
                loss = (loss
                        + opt.lambda_graph_gt * (loss_graph_gt + loss_stage2_gt)
                        + opt.lambda_edge_gt * loss_edge_gt
                        + opt.graph_temp_weight * loss_graph_temp)

            # EMA Teacher pseudo labels
            loss_sem_u = outputs[0].sum() * 0.0
            pseudo_ratio_a = pseudo_ratio_b = candidate_ratio = 0.0
            if pseudo_active and teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", enabled=not opt.no_amp):
                    teacher_out = teacher(batch["A"], batch["B"])
                    t_A, t_B, t_change = teacher_out

                # Candidate mask: unchanged + teacher says unchanged
                gt_unchanged = (batch["label_A"] == 0) & (batch["label_B"] == 0)
                t_change_prob = torch.sigmoid(t_change.float()).squeeze(1)
                teacher_unchanged = t_change_prob < opt.pseudo_change_thr
                candidate_mask = gt_unchanged & teacher_unchanged
                candidate_ratio = float(candidate_mask.float().mean())

                if graph and graph_result is not None and graph_pseudo_weight > 0:
                    # Complement: graph + Teacher
                    graph_pixels = reproject_stage2_batch(
                        graph_result.stage2_result,
                        graph_result.audits,
                        graph_result.masks)
                    complement = teacher_graph_complement(
                        teacher_probs_a=torch.softmax(t_A.float(), dim=1),
                        teacher_probs_b=torch.softmax(t_B.float(), dim=1),
                        teacher_change_logits=t_change,
                        labels_a=batch["label_A"],
                        labels_b=batch["label_B"],
                        graph_pred_a=graph_pixels.pred_a,
                        graph_conf_a=graph_pixels.conf_a,
                        graph_valid_a=graph_pixels.valid_a,
                        graph_pred_b=graph_pixels.pred_b,
                        graph_conf_b=graph_pixels.conf_b,
                        graph_valid_b=graph_pixels.valid_b,
                        graph_conf_threshold=opt.graph_confidence_threshold,
                        change_threshold=opt.pseudo_change_thr,
                        num_classes=CLASSES,
                        ignore_index=PSEUDO_IGNORE,
                    )
                    pseudo_a, pseudo_b = complement.pseudo_a, complement.pseudo_b
                    pseudo_ratio_a = float((pseudo_a != PSEUDO_IGNORE).float().mean())
                    pseudo_ratio_b = float((pseudo_b != PSEUDO_IGNORE).float().mean())
                    with torch.autocast("cuda", enabled=not opt.no_amp):
                        loss_sem_u_a = F.cross_entropy(
                            outputs[0].float(), pseudo_a, ignore_index=PSEUDO_IGNORE)
                        loss_sem_u_b = F.cross_entropy(
                            outputs[1].float(), pseudo_b, ignore_index=PSEUDO_IGNORE)
                    w = max(sem_u_weight, graph_pseudo_weight)
                    loss_sem_u = (loss_sem_u_a + loss_sem_u_b) / 2.0
                    loss = loss + w * loss_sem_u
                elif sem_u_weight > 0 and candidate_mask.any():
                    # Teacher-only pseudo labels
                    t_probs_a = torch.softmax(t_A.float(), dim=1)
                    t_probs_b = torch.softmax(t_B.float(), dim=1)
                    pseudo_a = entropy_pseudo_labels(
                        t_probs_a, candidate_mask, CLASSES, opt.pseudo_keep_ratio)
                    pseudo_b = entropy_pseudo_labels(
                        t_probs_b, candidate_mask, CLASSES, opt.pseudo_keep_ratio)
                    pseudo_ratio_a = float((pseudo_a != PSEUDO_IGNORE).float().mean())
                    pseudo_ratio_b = float((pseudo_b != PSEUDO_IGNORE).float().mean())
                    with torch.autocast("cuda", enabled=not opt.no_amp):
                        loss_sem_u_a = F.cross_entropy(
                            outputs[0].float(), pseudo_a, ignore_index=PSEUDO_IGNORE)
                        loss_sem_u_b = F.cross_entropy(
                            outputs[1].float(), pseudo_b, ignore_index=PSEUDO_IGNORE)
                    loss_sem_u = (loss_sem_u_a + loss_sem_u_b) / 2.0
                    loss = loss + sem_u_weight * loss_sem_u

            if not torch.isfinite(loss) or not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError("Nonfinite loss/output; stopping")
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
                    # EMA Teacher update after optimizer step
                    if teacher is not None:
                        update_ema_teacher(model, teacher, opt.ema_decay)
                else:
                    nonfinite_skips += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if graph:
                    graph.zero_grad()

            step += 1
            if index == 0 or (index + 1) % 50 == 0:
                record = {
                    "epoch": epoch, "step": step, "batch": index + 1,
                    "loss": float(loss.detach()),
                    "grad_norm": float(model_norm) if (index + 1) % accum == 0 else None,
                    **{k: float(v.detach()) for k, v in parts.items()},
                    "loss_sem_u": float(loss_sem_u.detach()),
                    "sem_u_weight": sem_u_weight,
                    "pseudo_ratio_a": pseudo_ratio_a,
                    "pseudo_ratio_b": pseudo_ratio_b,
                    "candidate_ratio": candidate_ratio,
                }
                if graph:
                    record.update(
                        graph_loss=float((opt.lambda_graph_gt * (loss_graph_gt + loss_stage2_gt)
                                          + opt.lambda_edge_gt * loss_edge_gt).detach()),
                        graph_gt=float(loss_graph_gt.detach()),
                        graph_stage2_gt=float(loss_stage2_gt.detach()),
                        graph_temp=float(loss_graph_temp.detach()),
                        graph_edge=float(loss_edge_gt.detach()),
                        graph_gt_nodes=graph_result.gt_node_count,
                        graph_pseudo_weight=graph_pseudo_weight,
                    )
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
        payload = {
            "model": model.state_dict(),
            "teacher": teacher.state_dict() if teacher else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1, "step": step, "best": best, "config": config,
            "graph": graph.state_dict() if graph else None,
        }
        atomic_save(payload, output / "last.pth")
        if score >= best:
            atomic_save(payload, output / "best.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

    print("TRAINING_COMPLETE", json.dumps({
        "best_Fscd": best, "elapsed_s": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
