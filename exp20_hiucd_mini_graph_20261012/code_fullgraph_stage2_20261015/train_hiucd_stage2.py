#!/usr/bin/env python3
"""Matched fully supervised baseline vs. SAM graph Stage-1 experiment."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.hiucd_mini import HiUCDMini
from utils.graph_stage1_training import GraphStage1TrainingRuntime
from models.graph.pseudo_fusion import (
    graph_pixel_pseudo_loss, reproject_stage2_batch,
)
from utils.pseudo_label import generate_entropy_pseudo_label

CLASSES = 10


class GraphCachedHiUCDMini(HiUCDMini):
    def __init__(self, *args, cache_root, load_masks, cache_config,
                 audit_config, max_regions, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_root = Path(cache_root)
        self.load_masks = load_masks
        self.cache_config = cache_config
        self.audit_config = audit_config
        self.max_regions = int(max_regions)
        self._audit_cache = {}

    def __getitem__(self, index):
        from utils.sam_node_audit import audit_sam_nodes

        sample = super().__getitem__(index)
        name = sample["name"]
        masks = self.load_masks(self.cache_root, "HiUCDMini", "train", name,
                                (512, 512), self.cache_config)
        k, horizontal, vertical = [int(value) for value in sample["geometry"]]
        masks = torch.rot90(masks, k, (1, 2))
        if horizontal:
            masks = masks.flip(2)
        if vertical:
            masks = masks.flip(1)
        if self.max_regions > 0:
            masks = masks[:self.max_regions]
        masks = masks.contiguous()

        audit = self._audit_cache.get(name)
        if audit is None:
            # Synchronized transforms preserve per-node counts and labels.
            audit = audit_sam_nodes(masks, sample["label_A"], sample["label_B"],
                                    CLASSES, self.audit_config)
            self._audit_cache[name] = audit
        sample["_graph_data"] = {"shared_masks": masks}
        sample["_graph_audit"] = audit
        return sample


def collate_graph_samples(samples):
    graph_data = [sample.pop("_graph_data") for sample in samples]
    audits = [sample.pop("_graph_audit") for sample in samples]
    batch = default_collate(samples)
    batch["graph_data"] = graph_data
    batch["graph_audits"] = audits
    return batch


def copy_best_checkpoint_from_history(resume_path, destination):
    source_dir = Path(resume_path).parent
    metrics_path = source_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError("Cannot resume best tracking without prior metrics.jsonl")
    candidates = []
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        score = record.get("Fscd")
        if isinstance(score, (int, float)) and math.isfinite(score):
            candidates.append((float(score), int(record["epoch"])))
    if not candidates:
        raise RuntimeError("Prior metrics.jsonl contains no valid checkpoint scores")
    score, epoch = max(candidates, key=lambda item: (item[0], item[1]))
    source = source_dir / ("epoch%03d.pth" % epoch)
    if not source.is_file():
        raise FileNotFoundError("Prior best checkpoint is missing: " + str(source))
    shutil.copy2(source, destination)
    return score


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(0)


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


def update_ema_teacher(student, teacher, decay, step):
    effective = min(1.0 - 1.0 / float(step + 1), float(decay))
    with torch.no_grad():
        for teacher_value, student_value in zip(teacher.state_dict().values(), student.state_dict().values()):
            if teacher_value.dtype.is_floating_point:
                teacher_value.mul_(effective).add_(student_value, alpha=1.0-effective)
            else:
                teacher_value.copy_(student_value)


def entropy_pseudo_pair(outputs, teacher_outputs, batch, epoch, opt, force=False):
    active = epoch >= opt.pseudo_start_epoch or force
    zero = outputs[0].sum() * 0.0
    if not active:
        return zero, 0, 0.0
    ramp = 1.0 if force else min(1.0, (epoch-opt.pseudo_start_epoch+1) / max(opt.pseudo_ramp_epochs, 1))
    alpha_progress = 0.0 if force else min(1.0, max(0.0, (epoch-opt.pseudo_start_epoch) / max(opt.epochs-opt.pseudo_start_epoch, 1)))
    alpha = opt.pseudo_alpha_start + (opt.pseudo_alpha_end-opt.pseudo_alpha_start)*alpha_progress
    changed = batch["change"].bool()
    candidate = batch["change_valid"] & (~changed)
    candidate &= (batch["label_A"] == 0) & (batch["label_B"] == 0)
    candidate &= torch.sigmoid(teacher_outputs[2].float()).squeeze(1) < opt.pseudo_change_threshold
    gamma = torch.full((CLASSES,), alpha, device=outputs[0].device)
    gamma[0] = 0.0
    pseudo_a, reliable_a, _ = generate_entropy_pseudo_label(teacher_outputs[0], candidate, CLASSES, gamma)
    pseudo_b, reliable_b, _ = generate_entropy_pseudo_label(teacher_outputs[1], candidate, CLASSES, gamma)
    terms = []
    if reliable_a.any(): terms.append(F.cross_entropy(outputs[0].float(), pseudo_a, ignore_index=255))
    if reliable_b.any(): terms.append(F.cross_entropy(outputs[1].float(), pseudo_b, ignore_index=255))
    loss = sum(terms)/len(terms) if terms else zero
    return loss * (opt.lambda_sem_u*ramp), int((reliable_a | reliable_b).sum()), alpha


def graph_veto_for_hiucd(teacher_outputs, batch, graph_pixels, opt):
    probs_a = torch.softmax(teacher_outputs[0].float(), dim=1)
    probs_b = torch.softmax(teacher_outputs[1].float(), dim=1)
    conf_a, pred_a = probs_a.max(1); conf_b, pred_b = probs_b.max(1)
    unchanged = batch["change_valid"] & (~batch["change"].bool())
    unchanged &= torch.sigmoid(teacher_outputs[2].float()).squeeze(1) < opt.pseudo_change_threshold
    semantic_ok = (pred_a > 0) & (pred_b > 0) & (pred_a < CLASSES) & (pred_b < CLASSES)
    semantic_ok &= (pred_a == pred_b)
    teacher_ok = (conf_a >= opt.teacher_threshold) & (conf_b >= opt.teacher_threshold)
    candidate = unchanged & semantic_ok & teacher_ok
    ga = graph_pixels.valid_a & (graph_pixels.conf_a >= opt.graph_confidence_threshold)
    gb = graph_pixels.valid_b & (graph_pixels.conf_b >= opt.graph_confidence_threshold)
    ga &= (graph_pixels.pred_a > 0) & (graph_pixels.pred_a < CLASSES)
    gb &= (graph_pixels.pred_b > 0) & (graph_pixels.pred_b < CLASSES)
    conflict = (ga & (graph_pixels.pred_a != pred_a)) | (gb & (graph_pixels.pred_b != pred_b))
    has_graph = ga | gb
    accepted = candidate & has_graph & (~conflict)
    accepted |= candidate & (~has_graph) & (conf_a >= opt.teacher_fallback_threshold) & (conf_b >= opt.teacher_fallback_threshold)
    target = torch.full_like(pred_a, 255)
    target[accepted] = pred_a[accepted]
    if bool((accepted & batch["change"].bool()).any()):
        raise AssertionError("Hi-UCD graph pseudo labels leaked into changed pixels")
    return target, accepted


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
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--resume", default="")
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
    parser.add_argument("--graph-max-regions", type=int, default=0,
                        help="0 keeps every cached SAM node")
    parser.add_argument("--stage2-start-epoch", type=int, default=55)
    parser.add_argument("--pseudo-start-epoch", type=int, default=30)
    parser.add_argument("--pseudo-ramp-epochs", type=int, default=5)
    parser.add_argument("--lambda-sem-u", type=float, default=0.5)
    parser.add_argument("--pseudo-alpha-start", type=float, default=0.5)
    parser.add_argument("--pseudo-alpha-end", type=float, default=0.8)
    parser.add_argument("--graph-pixel-weight", type=float, default=0.05)
    parser.add_argument("--graph-pseudo-ramp-epochs", type=int, default=10)
    parser.add_argument("--graph-local-topk", type=int, default=8)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.5)
    parser.add_argument("--teacher-threshold", type=float, default=0.9)
    parser.add_argument("--teacher-fallback-threshold", type=float, default=0.95)
    parser.add_argument("--graph-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--pseudo-change-threshold", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--graph-temp-weight", type=float, default=0.02)
    opt = parser.parse_args()
    if opt.verify_graph_gradient and (not opt.graph or not opt.smoke_steps):
        parser.error("Gradient verification requires --graph and --smoke-steps")
    if opt.smoke_steps and opt.benchmark_steps:
        parser.error("--smoke-steps and --benchmark-steps are mutually exclusive")
    if opt.benchmark_steps < 0 or opt.smoke_steps < 0:
        parser.error("Step limits must be non-negative")
    if opt.resume and opt.smoke_steps:
        parser.error("--resume cannot be combined with --smoke-steps")
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
    from main_model import ChangeDetection
    from types import SimpleNamespace
    model = ChangeDetection(SimpleNamespace(backbone="msam_96", neck="fpn+drop", head="fcn",
        pretrain="", num_classes=CLASSES, input_size=512, gradient_checkpointing=True)).to(device)
    init_hash = state_hash(model)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
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
                tau_u=opt.sam_tau_u, min_mask_pixels=4, min_changed_pixels=4),
            use_stage2=True, graph_local_topk=opt.graph_local_topk,
            graph_edge_threshold=opt.graph_edge_threshold)
        cache_config = SamMaskCacheConfig(min_area_ratio=opt.sam_min_area,
                                          max_area_ratio=opt.sam_max_area)
        graph.cache_loader = (load_shared_sam_masks, cache_config)
        graph.cache_root = Path(opt.cache_root)
        graph.max_regions = opt.graph_max_regions
        train = GraphCachedHiUCDMini(
            opt.root, split_path, "train", augment=not bool(opt.smoke_steps),
            cache_root=opt.cache_root, load_masks=load_shared_sam_masks,
            cache_config=cache_config, audit_config=graph.audit_config,
            max_regions=graph.max_regions)
    train_loader_options = {
        "batch_size": opt.batch_size, "shuffle": True, "num_workers": opt.workers,
        "pin_memory": True, "worker_init_fn": seed_worker, "generator": loader_gen,
    }
    if opt.workers:
        train_loader_options.update(persistent_workers=True, prefetch_factor=2)
    if graph:
        train_loader_options["collate_fn"] = collate_graph_samples
    loader = DataLoader(train, **train_loader_options)
    validation = DataLoader(val, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.workers, pin_memory=True,
                            persistent_workers=False)
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
        graph_policy="all cached SAM nodes; Stage1 GT node/edge + Stage2 sparse relation propagation and EMA pseudo labels",
        stage2_start_epoch=opt.stage2_start_epoch,
        pseudo_start_epoch=opt.pseudo_start_epoch,
        pseudo_ramp_epochs=opt.pseudo_ramp_epochs,
        lambda_sem_u=opt.lambda_sem_u,
        pseudo_alpha_start=opt.pseudo_alpha_start,
        pseudo_alpha_end=opt.pseudo_alpha_end,
        graph_pixel_weight=opt.graph_pixel_weight,
        graph_pseudo_ramp_epochs=opt.graph_pseudo_ramp_epochs,
        ema_decay=opt.ema_decay)
    config.update(speedup_version="worker_cached_graph_data_v1",
                  resume_checkpoint=str(Path(opt.resume).resolve()) if opt.resume else "",
                  exact_rng_resume=False)
    print("CONFIG", json.dumps(config, sort_keys=True), flush=True)
    best, step, start_epoch, started = -1.0, 0, 0, time.time()
    if opt.resume:
        checkpoint = torch.load(opt.resume, map_location="cpu")
        saved_config = checkpoint.get("config", {})
        match_keys = ("root", "split_json", "epochs", "seed", "batch_size",
                      "effective_batch_size", "workers", "lr", "graph", "cache_root",
                      "no_amp", "smoke_steps", "lambda_graph_gt", "lambda_edge_gt",
                      "sam_tau_gt", "sam_tau_purity", "sam_tau_u", "sam_min_area",
                      "sam_max_area", "graph_node_dim", "graph_max_regions",
                      "stage2_start_epoch", "pseudo_start_epoch", "pseudo_ramp_epochs",
                      "lambda_sem_u", "pseudo_alpha_start", "pseudo_alpha_end",
                      "graph_pixel_weight", "graph_pseudo_ramp_epochs",
                      "graph_local_topk", "graph_edge_threshold", "teacher_threshold",
                      "teacher_fallback_threshold", "graph_confidence_threshold",
                      "pseudo_change_threshold", "ema_decay", "graph_temp_weight")
        mismatches = {key: (saved_config.get(key), getattr(opt, key))
                      for key in match_keys if saved_config.get(key) != getattr(opt, key)}
        if mismatches:
            raise ValueError("Resume configuration mismatch: " + json.dumps(mismatches, sort_keys=True))
        if saved_config.get("split_sha256") != manifest["split_sha256"]:
            raise ValueError("Resume checkpoint split hash does not match current split")
        if saved_config.get("initial_model_sha256") != init_hash:
            raise ValueError("Resume checkpoint initialization hash does not match this model")
        model.load_state_dict(checkpoint["model"], strict=True)
        teacher.load_state_dict(checkpoint.get("teacher", checkpoint["model"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if graph:
            if checkpoint.get("graph") is None:
                raise ValueError("Resume checkpoint is missing graph optimizer/state")
            graph.load_state_dict(checkpoint["graph"])
        elif checkpoint.get("graph") is not None:
            raise ValueError("Graph checkpoint cannot be resumed without --graph")
        start_epoch = int(checkpoint["epoch"])
        step = int(checkpoint["step"])
        best = float(checkpoint.get("best", -1.0))
        if start_epoch >= opt.epochs:
            raise ValueError("Resume checkpoint has already reached requested epochs")
        if not opt.benchmark_steps:
            history_best = copy_best_checkpoint_from_history(opt.resume, output / "best.pth")
            best = max(best, history_best)
        print("RESUME", json.dumps({"start_epoch": start_epoch, "step": step,
                                   "best": best, "source": str(Path(opt.resume).resolve())}),
              flush=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    nonfinite_skips = 0
    accum = 1 if opt.smoke_steps else opt.effective_batch_size // opt.batch_size
    benchmark_start = None
    benchmark_done = 0
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        if graph:
            graph.train()
            graph.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        interval_start = time.perf_counter()
        interval_data_wait = 0.0
        interval_host_step = 0.0
        interval_batches = 0
        for index in range(len(loader)):
            fetch_start = time.perf_counter()
            raw_batch = next(iterator)
            interval_data_wait += time.perf_counter() - fetch_start
            step_start = time.perf_counter()
            if benchmark_start is None:
                benchmark_start = fetch_start
            batch = move(raw_batch, device)
            graph_data = None
            audits = None
            if graph:
                graph_data = [{"shared_masks": item["shared_masks"].to(
                    device, non_blocking=True)} for item in raw_batch["graph_data"]]
                audits = raw_batch["graph_audits"]
            with torch.autocast("cuda", enabled=not opt.no_amp):
                if graph:
                    outputs, features = model(batch["A"], batch["B"], return_graph_features=True,
                                              detach_graph_features=False)
                else:
                    outputs = model(batch["A"], batch["B"])
                    features = None
                loss, parts = supervised_loss(outputs, batch)
            force_stage2 = bool(opt.smoke_steps)
            stage2_active = epoch >= opt.stage2_start_epoch or force_stage2
            with torch.no_grad(), torch.autocast("cuda", enabled=not opt.no_amp):
                teacher_outputs = teacher(batch["A"], batch["B"])
            if graph:
                with torch.random.fork_rng(devices=[device.index or 0]):
                    torch.manual_seed(opt.seed + step)
                    masks = [item["shared_masks"] for item in graph_data]
                    nodes = graph.projector(*(x.float() for x in features), masks)
                    graph_result = graph.stage1(nodes, audits)
                    stage2_result = None
                    if stage2_active:
                        live_nodes = graph.projector(*(x.float() for x in features), masks)
                        stage2_result = graph.stage2(live_nodes, audits, graph.stage1)
                graph_loss = opt.lambda_graph_gt * graph_result.loss_graph_gt + opt.lambda_edge_gt * graph_result.loss_edge_gt
                if stage2_result is not None:
                    graph_loss = graph_loss + opt.lambda_graph_gt * stage2_result.loss_graph_gt + opt.graph_temp_weight * stage2_result.loss_temporal
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
            semantic_pseudo_loss, semantic_pseudo_pixels, alpha_now = entropy_pseudo_pair(
                outputs, teacher_outputs, batch, epoch, opt, force=bool(opt.smoke_steps))
            graph_pixel_loss = outputs[0].sum() * 0.0
            graph_pseudo_pixels = 0
            pixel_weight = 0.0
            if stage2_active and stage2_result is not None:
                graph_pixels = reproject_stage2_batch(stage2_result, audits, [item["shared_masks"] for item in graph_data])
                graph_target, graph_valid = graph_veto_for_hiucd(teacher_outputs, batch, graph_pixels, opt)
                graph_pixel_loss = graph_pixel_pseudo_loss(outputs[0], outputs[1], graph_target, graph_valid, ignore_index=255)
                ramp = 1.0 if opt.smoke_steps else min(1.0, max(0.0, (epoch-opt.stage2_start_epoch) / max(opt.graph_pseudo_ramp_epochs, 1)))
                pixel_weight = opt.graph_pixel_weight * ramp
                graph_pseudo_pixels = int(graph_valid.sum())
            loss = loss + semantic_pseudo_loss + pixel_weight * graph_pixel_loss
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
                    update_ema_teacher(model, teacher, opt.ema_decay, step)
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
            benchmark_done += 1
            interval_batches += 1
            interval_host_step += time.perf_counter() - step_start
            if index == 0 or (index + 1) % 50 == 0:
                torch.cuda.synchronize(device)
                interval_elapsed = time.perf_counter() - interval_start
                record = {"epoch": epoch, "step": step, "batch": index + 1,
                          "interval_s": round(interval_elapsed, 3),
                          "seconds_per_batch": round(interval_elapsed / interval_batches, 4),
                          "data_wait_s": round(interval_data_wait, 3),
                          "host_step_s": round(interval_host_step, 3),
                          "loss": float(loss.detach()), "grad_norm": float(model_norm) if (index + 1) % accum == 0 else None,
                          **{k: float(v.detach()) for k, v in parts.items()}}
                if graph:
                    record.update(graph_loss=float(graph_loss.detach()), graph_gt=float(graph_result.loss_graph_gt.detach()),
                                  graph_edge=float(graph_result.loss_edge_gt.detach()), graph_gt_nodes=graph_result.gt_node_count,
                                  stage2_gt=float(stage2_result.loss_graph_gt.detach()) if stage2_result is not None else 0.0,
                                  stage2_temporal=float(stage2_result.loss_temporal.detach()) if stage2_result is not None else 0.0,
                                  stage2_nodes=stage2_result.node_count if stage2_result is not None else 0,
                                  full_sam_nodes=sum(int(item["shared_masks"].shape[0]) for item in graph_data),
                                  semantic_pseudo=float(semantic_pseudo_loss.detach()), semantic_pseudo_pixels=semantic_pseudo_pixels,
                                  pseudo_alpha=alpha_now, graph_pixel_pseudo=float(graph_pixel_loss.detach()),
                                  graph_pixel_weight=pixel_weight, graph_pseudo_pixels=graph_pseudo_pixels)
                record["nonfinite_gradient_skips"] = nonfinite_skips
                print("TRAIN", json.dumps(record), flush=True)
                interval_start = time.perf_counter()
                interval_data_wait = 0.0
                interval_host_step = 0.0
                interval_batches = 0
            if opt.smoke_steps and step >= opt.smoke_steps:
                smoke_payload = {"model": model.state_dict(), "teacher": teacher.state_dict(),
                                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                 "graph": graph.state_dict(), "epoch": 0, "step": step,
                                 "config": config}
                smoke_checkpoint = output / "smoke_checkpoint.pth"
                atomic_save(smoke_payload, smoke_checkpoint)
                restored = torch.load(smoke_checkpoint, map_location="cpu")
                if "stage2" not in restored["graph"] or "teacher" not in restored:
                    raise AssertionError("Smoke checkpoint omitted EMA teacher or Stage2 state")
                print("SMOKE_CHECKPOINT_VERIFIED", json.dumps({"path": str(smoke_checkpoint),
                      "graph_stage": restored["graph"]["graph_stage"], "step": restored["step"]}), flush=True)
                print("SMOKE_COMPLETE", flush=True)
                return
            if opt.benchmark_steps and benchmark_done >= opt.benchmark_steps:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - benchmark_start
                print("BENCHMARK_COMPLETE", json.dumps({"microbatches": benchmark_done,
                    "elapsed_s": round(elapsed, 3),
                    "seconds_per_batch": round(elapsed / benchmark_done, 4)}), flush=True)
                return
        metrics = evaluate(model, validation, device, not opt.no_amp)
        print("VAL", epoch, json.dumps(metrics), flush=True)
        score = metrics["Fscd"] if metrics["Fscd"] is not None else -1.0
        improved = score > best
        best = max(best, score)
        scheduler.step()
        payload = {"model": model.state_dict(), "teacher": teacher.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "epoch": epoch + 1, "step": step, "best": best, "config": config,
                   "graph": graph.state_dict() if graph else None}
        atomic_save(payload, output / ("epoch%03d.pth" % epoch))
        atomic_save(payload, output / "last.pth")
        if improved:
            atomic_save(payload, output / "best.pth")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
    print("TRAINING_COMPLETE", json.dumps({"best_Fscd": best, "elapsed_s": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()

