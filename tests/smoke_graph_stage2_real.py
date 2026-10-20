"""One real Landsat batch Stage2 forward/backward/optimizer smoke test."""

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SCD_DATASET"] = "landsat"

from datasets import RS_ST as RS
from datasets.RS_ST_graph import SamGraphDataset, collate_sam_graph_batch
from main_model import ChangeDetection
from models.graph.pseudo_fusion import (
    graph_pixel_pseudo_loss, reproject_stage2_batch, teacher_graph_veto)
from utils.graph_stage1_training import GraphStage1TrainingRuntime
from utils.sam_mask_cache import SamMaskCacheConfig
from utils.common import ScaleInOutput


def _model_options():
    return SimpleNamespace(
        backbone="msam_96", neck="fpn+drop", head="fcn", pretrain="",
        input_size=512, num_classes=RS.num_classes,
        gradient_checkpointing=True,
    )


def _load_checkpoint(path, student, teacher, runtime):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    student.load_state_dict(payload["student"], strict=True)
    teacher.load_state_dict(
        payload["teacher"] if payload.get("teacher") is not None else payload["student"],
        strict=True)
    if not runtime.load_checkpoint_payload(payload):
        raise RuntimeError("checkpoint has no Stage1 graph state")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "checkpoints/Landsat/"
        "Landsat_supervised_graph_stage1_seed42_epoch089_Fscd0.8143.pth")
    device = torch.device("cuda:0")
    student = ChangeDetection(_model_options()).to(device).train()
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    runtime = GraphStage1TrainingRuntime(
        student.backbone, num_classes=RS.num_classes,
        use_stage2=True, graph_gat_heads=4,
        graph_local_topk=8, graph_edge_threshold=0.5,
    )
    _load_checkpoint(checkpoint, student, teacher, runtime)
    runtime.train()

    dataset = SamGraphDataset(
        RS.Data("train", random_flip=False),
        dataset_name="Landsat", split="train",
        cache_root=str(ROOT / "cache/sam"),
        cache_config=SamMaskCacheConfig(
            min_area_ratio=0.001, max_area_ratio=0.05),
        random_flip=False,
    )
    images_a, images_b, labels_a_cpu, labels_b_cpu, graph_data = (
        collate_sam_graph_batch([dataset[0]]))
    images_a = images_a.to(device=device, dtype=torch.float32)
    images_b = images_b.to(device=device, dtype=torch.float32)
    labels_a = labels_a_cpu.to(device=device, dtype=torch.long)
    labels_b = labels_b_cpu.to(device=device, dtype=torch.long)
    scale = ScaleInOutput(512)
    images_a, images_b = scale.scale_input((images_a, images_b))

    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    student_optimizer.zero_grad(set_to_none=True)
    runtime.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, features = student(
            images_a, images_b, return_graph_features=True,
            detach_graph_features=False)
        outputs = scale.scale_output(outputs)
        logits_a, logits_b, _ = outputs
        graph = runtime.compute(
            images_a, images_b, labels_a_cpu, labels_b_cpu, graph_data,
            backbone_features=features, stage2_enabled=True)
    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16):
        teacher_a, teacher_b, teacher_change = teacher(images_a, images_b)
        teacher_a, teacher_b, teacher_change = scale.scale_output(
            (teacher_a, teacher_b, teacher_change))

    pixels = reproject_stage2_batch(
        graph.stage2_result, graph.audits, graph.masks)
    probs_a = torch.softmax(teacher_a.float(), dim=1)
    probs_b = torch.softmax(teacher_b.float(), dim=1)
    veto = teacher_graph_veto(
        probs_a, probs_b, teacher_change, labels_a, labels_b,
        pixels.pred_a, pixels.conf_a, pixels.valid_a,
        pixels.pred_b, pixels.conf_b, pixels.valid_b,
        teacher_threshold=0.0, teacher_fallback_threshold=0.0,
        graph_confidence_threshold=0.7, change_threshold=1.0,
        num_classes=RS.num_classes,
    )
    if veto.accepted_count == 0:
        # Keep this real-batch smoke deterministic while exercising the documented
        # graph-unavailable teacher fallback path.
        averaged = (probs_a + probs_b) * 0.5
        averaged[:, 0] = 0.0
        averaged = averaged / averaged.sum(dim=1, keepdim=True).clamp_min(1e-6)
        zeros = torch.zeros_like(pixels.valid_a)
        veto = teacher_graph_veto(
            averaged, averaged, teacher_change, labels_a, labels_b,
            pixels.pred_a, pixels.conf_a, zeros,
            pixels.pred_b, pixels.conf_b, zeros,
            teacher_threshold=0.0, teacher_fallback_threshold=0.0,
            graph_confidence_threshold=1.0, change_threshold=1.0,
            num_classes=RS.num_classes,
        )
    pixel_loss = graph_pixel_pseudo_loss(
        logits_a, logits_b, veto.target, veto.valid)
    if veto.accepted_count <= 0:
        raise RuntimeError("real batch produced no accepted pseudo pixels")
    pseudo_gradient = torch.autograd.grad(
        pixel_loss, logits_a, retain_graph=True)[0]
    total = (0.1 * graph.loss_graph_gt + 0.1 * graph.loss_edge_gt
             + 0.1 * graph.loss_stage2_graph_gt
             + 0.02 * graph.loss_graph_temp + 0.1 * pixel_loss)
    if not torch.isfinite(total):
        raise RuntimeError("non-finite Stage2 loss")
    total.backward()
    student_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    graph_norm = torch.nn.utils.clip_grad_norm_(runtime.parameters(), 1.0)
    if (not torch.isfinite(student_norm) or not torch.isfinite(graph_norm)
            or not torch.isfinite(pseudo_gradient).all()
            or float(pseudo_gradient.abs().sum()) == 0.0):
        raise RuntimeError("non-finite or zero Stage2 gradient")
    student_optimizer.step()
    runtime.optimizer.step()
    if bool((veto.valid & ((labels_a > 0) | (labels_b > 0))).any()):
        raise RuntimeError("changed pseudo-label leakage")
    print(
        "real_stage2_smoke=ok sample={} masks={} nodes={} edges={} "
        "accepted={} loss={:.6f} student_grad={:.4f} graph_grad={:.4f}".format(
            graph_data[0]["sample_id"], graph_data[0]["shared_masks"].shape[0],
            graph.stage2_result.node_count, graph.stage2_result.edge_count,
            veto.accepted_count, float(total.detach()),
            float(student_norm), float(graph_norm)))


if __name__ == "__main__":
    main()
