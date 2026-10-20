import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

_dataset_parser = argparse.ArgumentParser(add_help=False)
_dataset_parser.add_argument(
    "--dataset", choices=("landsat", "second"),
    default=os.environ.get("SCD_DATASET", "landsat").lower())
_selected_dataset, _ = _dataset_parser.parse_known_args()
os.environ["SCD_DATASET"] = _selected_dataset.dataset

from configs.datasets import get_dataset_config
from datasets import RS_ST as RS
from main_model import ChangeDetection
from utils.common import ScaleInOutput
from utils.pseudo_checkpoint import (
    select_model_state, validate_checkpoint_num_classes)


NET_NAME = "SAMNet"
DATASET_CONFIG = get_dataset_config(_selected_dataset.dataset)
DATA_NAME = DATASET_CONFIG.name


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor_to_uint8_img(x):
    """
    x: Tensor [C,H,W] or [H,W]
    将输入图像转成可保存的 uint8 RGB。
    如果原图做过归一化，这里用 min-max 拉伸，主要用于可视化。
    """
    x = x.detach().cpu().float()

    if x.dim() == 3:
        if x.shape[0] in [1, 3]:
            x = x.permute(1, 2, 0)
        x = x.numpy()
    elif x.dim() == 2:
        x = x.numpy()
    else:
        raise ValueError("Unsupported image shape: {}".format(x.shape))

    x_min, x_max = np.min(x), np.max(x)
    x = (x - x_min) / (x_max - x_min + 1e-6)
    x = (x * 255).clip(0, 255).astype(np.uint8)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x


def binary_to_color(mask):
    """
    mask: [H,W] bool or 0/1
    输出黑白二值图。
    """
    mask = mask.astype(bool)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def conf_to_gray(conf):
    """
    conf: [H,W], 0~1
    输出灰度置信度热力图。
    """
    conf = np.nan_to_num(conf)
    conf = (conf * 255).clip(0, 255).astype(np.uint8)
    return np.stack([conf, conf, conf], axis=-1)


def label_to_color(label):
    """
    label: [H,W], 类别编号
    优先使用数据集自带 Index2Color。
    """
    label = label.astype(np.int64)
    try:
        color = RS.Index2Color(label)
        color = np.asarray(color)
        if color.max() <= 1:
            color = (color * 255).astype(np.uint8)
        else:
            color = color.astype(np.uint8)
        return color
    except Exception:
        # 兜底调色板，避免 Index2Color 出错
        palette = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 128, 0],
            [128, 0, 128],
            [0, 128, 128],
            [255, 128, 0],
            [128, 255, 0],
            [0, 128, 255],
            [128, 0, 255],
            [255, 0, 128],
            [0, 255, 128],
        ], dtype=np.uint8)
        label = np.clip(label, 0, len(palette) - 1)
        return palette[label]


def load_checkpoint(model, pth_path, device, checkpoint_model, dataset_name,
                    num_classes):
    print("Loading checkpoint:", pth_path)
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict, selected_owner = select_model_state(
        checkpoint, checkpoint_model)
    validate_checkpoint_num_classes(
        state_dict, num_classes, dataset_name)
    model.load_state_dict(state_dict, strict=True)

    print("Dataset:", dataset_name)
    print("Checkpoint model:", selected_owner)
    print("Checkpoint loaded strictly: {} parameters".format(
        len(state_dict)))


def compute_binary_metrics(tp, fp, fn, tn):
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return precision, recall, f1, iou, acc


def update_semantic_iou(pred, label, mask, num_classes, inter, union):
    """
    只在 mask=True 的位置评估语义预测。
    一般 mask = label > 0，也就是只在有语义真值的变化区域评估。
    """
    pred = pred[mask]
    label = label[mask]

    for c in range(1, num_classes):
        pred_c = pred == c
        label_c = label == c
        inter[c] += np.logical_and(pred_c, label_c).sum()
        union[c] += np.logical_or(pred_c, label_c).sum()


def save_visualization(
    save_dir,
    idx_name,
    img_A,
    img_B,
    gt_A,
    gt_B,
    gt_change,
    pseudo_change,
    scd_pred_A,
    scd_pred_B,
    pseudo_semantic_valid,
    pseudo_conf
):
    img_A_vis = tensor_to_uint8_img(img_A)
    img_B_vis = tensor_to_uint8_img(img_B)

    gt_change_vis = binary_to_color(gt_change)
    pseudo_change_vis = binary_to_color(pseudo_change)
    gt_A_vis = label_to_color(gt_A)
    gt_B_vis = label_to_color(gt_B)
    scd_pred_A_vis = label_to_color(scd_pred_A)
    scd_pred_B_vis = label_to_color(scd_pred_B)
    pseudo_sem_vis = label_to_color(pseudo_semantic_valid)
    conf_vis = conf_to_gray(pseudo_conf)

    io.imsave(os.path.join(save_dir, idx_name + "_A.png"), img_A_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_B.png"), img_B_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_gt_A.png"), gt_A_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_gt_B.png"), gt_B_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_gt_change.png"), gt_change_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_pseudo_change.png"), pseudo_change_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_scd_pred_A.png"), scd_pred_A_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_scd_pred_B.png"), scd_pred_B_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_pseudo_sem_valid.png"), pseudo_sem_vis)
    io.imsave(os.path.join(save_dir, idx_name + "_conf.png"), conf_vis)

    concat = np.concatenate([
        img_A_vis,
        img_B_vis,
        gt_change_vis,
        pseudo_change_vis,
        pseudo_sem_vis,
        conf_vis
    ], axis=1)

    io.imsave(os.path.join(save_dir, idx_name + "_compare.png"), concat)

    scd_concat = np.concatenate([
        img_A_vis,
        img_B_vis,
        gt_change_vis,
        pseudo_change_vis,
        gt_A_vis,
        scd_pred_A_vis,
        gt_B_vis,
        scd_pred_B_vis
    ], axis=1)

    io.imsave(os.path.join(save_dir, idx_name + "_scd_compare.png"), scd_concat)


def inspect_pseudo_quality(val_loader, net, scale, device, opt):
    net.eval()
    torch.cuda.empty_cache()

    save_dir = os.path.join(
        working_path,
        "visualizations",
        DATA_NAME,
        NET_NAME,
        "pseudo_quality"
    )
    mkdir(save_dir)

    num_classes = RS.num_classes

    # 二值变化伪标签统计
    change_tp = 0
    change_fp = 0
    change_fn = 0
    change_tn = 0

    # 语义伪标签统计：在有真实语义标签的变化区域上评估
    sem_correct = 0
    sem_total = 0
    sem_high_correct = 0
    sem_high_total = 0

    sem_inter = np.zeros(num_classes, dtype=np.float64)
    sem_union = np.zeros(num_classes, dtype=np.float64)
    sem_high_inter = np.zeros(num_classes, dtype=np.float64)
    sem_high_union = np.zeros(num_classes, dtype=np.float64)

    # 未变化区域伪标签统计：这里没有真实语义标签，只统计覆盖率和置信度
    unchanged_total = 0
    unchanged_valid = 0
    unchanged_conf_sum = 0.0
    unchanged_conf_num = 0

    pseudo_class_count = np.zeros(num_classes, dtype=np.float64)

    start = time.time()
    global_sample_idx = 0

    with torch.no_grad():
        for vi, data in enumerate(val_loader):
            imgs_A, imgs_B, labels_A, labels_B = data

            raw_imgs_A = imgs_A.clone()
            raw_imgs_B = imgs_B.clone()

            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()
            labels_A = labels_A.to(device).long()
            labels_B = labels_B.to(device).long()

            imgs_A_scaled, imgs_B_scaled = scale.scale_input((imgs_A, imgs_B))
            outs = net(imgs_A_scaled, imgs_B_scaled)
            outs = scale.scale_output(outs)

            outputs_A, outputs_B, out_change = outs

            # -------------------------
            # 1. 变化伪标签质量
            # -------------------------
            change_prob = torch.sigmoid(out_change).squeeze(1)
            pseudo_change = change_prob >= opt.change_threshold
            pred_A_full = torch.argmax(outputs_A, dim=1)
            pred_B_full = torch.argmax(outputs_B, dim=1)
            scd_pred_A = pred_A_full * pseudo_change.long()
            scd_pred_B = pred_B_full * pseudo_change.long()

            # Landsat / SECOND 这类 SCD 标签通常 0 表示未变化，>0 表示变化语义类别
            gt_change = (labels_A > 0) | (labels_B > 0)

            change_tp += ((pseudo_change == 1) & (gt_change == 1)).sum().item()
            change_fp += ((pseudo_change == 1) & (gt_change == 0)).sum().item()
            change_fn += ((pseudo_change == 0) & (gt_change == 1)).sum().item()
            change_tn += ((pseudo_change == 0) & (gt_change == 0)).sum().item()

            # -------------------------
            # 2. 语义伪标签质量
            # -------------------------
            # 这里沿用你训练伪标签的逻辑：去掉 0 类，只在 1~C-1 中选最大概率
            probs_A = F.softmax(outputs_A, dim=1)[:, 1:]
            probs_B = F.softmax(outputs_B, dim=1)[:, 1:]

            conf_A, pred_A_0 = torch.max(probs_A, dim=1)
            conf_B, pred_B_0 = torch.max(probs_B, dim=1)

            pseudo_A = pred_A_0 + 1
            pseudo_B = pred_B_0 + 1

            valid_A = labels_A > 0
            valid_B = labels_B > 0

            sem_correct += ((pseudo_A == labels_A) & valid_A).sum().item()
            sem_correct += ((pseudo_B == labels_B) & valid_B).sum().item()
            sem_total += valid_A.sum().item()
            sem_total += valid_B.sum().item()

            high_A = valid_A & (conf_A >= opt.pseudo_threshold)
            high_B = valid_B & (conf_B >= opt.pseudo_threshold)

            sem_high_correct += ((pseudo_A == labels_A) & high_A).sum().item()
            sem_high_correct += ((pseudo_B == labels_B) & high_B).sum().item()
            sem_high_total += high_A.sum().item()
            sem_high_total += high_B.sum().item()

            pseudo_A_np = pseudo_A.detach().cpu().numpy()
            pseudo_B_np = pseudo_B.detach().cpu().numpy()
            labels_A_np = labels_A.detach().cpu().numpy()
            labels_B_np = labels_B.detach().cpu().numpy()
            valid_A_np = valid_A.detach().cpu().numpy()
            valid_B_np = valid_B.detach().cpu().numpy()
            high_A_np = high_A.detach().cpu().numpy()
            high_B_np = high_B.detach().cpu().numpy()

            for b in range(labels_A.shape[0]):
                update_semantic_iou(
                    pseudo_A_np[b],
                    labels_A_np[b],
                    valid_A_np[b],
                    num_classes,
                    sem_inter,
                    sem_union
                )
                update_semantic_iou(
                    pseudo_B_np[b],
                    labels_B_np[b],
                    valid_B_np[b],
                    num_classes,
                    sem_inter,
                    sem_union
                )

                update_semantic_iou(
                    pseudo_A_np[b],
                    labels_A_np[b],
                    high_A_np[b],
                    num_classes,
                    sem_high_inter,
                    sem_high_union
                )
                update_semantic_iou(
                    pseudo_B_np[b],
                    labels_B_np[b],
                    high_B_np[b],
                    num_classes,
                    sem_high_inter,
                    sem_high_union
                )

            # -------------------------
            # 3. 未变化区域伪标签覆盖率
            # -------------------------
            avg_probs = (probs_A + probs_B) * 0.5
            pseudo_conf, pseudo_sem_0 = torch.max(avg_probs, dim=1)
            pseudo_sem = pseudo_sem_0 + 1

            unchanged_mask = (labels_A == 0) & (labels_B == 0)
            valid_unchange_pseudo = unchanged_mask & (pseudo_conf >= opt.pseudo_threshold)

            unchanged_total += unchanged_mask.sum().item()
            unchanged_valid += valid_unchange_pseudo.sum().item()

            if unchanged_mask.sum().item() > 0:
                unchanged_conf_sum += pseudo_conf[unchanged_mask].sum().item()
                unchanged_conf_num += unchanged_mask.sum().item()

            pseudo_sem_valid_np = pseudo_sem.detach().cpu().numpy()
            valid_unchange_np = valid_unchange_pseudo.detach().cpu().numpy()

            for c in range(1, num_classes):
                pseudo_class_count[c] += ((pseudo_sem_valid_np == c) & valid_unchange_np).sum()

            # -------------------------
            # 4. 保存可视化
            # -------------------------
            if global_sample_idx < opt.max_vis:
                batch_size = labels_A.shape[0]
                for b in range(batch_size):
                    if global_sample_idx >= opt.max_vis:
                        break

                    gt_change_np = gt_change[b].detach().cpu().numpy().astype(np.uint8)
                    pseudo_change_np = pseudo_change[b].detach().cpu().numpy().astype(np.uint8)
                    label_A_np = labels_A[b].detach().cpu().numpy().astype(np.int64)
                    label_B_np = labels_B[b].detach().cpu().numpy().astype(np.int64)
                    scd_pred_A_np = scd_pred_A[b].detach().cpu().numpy().astype(np.int64)
                    scd_pred_B_np = scd_pred_B[b].detach().cpu().numpy().astype(np.int64)
                    pseudo_conf_np = pseudo_conf[b].detach().cpu().numpy()

                    # 只显示高置信未变化区域的语义伪标签，其它地方置为 0
                    pseudo_sem_show = pseudo_sem[b].detach().cpu().numpy().astype(np.int64)
                    valid_show = valid_unchange_pseudo[b].detach().cpu().numpy().astype(bool)
                    pseudo_sem_show = pseudo_sem_show * valid_show.astype(np.int64)

                    idx_name = "sample_{:04d}".format(global_sample_idx)

                    save_visualization(
                        save_dir=save_dir,
                        idx_name=idx_name,
                        img_A=raw_imgs_A[b],
                        img_B=raw_imgs_B[b],
                        gt_A=label_A_np,
                        gt_B=label_B_np,
                        gt_change=gt_change_np,
                        pseudo_change=pseudo_change_np,
                        scd_pred_A=scd_pred_A_np,
                        scd_pred_B=scd_pred_B_np,
                        pseudo_semantic_valid=pseudo_sem_show,
                        pseudo_conf=pseudo_conf_np
                    )

                    global_sample_idx += 1

            if (vi + 1) % opt.print_freq == 0:
                precision, recall, f1, iou, acc = compute_binary_metrics(
                    change_tp, change_fp, change_fn, change_tn
                )
                print(
                    "[{}/{}] change_f1={:.4f}, change_iou={:.4f}, change_acc={:.4f}, "
                    "sem_acc={:.4f}, unchanged_coverage={:.4f}".format(
                        vi + 1,
                        len(val_loader),
                        f1,
                        iou,
                        acc,
                        sem_correct / max(sem_total, 1),
                        unchanged_valid / max(unchanged_total, 1)
                    )
                )

    # -------------------------
    # 汇总指标
    # -------------------------
    precision, recall, change_f1, change_iou, change_acc = compute_binary_metrics(
        change_tp, change_fp, change_fn, change_tn
    )

    sem_acc = sem_correct / max(sem_total, 1)
    sem_high_acc = sem_high_correct / max(sem_high_total, 1)

    valid_iou_classes = sem_union[1:] > 0
    if valid_iou_classes.sum() > 0:
        sem_miou = np.mean(sem_inter[1:][valid_iou_classes] / (sem_union[1:][valid_iou_classes] + 1e-7))
    else:
        sem_miou = 0.0

    valid_high_iou_classes = sem_high_union[1:] > 0
    if valid_high_iou_classes.sum() > 0:
        sem_high_miou = np.mean(
            sem_high_inter[1:][valid_high_iou_classes] / (sem_high_union[1:][valid_high_iou_classes] + 1e-7)
        )
    else:
        sem_high_miou = 0.0

    unchanged_coverage = unchanged_valid / max(unchanged_total, 1)
    unchanged_conf_mean = unchanged_conf_sum / max(unchanged_conf_num, 1)

    total_valid_pseudo = pseudo_class_count.sum()
    class_ratio = pseudo_class_count / max(total_valid_pseudo, 1)

    print("\n================ Pseudo Label Quality Summary ================")
    print("Checkpoint:", opt.pth_path)
    print("Dataset:", DATA_NAME)
    print("Num classes:", num_classes)
    print("Input size:", opt.input_size)
    print("Pseudo threshold:", opt.pseudo_threshold)
    print("Change threshold:", opt.change_threshold)

    print("\n[1] Change pseudo quality")
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("F1:        {:.4f}".format(change_f1))
    print("IoU:       {:.4f}".format(change_iou))
    print("Acc:       {:.4f}".format(change_acc))

    print("\n[2] Semantic pseudo quality on labeled changed pixels")
    print("Semantic Acc:       {:.4f}".format(sem_acc))
    print("Semantic mIoU:      {:.4f}".format(sem_miou))
    print("High-conf Sem Acc:  {:.4f}".format(sem_high_acc))
    print("High-conf Sem mIoU: {:.4f}".format(sem_high_miou))
    print("High-conf Sem Pixels:", int(sem_high_total))

    print("\n[3] Pseudo labels on unchanged / unlabeled regions")
    print("Unchanged total pixels:", int(unchanged_total))
    print("High-conf pseudo pixels:", int(unchanged_valid))
    print("Coverage@threshold: {:.4f}".format(unchanged_coverage))
    print("Mean confidence on unchanged regions: {:.4f}".format(unchanged_conf_mean))

    print("\n[4] High-conf pseudo semantic class distribution")
    for c in range(1, num_classes):
        print("Class {:02d}: count={}, ratio={:.4f}".format(
            c,
            int(pseudo_class_count[c]),
            class_ratio[c]
        ))

    print("\nVisualization saved to:")
    print(save_dir)
    print("Total time: {:.1f}s".format(time.time() - start))


def main(opt):
    if torch.cuda.is_available():
        cuda_id = int(str(opt.cuda).split(",")[0])
        torch.cuda.set_device(cuda_id)
        device = torch.device("cuda:{}".format(cuda_id))
        print("Using CUDA device:", cuda_id)
    else:
        device = torch.device("cpu")
        print("Using CPU")

    opt.num_classes = RS.num_classes

    scale = ScaleInOutput(opt.input_size)

    net = ChangeDetection(opt).to(device)
    load_checkpoint(
        net, opt.pth_path, device, opt.checkpoint_model, DATA_NAME,
        RS.num_classes)
    net.eval()

    val_set = RS.Data("val")
    val_loader = DataLoader(
        val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    inspect_pseudo_quality(val_loader, net, scale, device, opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inspect pseudo label quality")

    parser.add_argument(
        "--dataset", choices=("landsat", "second"),
        default=_selected_dataset.dataset)
    parser.add_argument(
        "--checkpoint-model", choices=("teacher", "student"),
        default="teacher",
        help="Model state to load from a structured training checkpoint.")
    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str, default="")

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)

    parser.add_argument("--pseudo-threshold", type=float, default=0.8)
    parser.add_argument("--change-threshold", type=float, default=0.5)
    parser.add_argument("--max-vis", type=int, default=30)
    parser.add_argument("--print-freq", type=int, default=50)

    parser.add_argument(
        "--pth_path",
        type=str,
        required=True,
        help="Checkpoint path; must match the selected dataset."
    )

    opt = parser.parse_args()
    main(opt)