import os
import time
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from skimage import io
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from utils.common import ScaleInOutput, CosOneCycle
from utils.pseudo_label import (
    generate_entropy_pseudo_label, accumulate_class_precision_stats,
    update_class_precision_from_stats, create_pseudo_alpha_diagnostics,
    accumulate_pseudo_alpha_diagnostics, pseudo_alpha_diagnostic_rows)
from utils.experiment_state import build_training_state, restore_training_state
from utils.semantic_pseudo import build_semantic_regions, supervised_semantic_loss
from utils.training_performance import (
    IntervalTimer, TrainingMetricAccumulator, add_performance_arguments,
    change_accuracy, configure_cudnn, is_sample_step, loader_worker_options)
from utils.graph_stage1_training import (
    add_graph_stage1_arguments, build_graph_stage1_runtime, build_sam_cache_config)
from models.graph.pseudo_fusion import (
    graph_pixel_pseudo_loss, reproject_stage2_batch,
    teacher_graph_veto, teacher_graph_complement, ComplementResult)
from datasets.RS_ST_graph import SamGraphDataset, collate_sam_graph_batch


# Data and model choose
_dataset_parser = argparse.ArgumentParser(add_help=False)
_dataset_parser.add_argument("--dataset", choices=("landsat", "second"),
                             default=os.environ.get("SCD_DATASET", "landsat"))
_selected_dataset, _ = _dataset_parser.parse_known_args()
os.environ["SCD_DATASET"] = _selected_dataset.dataset
###############################################
from configs.datasets import resolve_training_config
from datasets import RS_ST as RS
from main_model import ChangeDetection

NET_NAME = 'SAMNet'

training_config = resolve_training_config(
    os.sys.argv[1:], working_path=working_path, model_num_classes=RS.num_classes)
DATA_NAME = training_config.dataset.name

###############################################
# Options
###############################################
args = {
    'train_batch_size': 2,
    'val_batch_size': 2,
    'lr': 0.1,
    'epochs': 100,
    'gpu': True,
    'psd_TTA': True,
    'lr_decay_power': 1.5,
    'train_crop_size': False,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'pseudo_thred': 0.9,
    'ema_decay': 0.99,
    'lambda_sem_u': 0.5,
    'lambda_sem_sup': 1.0,
    'pred_dir': str(training_config.pred_dir),
    'vis_dir': str(training_config.vis_dir),
    'chkpt_dir': str(training_config.checkpoint_dir),
    'log_dir': str(training_config.log_dir),
    'load_path': str(training_config.load_path)
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['vis_dir']): os.makedirs(args['vis_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])
scale = ScaleInOutput(512)


def _tensor_stats(tensor):
    if not torch.is_tensor(tensor):
        return 'not_tensor'
    with torch.no_grad():
        data = tensor.detach()
        finite = torch.isfinite(data)
        if not finite.any():
            return 'shape={} dtype={} finite=0/{} all_nonfinite'.format(
                tuple(data.shape), data.dtype, data.numel())
        finite_data = data[finite].float()
        return 'shape={} dtype={} finite={}/{} min={:.4g} max={:.4g} mean={:.4g}'.format(
            tuple(data.shape), data.dtype, int(finite.sum().item()), data.numel(),
            float(finite_data.min().item()), float(finite_data.max().item()),
            float(finite_data.mean().item()))


def _flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _first_nonfinite_state(module):
    for name, tensor in module.named_parameters():
        if tensor is not None and not torch.isfinite(tensor).all().item():
            return 'parameter', name, _tensor_stats(tensor)
    for name, tensor in module.named_buffers():
        if tensor is not None and torch.is_floating_point(tensor) and not torch.isfinite(tensor).all().item():
            return 'buffer', name, _tensor_stats(tensor)
    return None, None, None


def _module_is_finite(module):
    """Return False when a model parameter or floating-point buffer is invalid."""
    if module is None:
        return True
    with torch.no_grad():
        for tensor in module.parameters():
            if not torch.isfinite(tensor).all().item():
                return False
        for tensor in module.buffers():
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all().item():
                return False
    return True


def _repair_nonfinite_state(module, reference=None):
    """Repair invalid state from a finite reference, falling back to zero."""
    if module is None:
        return []
    reference_state = reference.state_dict() if reference is not None else {}
    repaired = []
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            if not torch.is_floating_point(tensor):
                continue
            if torch.isfinite(tensor).all().item():
                continue
            ref = reference_state.get(name)
            if (ref is not None and ref.shape == tensor.shape
                    and torch.isfinite(ref).all().item()):
                tensor.copy_(ref.to(device=tensor.device, dtype=tensor.dtype))
            else:
                tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            repaired.append(name)
    return repaired


def _repair_nonfinite_optimizer_state(optimizer):
    """Reset invalid Adam/AdamW state tensors before they poison future steps."""
    repaired = 0
    if optimizer is None:
        return repaired
    with torch.no_grad():
        for state in optimizer.state.values():
            for value in state.values():
                if (torch.is_tensor(value) and torch.is_floating_point(value)
                        and not torch.isfinite(value).all().item()):
                    value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    repaired += 1
    return repaired


def _attach_nonfinite_hooks(module):
    state = {'triggered': False}
    handles = []

    def hook(name):
        def _hook(mod, inputs, output):
            if state['triggered']:
                return
            tensors = _flatten_tensors(output)
            if tensors and any(not torch.isfinite(t).all().item() for t in tensors):
                state['triggered'] = True
                input_stats = [_tensor_stats(t) for t in _flatten_tensors(inputs)[:3]]
                output_stats = [_tensor_stats(t) for t in tensors[:3]]
                raise FloatingPointError(
                    'First non-finite module output: {} ({}) inputs={} outputs={}'.format(
                        name, mod.__class__.__name__, input_stats, output_stats))
        return _hook

    for name, child in module.named_modules():
        if name:
            handles.append(child.register_forward_hook(hook(name)))
    return handles


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(net, opt):
    if opt.finetune:
        params = [
            {"params": [param for name, param in net.named_parameters()
                        if param.requires_grad and "backbone" in name],
             "lr": opt.learning_rate / 10},
            {"params": [param for name, param in net.named_parameters()
                        if param.requires_grad and "backbone" not in name],
             "lr": opt.learning_rate},
        ]
        print("Using finetune for model")
    else:
        params = [param for param in net.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.001)


def update_ema_teacher(student, teacher, ema_decay, global_step):
    """Update teacher with the exponential moving average of the student."""
    ema_decay = min(1.0 - 1.0 / float(global_step + 1), ema_decay)
    with torch.no_grad():
        student_state = student.state_dict()
        teacher_state = teacher.state_dict()
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key].detach()
            if teacher_value.dtype.is_floating_point:
                # A skipped/non-finite student update must never poison EMA.
                if not torch.isfinite(student_value).all().item():
                    continue
                if not torch.isfinite(teacher_value).all().item():
                    teacher_value.copy_(student_value)
                    continue
                teacher_value.mul_(ema_decay).add_(student_value, alpha=1.0 - ema_decay)
            else:
                teacher_value.copy_(student_value)


def get_lambda_sem_u(opt, curr_epoch):
    if (not opt.use_semantic_pseudo) or curr_epoch < opt.pseudo_start_epoch:
        return 0.0
    ramp = min(1.0, float(curr_epoch - opt.pseudo_start_epoch + 1) / float(max(opt.pseudo_rampup_epoch, 1)))
    return opt.lambda_sem_u * ramp


def get_graph_pixel_pseudo_weight(opt, curr_epoch):
    # Unified flow: GAT runs from epoch 0, weight ramps up from pseudo_start_epoch.
    # --graph-stage2-start-epoch is no longer used (kept for backward compat).
    if not opt.use_sam_graph_stage1:
        return 0.0
    start = max(int(opt.pseudo_start_epoch), 0)
    if curr_epoch < start:
        return 0.0
    ramp_epochs = max(int(opt.graph_pseudo_ramp_epochs), 1)
    progress = min(1.0, float(curr_epoch - start) / float(ramp_epochs))
    return float(opt.graph_pixel_pseudo_weight) * progress


def get_gamma_per_class(opt, curr_epoch, num_classes, class_precision, device):
    if curr_epoch < opt.pseudo_start_epoch:
        base_ratio = opt.pseudo_alpha_start
    else:
        denom = max(opt.epochs - opt.pseudo_start_epoch, 1)
        progress = min(1.0, float(curr_epoch - opt.pseudo_start_epoch) / float(denom))
        base_ratio = opt.pseudo_alpha_start + (opt.pseudo_alpha_end - opt.pseudo_alpha_start) * progress

    gamma = torch.ones(num_classes, device=device) * base_ratio
    gamma = gamma * class_precision.to(device=device)
    gamma = gamma.clamp(min=opt.pseudo_min_ratio, max=opt.pseudo_alpha_end)
    gamma[0] = 0.0
    return gamma


def semantic_pseudo_loss_entropy(
    outputs_A,
    outputs_B,
    teacher_outputs_A,
    teacher_outputs_B,
    teacher_change,
    labels_A,
    labels_B,
    gamma_per_class,
    opt,
):
    num_classes = outputs_A.size(1)
    gt_unchanged_mask = (labels_A == 0) & (labels_B == 0)
    teacher_change_prob = torch.sigmoid(teacher_change).squeeze(1)
    teacher_unchanged_mask = teacher_change_prob < opt.pseudo_change_thr
    candidate_mask = gt_unchanged_mask & teacher_unchanged_mask

    pseudo_A, reliable_A, entropy_A = generate_entropy_pseudo_label(
        teacher_outputs_A, candidate_mask, num_classes, gamma_per_class, opt.pseudo_ignore_index)
    pseudo_B, reliable_B, entropy_B = generate_entropy_pseudo_label(
        teacher_outputs_B, candidate_mask, num_classes, gamma_per_class, opt.pseudo_ignore_index)

    loss_terms = []
    if reliable_A.any():
        loss_terms.append(F.cross_entropy(outputs_A, pseudo_A, ignore_index=opt.pseudo_ignore_index))
    if reliable_B.any():
        loss_terms.append(F.cross_entropy(outputs_B, pseudo_B, ignore_index=opt.pseudo_ignore_index))
    if not loss_terms:
        zero_loss = outputs_A.sum() * 0.0
        return zero_loss, zero_loss.detach(), pseudo_A, pseudo_B, reliable_A | reliable_B, entropy_A, entropy_B

    loss = sum(loss_terms) / len(loss_terms)
    reliable_ratio = ((reliable_A | reliable_B).float().mean()).detach()
    return loss, reliable_ratio, pseudo_A, pseudo_B, reliable_A | reliable_B, entropy_A, entropy_B


def image_tensor_to_uint8(img, time_key):
    img = img.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = RS.tensor2int(img, time_key)
    return np.clip(img, 0, 255).astype(np.uint8)


def save_pseudo_visualization(imgs_A, imgs_B, labels_A, labels_B, pseudo_A, pseudo_B,
                              valid_mask, curr_epoch, iteration):
    if pseudo_A is None or pseudo_B is None or valid_mask is None:
        return

    with torch.no_grad():
        unchanged_mask = ((labels_A == 0) & (labels_B == 0))[0].detach().cpu().numpy().astype(np.uint8) * 255
        valid = valid_mask[0].detach().cpu().bool()
        pseudo_A_vis = pseudo_A[0].detach().cpu().long()
        pseudo_B_vis = pseudo_B[0].detach().cpu().long()
        pseudo_A_vis[(~valid) | (pseudo_A_vis < 0) | (pseudo_A_vis >= RS.num_classes)] = 0
        pseudo_B_vis[(~valid) | (pseudo_B_vis < 0) | (pseudo_B_vis >= RS.num_classes)] = 0

        img_A = image_tensor_to_uint8(imgs_A[0], 'A')
        img_B = image_tensor_to_uint8(imgs_B[0], 'B')
        unchanged_rgb = np.repeat(unchanged_mask[:, :, None], 3, axis=2)
        valid_rgb = np.repeat((valid.numpy().astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        pseudo_A_color = RS.Index2Color(pseudo_A_vis.numpy())
        pseudo_B_color = RS.Index2Color(pseudo_B_vis.numpy())
        canvas = np.concatenate([img_A, img_B, unchanged_rgb, pseudo_A_color, pseudo_B_color, valid_rgb], axis=1)

    out_name = 'epoch%03d_iter%05d.png' % (curr_epoch, iteration)
    io.imsave(os.path.join(args['vis_dir'], out_name), canvas)


def _run_mode(opt):
    if opt.use_sam_graph_stage2:
        suffix = "_graph_stage2"
    else:
        suffix = "_graph_stage1" if opt.use_sam_graph_stage1 else ""
    if opt.use_semantic_pseudo:
        return "pseudo" + suffix
    if opt.use_ema_teacher:
        return "ema" + suffix
    return "supervised" + suffix


def _run_config(opt, dataset_name):
    return {
        "dataset": dataset_name, "num_classes": opt.num_classes,
        "seed": opt.seed, "mode": _run_mode(opt),
        "use_ema_teacher": opt.use_ema_teacher, "use_semantic_pseudo": opt.use_semantic_pseudo,
        "ema_decay": opt.ema_decay, "pseudo_start_epoch": opt.pseudo_start_epoch,
        "pseudo_alpha_start": opt.pseudo_alpha_start, "pseudo_alpha_end": opt.pseudo_alpha_end,
        "pseudo_min_ratio": opt.pseudo_min_ratio, "pseudo_ignore_index": opt.pseudo_ignore_index,
        "pseudo_change_thr": opt.pseudo_change_thr, "lambda_sem_u": opt.lambda_sem_u,
        "lambda_sem_sup": opt.lambda_sem_sup, "pseudo_rampup_epoch": opt.pseudo_rampup_epoch,
        "pseudo_precision_init": opt.pseudo_precision_init,
        "pseudo_precision_momentum": opt.pseudo_precision_momentum,
        "use_sam_graph_stage1": opt.use_sam_graph_stage1,
        "lambda_graph_gt": opt.lambda_graph_gt,
        "lambda_edge_gt": opt.lambda_edge_gt,
        "graph_node_dim": opt.graph_node_dim,
        "use_sam_graph_stage2": opt.use_sam_graph_stage2,
        "graph_stage2_start_epoch": opt.graph_stage2_start_epoch,
        "graph_local_topk": opt.graph_local_topk,
        "graph_edge_threshold": opt.graph_edge_threshold,
        "graph_temp_weight": opt.graph_temp_weight,
        "graph_pixel_pseudo_weight": opt.graph_pixel_pseudo_weight,
        "graph_pseudo_ramp_epochs": opt.graph_pseudo_ramp_epochs,
    }

def main(opt):
    training_config = resolve_training_config(
        ["--dataset", opt.dataset, "--num-classes", str(opt.num_classes)],
        working_path=working_path,
        model_num_classes=opt.num_classes)
    global scale
    seed_everything(opt.seed)
    cudnn_state = configure_cudnn(opt.deterministic)
    if torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        if opt.cuda < 0 or opt.cuda >= visible_device_count:
            raise ValueError(
                "Invalid logical CUDA device {}: this process sees {} device(s). "
                "When CUDA_VISIBLE_DEVICES selects one physical GPU, use --cuda 0.".format(
                    opt.cuda, visible_device_count
                )
            )
        torch.cuda.set_device(int(opt.cuda))
    scale = ScaleInOutput(opt.input_size)
    args['epochs'] = opt.epochs
    args['gpu'] = torch.cuda.is_available()
    net = ChangeDetection(opt).cuda()
    net_psd = None
    if opt.use_ema_teacher:
        net_psd = copy.deepcopy(net)
        net_psd.eval()
        freeze_model(net_psd)

    graph_runtime = None
    if opt.use_sam_graph_stage1:
        base_train_set = RS.Data('train', random_flip=False)
        train_set = SamGraphDataset(
            base_train_set,
            dataset_name=training_config.dataset.name,
            split="train",
            cache_root=opt.sam_cache_root,
            cache_config=build_sam_cache_config(opt),
            random_flip=True,
        )
        collate_fn = collate_sam_graph_batch
        graph_runtime = build_graph_stage1_runtime(
            net.backbone, RS.num_classes, opt)
    else:
        train_set = RS.Data('train', random_flip=True)
        collate_fn = None
    generator = torch.Generator()
    generator.manual_seed(opt.seed)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=args['gpu'],
                              drop_last=False, generator=generator,
                              collate_fn=collate_fn,
                              **loader_worker_options(
                                  opt.num_workers, opt.persistent_workers))
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=args['gpu'],
                            **loader_worker_options(
                                opt.num_workers, opt.persistent_workers))

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = build_optimizer(net, opt)
    scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs)
    resume_state = None
    if opt.resume:
        if not os.path.isfile(opt.resume):
            raise FileNotFoundError("Resume checkpoint not found: {}".format(opt.resume))
        try:
            checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(opt.resume, map_location="cpu")
        resume_state = restore_training_state(
            checkpoint, net, net_psd, optimizer, scheduler,
            active_run_config={"dataset": training_config.dataset.name,
                               "num_classes": training_config.model_num_classes})
        print("Resumed complete training state from {} at epoch {} optimizer_step {}".format(
            opt.resume, resume_state.epoch, resume_state.optimizer_step))
        if graph_runtime is not None:
            restored_graph = graph_runtime.load_checkpoint_payload(checkpoint)
            print("SAM graph Stage1 state {} from checkpoint.".format(
                "restored" if restored_graph else "not found; initialized fresh"))
            if (restored_graph and opt.use_sam_graph_stage2
                    and graph_runtime.loaded_format_version == 1):
                print("Stage2 propagation state initialized fresh")
    else:
        for _ in range(opt.start_epoch):
            scheduler.step()

    train(train_loader, net, net_psd, criterion, optimizer, scheduler, val_loader,
          resume_state, _run_config(opt, training_config.dataset.name), graph_runtime)
    writer.close()
    print('Training finished.')

def train(train_loader, net, net_psd, criterion, optimizer, scheduler, val_loader,
          resume_state, run_config, graph_runtime=None):
    bestaccT = 0
    bestFscdV = 0.0
    bestaccV = 0.0
    bestloss = 1.0
    begin_time = time.time()
    criterion_sc = ChangeSimilarity().cuda()
    device = next(net.parameters()).device
    class_precision = torch.ones(RS.num_classes, device=device) * opt.pseudo_precision_init
    class_precision[0] = 0.0
    optimizer_step = 0
    teacher_nonfinite_skips = 0
    if resume_state is not None:
        class_precision = resume_state.class_precision.to(device=device)
        optimizer_step = resume_state.optimizer_step
        bestaccT = resume_state.best_metrics.get("train_acc", bestaccT)
        bestFscdV = resume_state.best_metrics.get("Fscd", bestFscdV)
        bestaccV = resume_state.best_metrics.get("accuracy", bestaccV)
        bestloss = resume_state.best_metrics.get("loss", bestloss)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    debug_handles = _attach_nonfinite_hooks(net) if opt.debug_nonfinite else []
    if opt.debug_nonfinite:
        print("Non-finite forward hooks enabled.")

    accumulate_iter = opt.effective_batch_size // opt.batch_size
    print("Accumulate_iter={} batch_size={} effective_batch_size={}".format(
        accumulate_iter, opt.batch_size, opt.batch_size * accumulate_iter))
    print("EMA teacher={} semantic_pseudo={} pseudo_start_epoch={} lambda_sem_u={}".format(
        opt.use_ema_teacher, opt.use_semantic_pseudo, opt.pseudo_start_epoch, opt.lambda_sem_u))
    print("AMP={} gradient_checkpointing={} cuda={} ({})".format(
        opt.amp, opt.gradient_checkpointing, opt.cuda, torch.cuda.get_device_name(device)))
    print("cuDNN benchmark={} deterministic={} persistent_workers={} timing_interval={}".format(
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        opt.persistent_workers, opt.timing_interval))

    curr_epoch = resume_state.epoch if resume_state is not None else opt.start_epoch
    optimizer.zero_grad(set_to_none=True)
    _repair_nonfinite_optimizer_state(optimizer)
    if graph_runtime is not None:
        graph_runtime.zero_grad(set_to_none=True)
        _repair_nonfinite_optimizer_state(graph_runtime.optimizer)
    while True:
        #torch.cuda.empty_cache()
        net.train()
        if graph_runtime is not None:
            graph_runtime.train()

        start = time.time()
        train_metrics = TrainingMetricAccumulator(
            (
                "sem_sup", "change", "bn", "sc", "pseudo", "pseudo_ratio",
                "lambda_sem_u", "graph_gt", "edge_gt", "gt_nodes", "accuracy",
                "graph_stage2_gt", "graph_temp", "graph_pixel_pseudo",
                "graph_pseudo_weight", "graph_nodes", "graph_edges",
                "graph_valid_ratio", "teacher_candidate_ratio",
                "graph_veto_ratio", "graph_fallback_ratio", "graph_accepted_ratio",
                "u_nodes", "ignore_nodes", "graph_confidence",
                "edge_reliability", "propagation_edges", "spatial_edges",
                "temporal_edges", "graph_gt_accuracy", "student_grad_norm",
                "graph_grad_norm",
            ),
            device=device,
        )
        epoch_precision_tp = torch.zeros(RS.num_classes, device=device)
        epoch_precision_fp = torch.zeros(RS.num_classes, device=device)
        pseudo_alpha_diagnostics = None
        if (opt.use_ema_teacher and opt.use_semantic_pseudo
                and curr_epoch >= opt.pseudo_start_epoch
                and opt.pseudo_diagnostic_interval > 0):
            pseudo_alpha_diagnostics = create_pseudo_alpha_diagnostics(
                RS.num_classes, device=device)

        curr_iter = curr_epoch * len(train_loader)
        throughput_timer = IntervalTimer(opt.timing_interval)
        throughput_timer.start_data()
        for i, data in enumerate(train_loader):
            throughput_timer.start_compute()
            running_iter = curr_iter + i + 1
            # Unified flow: Stage2 GAT always active when graph is loaded.
            # Weight ramps up from pseudo_start_epoch via get_graph_pixel_pseudo_weight.
            stage2_active = graph_runtime is not None
            if graph_runtime is None:
                imgs_A, imgs_B, labels_A, labels_B = data
                graph_data = None
            else:
                imgs_A, imgs_B, labels_A, labels_B, graph_data = data
            audit_labels_A, audit_labels_B = labels_A, labels_B
            if args['gpu']:
                imgs_A = imgs_A.cuda(non_blocking=True).float()
                imgs_B = imgs_B.cuda(non_blocking=True).float()
                labels_A = labels_A.cuda(non_blocking=True).long()
                labels_B = labels_B.cuda(non_blocking=True).long()
            changed_mask, _ = build_semantic_regions(labels_A, labels_B)
            labels_bn = changed_mask.unsqueeze(1).float()
            vis_imgs_A, vis_imgs_B = imgs_A.detach(), imgs_B.detach()

            imgs_A, imgs_B = scale.scale_input((imgs_A, imgs_B))

            graph_backbone_features = None
            masks_list = (
                [item["shared_masks"] for item in graph_data]
                if graph_data is not None else None
            )
            with torch.cuda.amp.autocast(enabled=opt.amp):
                if graph_runtime is None:
                    outs = net(imgs_A, imgs_B)
                else:
                    outs, graph_backbone_features = net(
                        imgs_A, imgs_B, return_graph_features=True,
                        detach_graph_features=not stage2_active,
                        masks_list=masks_list)
                outs = scale.scale_output(outs)
                outputs_A, outputs_B, out_change = outs

                assert outputs_A.size()[1] == RS.num_classes

                student_forward_tensors = [outputs_A, outputs_B, out_change]
                if graph_backbone_features is not None:
                    student_forward_tensors.extend(graph_backbone_features)
                if any(not torch.isfinite(tensor).all().item()
                       for tensor in student_forward_tensors):
                    repaired = _repair_nonfinite_state(net, net_psd)
                    optimizer.zero_grad(set_to_none=True)
                    _repair_nonfinite_optimizer_state(optimizer)
                    if graph_runtime is not None:
                        graph_runtime.zero_grad(set_to_none=True)
                        _repair_nonfinite_optimizer_state(graph_runtime.optimizer)
                        graph_runtime.stage2_counters["nonfinite_skips"] += 1
                    print("[Warning] non-finite student forward at epoch {} iter {}; "
                          "batch skipped; repaired_state={}".format(
                              curr_epoch, i + 1, repaired))
                    throughput_timer.finish_step(
                        i + 1, torch.cuda.synchronize if args['gpu'] else None)
                    if i + 1 < len(train_loader):
                        throughput_timer.start_data()
                    continue

                loss_sem_sup = supervised_semantic_loss(outputs_A, outputs_B, labels_A, labels_B)
                loss_bn = weighted_BCE_logits(out_change, labels_bn)
                loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
                loss_change = loss_bn + loss_sc

            pseudo_A = None
            pseudo_B = None
            valid_mask = None
            loss_pseudo = outputs_A.sum() * 0.0
            pseudo_ratio = loss_pseudo.detach()
            lambda_sem_u_current = get_lambda_sem_u(opt, curr_epoch)
            semantic_pseudo_active = bool(
                opt.use_ema_teacher and opt.use_semantic_pseudo
                and curr_epoch >= opt.pseudo_start_epoch)
            teacher_outputs_A = None
            teacher_outputs_B = None
            teacher_change = None

            teacher_ready = False
            if semantic_pseudo_active or stage2_active:
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=opt.amp):
                    teacher_outs = net_psd(imgs_A, imgs_B)
                    teacher_outs = scale.scale_output(teacher_outs)
                    teacher_outputs_A, teacher_outputs_B, teacher_change = teacher_outs
                teacher_tensors = [
                    teacher_outputs_A, teacher_outputs_B, teacher_change]
                teacher_ready = all(
                    torch.isfinite(tensor).all().item()
                    for tensor in teacher_tensors)
                if not teacher_ready:
                    teacher_nonfinite_skips += 1
                    repaired = _repair_nonfinite_state(net_psd, net)
                    teacher_outputs_A = None
                    teacher_outputs_B = None
                    teacher_change = None
                    print("[Warning] non-finite EMA teacher output at epoch {} iter {}; "
                          "pseudo/graph-veto skipped; repaired_state={}".format(
                              curr_epoch, i + 1, repaired))

                if semantic_pseudo_active and teacher_ready:
                    with torch.no_grad():
                        valid_A = labels_A > 0
                        valid_B = labels_B > 0
                        if valid_A.any():
                            epoch_precision_tp, epoch_precision_fp = accumulate_class_precision_stats(
                                teacher_outputs_A, labels_A, valid_A, RS.num_classes,
                                epoch_precision_tp, epoch_precision_fp)
                        if valid_B.any():
                            epoch_precision_tp, epoch_precision_fp = accumulate_class_precision_stats(
                                teacher_outputs_B, labels_B, valid_B, RS.num_classes,
                                epoch_precision_tp, epoch_precision_fp)
                        gamma_per_class = get_gamma_per_class(
                            opt, curr_epoch, RS.num_classes, class_precision, device)
                        teacher_unchanged = (
                            torch.sigmoid(teacher_change).squeeze(1)
                            < opt.pseudo_change_thr)
                        unlabeled_candidate = (
                            (labels_A == 0) & (labels_B == 0) & teacher_unchanged)
                        if is_sample_step(i + 1, opt.pseudo_diagnostic_interval):
                            accumulate_pseudo_alpha_diagnostics(
                                pseudo_alpha_diagnostics, teacher_outputs_A, labels_A,
                                valid_A, unlabeled_candidate)
                            accumulate_pseudo_alpha_diagnostics(
                                pseudo_alpha_diagnostics, teacher_outputs_B, labels_B,
                                valid_B, unlabeled_candidate)

                    with torch.cuda.amp.autocast(enabled=opt.amp):
                        loss_pseudo, pseudo_ratio, pseudo_A, pseudo_B, valid_mask, _, _ = semantic_pseudo_loss_entropy(
                            outputs_A, outputs_B, teacher_outputs_A, teacher_outputs_B, teacher_change,
                            labels_A, labels_B, gamma_per_class, opt)

                    if is_sample_step(i + 1, opt.pseudo_visualization_interval):
                        save_pseudo_visualization(
                            vis_imgs_A, vis_imgs_B, labels_A, labels_B,
                            pseudo_A, pseudo_B, valid_mask, curr_epoch, i + 1)

            with torch.cuda.amp.autocast(enabled=opt.amp):
                loss = opt.lambda_sem_sup * loss_sem_sup + loss_change + lambda_sem_u_current * loss_pseudo

            loss_graph_gt = outputs_A.sum() * 0.0
            loss_edge_gt = outputs_A.sum() * 0.0
            loss_stage2_graph_gt = outputs_A.sum() * 0.0
            loss_graph_temp = outputs_A.sum() * 0.0
            loss_graph_pixel_pseudo = outputs_A.sum() * 0.0
            graph_pseudo_weight = get_graph_pixel_pseudo_weight(opt, curr_epoch)
            graph_veto = None
            graph_pixels = None
            graph_result = None
            if graph_runtime is not None:
                with torch.cuda.amp.autocast(enabled=opt.amp):
                    graph_result = graph_runtime.compute(
                        imgs_A, imgs_B, audit_labels_A, audit_labels_B, graph_data,
                        backbone_features=graph_backbone_features,
                        stage2_enabled=stage2_active)
                loss_graph_gt = graph_result.loss_graph_gt
                loss_edge_gt = graph_result.loss_edge_gt
                loss_stage2_graph_gt = graph_result.loss_stage2_graph_gt
                loss_graph_temp = graph_result.loss_graph_temp
                loss = (
                    loss
                    + opt.lambda_graph_gt * loss_graph_gt
                    + opt.lambda_edge_gt * loss_edge_gt
                    + opt.lambda_graph_gt * loss_stage2_graph_gt
                    + opt.graph_temp_weight * loss_graph_temp
                )
                if stage2_active and teacher_ready and graph_pseudo_weight > 0:
                    graph_pixels = reproject_stage2_batch(
                        graph_result.stage2_result,
                        graph_result.audits,
                        graph_result.masks,
                    )
                    complement = teacher_graph_complement(
                        teacher_probs_a=torch.softmax(teacher_outputs_A.float(), dim=1),
                        teacher_probs_b=torch.softmax(teacher_outputs_B.float(), dim=1),
                        teacher_change_logits=teacher_change,
                        labels_a=labels_A,
                        labels_b=labels_B,
                        graph_pred_a=graph_pixels.pred_a,
                        graph_conf_a=graph_pixels.conf_a,
                        graph_valid_a=graph_pixels.valid_a,
                        graph_pred_b=graph_pixels.pred_b,
                        graph_conf_b=graph_pixels.conf_b,
                        graph_valid_b=graph_pixels.valid_b,
                        graph_conf_threshold=opt.graph_confidence_threshold,
                        change_threshold=opt.pseudo_change_thr,
                        num_classes=RS.num_classes,
                        ignore_index=opt.pseudo_ignore_index,
                    )
                    with torch.cuda.amp.autocast(enabled=opt.amp):
                        loss_graph_pixel_pseudo = graph_pixel_pseudo_loss(
                            outputs_A, outputs_B,
                            complement.pseudo_a, complement.pseudo_a != opt.pseudo_ignore_index,
                            ignore_index=opt.pseudo_ignore_index)
                    loss = loss + graph_pseudo_weight * loss_graph_pixel_pseudo
                    graph_runtime.stage2_counters["accepted_pixels"] += (
                        complement.graph_pixel_count_a + complement.teacher_pixel_count_a)

            loss_components = {
                'loss_sem_sup': loss_sem_sup,
                'loss_bn': loss_bn,
                'loss_sc': loss_sc,
                'loss_sem_u': loss_pseudo,
                'loss_graph_gt': loss_graph_gt,
                'loss_edge_gt': loss_edge_gt,
                'loss_stage2_graph_gt': loss_stage2_graph_gt,
                'loss_graph_temp': loss_graph_temp,
                'loss_graph_pixel_pseudo': loss_graph_pixel_pseudo,
                'loss_total': loss,
            }
            non_finite_losses = [
                name for name, value in loss_components.items()
                if not torch.isfinite(value).all().item()
            ]
            if non_finite_losses:
                non_finite_outputs = [
                    name for name, value in (
                        ('outputs_A', outputs_A),
                        ('outputs_B', outputs_B),
                        ('out_change', out_change),
                    ) if not torch.isfinite(value).all().item()
                ]
                state_kind, state_name, state_stats = _first_nonfinite_state(net)
                print("Non-finite debug: imgs_A={} imgs_B={} labels_A={} labels_B={} first_state={}:{} {}".format(
                    _tensor_stats(imgs_A), _tensor_stats(imgs_B), _tensor_stats(labels_A),
                    _tensor_stats(labels_B), state_kind, state_name, state_stats))
                optimizer.zero_grad(set_to_none=True)
                _repair_nonfinite_optimizer_state(optimizer)
                if graph_runtime is not None:
                    graph_runtime.zero_grad(set_to_none=True)
                    _repair_nonfinite_optimizer_state(graph_runtime.optimizer)
                    graph_runtime.stage2_counters["nonfinite_skips"] += 1
                print('[Warning] non-finite loss at epoch {} iter {}: losses={} outputs={}; '
                      'student/graph optimizer and EMA updates skipped.'.format(
                          curr_epoch, i + 1, non_finite_losses, non_finite_outputs))
                throughput_timer.finish_step(
                    i + 1, torch.cuda.synchronize if args['gpu'] else None)
                if i + 1 < len(train_loader):
                    throughput_timer.start_data()
                continue

            window_start = (i // accumulate_iter) * accumulate_iter
            window_size = min(accumulate_iter, len(train_loader) - window_start)
            scaler.scale(loss / window_size).backward()
            should_step = ((i + 1) % accumulate_iter) == 0 or (i + 1) == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)
                graph_grad_norm = torch.zeros((), device=device)
                if graph_runtime is not None:
                    scaler.unscale_(graph_runtime.optimizer)
                    graph_grad_norm = torch.nn.utils.clip_grad_norm_(
                        graph_runtime.parameters(), opt.graph_grad_clip)
                if not torch.isfinite(grad_norm) or not torch.isfinite(graph_grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    _repair_nonfinite_optimizer_state(optimizer)
                    if graph_runtime is not None:
                        graph_runtime.zero_grad(set_to_none=True)
                        _repair_nonfinite_optimizer_state(graph_runtime.optimizer)
                        graph_runtime.stage2_counters["nonfinite_skips"] += 1
                    scaler.update()
                    print('[Warning] non-finite gradient at epoch {} iter {}; '
                          'optimizer and EMA updates skipped.'.format(curr_epoch, i + 1))
                    throughput_timer.finish_step(
                        i + 1, torch.cuda.synchronize if args['gpu'] else None)
                    if i + 1 < len(train_loader):
                        throughput_timer.start_data()
                    continue
                scaler.step(optimizer)
                if graph_runtime is not None:
                    scaler.step(graph_runtime.optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _repair_nonfinite_optimizer_state(optimizer)
                if graph_runtime is not None:
                    graph_runtime.zero_grad(set_to_none=True)
                    _repair_nonfinite_optimizer_state(graph_runtime.optimizer)
                optimizer_step += 1
                if not _module_is_finite(net):
                    repaired = _repair_nonfinite_state(net, net_psd)
                    print("[Warning] non-finite student state after optimizer step "
                          "{} repaired_state={}".format(
                              optimizer_step, repaired))
                if opt.use_ema_teacher:
                    update_ema_teacher(net, net_psd, opt.ema_decay, optimizer_step)

            stage2_result = (
                graph_result.stage2_result if graph_result is not None else None)
            pixel_count = max(int(labels_A.numel()), 1)
            graph_valid_ratio = (
                0.5 * (graph_pixels.valid_a.float().mean()
                       + graph_pixels.valid_b.float().mean())
                if graph_pixels is not None else 0.0)
            teacher_candidate_ratio = (
                graph_veto.teacher_candidate_count / pixel_count
                if graph_veto is not None else 0.0)
            graph_veto_ratio = (
                graph_veto.veto_count / max(graph_veto.teacher_candidate_count, 1)
                if graph_veto is not None else 0.0)
            graph_fallback_ratio = (
                graph_veto.fallback_count / max(graph_veto.teacher_candidate_count, 1)
                if graph_veto is not None else 0.0)
            graph_accepted_ratio = (
                graph_veto.accepted_count / pixel_count
                if graph_veto is not None else 0.0)
            u_node_count = 0
            ignore_node_count = 0
            if graph_result is not None:
                for audit in graph_result.audits:
                    u_node_count += int((audit.states_a == 2).sum() + (audit.states_b == 2).sum())
                    ignore_node_count += int((audit.states_a == 0).sum() + (audit.states_b == 0).sum())
            graph_confidence = 0.0
            if stage2_result is not None:
                confidences = stage2_result.confidences_a + stage2_result.confidences_b
                valid_confidences = [
                    confidence[valid]
                    for confidence, valid in zip(
                        confidences,
                        stage2_result.valid_a + stage2_result.valid_b)
                    if valid.any()]
                if valid_confidences:
                    graph_confidence = torch.cat(valid_confidences).mean()
            train_metrics.update(
                sem_sup=loss_sem_sup,
                change=loss_change,
                bn=loss_bn,
                sc=loss_sc,
                pseudo=loss_pseudo,
                pseudo_ratio=pseudo_ratio,
                lambda_sem_u=lambda_sem_u_current,
                graph_gt=loss_graph_gt,
                edge_gt=loss_edge_gt,
                gt_nodes=graph_result.gt_node_count if graph_result is not None else 0,
                graph_stage2_gt=loss_stage2_graph_gt,
                graph_temp=loss_graph_temp,
                graph_pixel_pseudo=loss_graph_pixel_pseudo,
                graph_pseudo_weight=graph_pseudo_weight,
                graph_nodes=stage2_result.node_count if stage2_result is not None else 0,
                graph_edges=stage2_result.edge_count if stage2_result is not None else 0,
                graph_valid_ratio=graph_valid_ratio,
                teacher_candidate_ratio=teacher_candidate_ratio,
                graph_veto_ratio=graph_veto_ratio,
                graph_fallback_ratio=graph_fallback_ratio,
                graph_accepted_ratio=graph_accepted_ratio,
                u_nodes=u_node_count,
                ignore_nodes=ignore_node_count,
                graph_confidence=graph_confidence,
                edge_reliability=(
                    stage2_result.mean_edge_reliability if stage2_result is not None else 0),
                propagation_edges=(
                    stage2_result.propagation_edge_count if stage2_result is not None else 0),
                spatial_edges=(
                    stage2_result.spatial_edge_count if stage2_result is not None else 0),
                temporal_edges=(
                    stage2_result.temporal_edge_count if stage2_result is not None else 0),
                graph_gt_accuracy=(
                    stage2_result.graph_gt_correct / max(stage2_result.graph_gt_count, 1)
                    if stage2_result is not None else 0),
                student_grad_norm=grad_norm if should_step else 0,
                graph_grad_norm=graph_grad_norm if should_step else 0,
                accuracy=change_accuracy(
                    outputs_A, outputs_B, out_change, labels_A, labels_B),
            )

            timing_sample = throughput_timer.finish_step(
                i + 1, torch.cuda.synchronize if args['gpu'] else None)
            if timing_sample is not None:
                print('[throughput] iter={} data={:.3f}s compute={:.3f}s step={:.3f}s'.format(
                    i + 1, timing_sample['data_seconds'],
                    timing_sample['compute_seconds'], timing_sample['step_seconds']))
                for timing_name, timing_value in timing_sample.items():
                    writer.add_scalar(
                        'train/throughput/' + timing_name, timing_value, running_iter)
            if i + 1 < len(train_loader):
                throughput_timer.start_data()

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                metric_snapshot = train_metrics.snapshot()
                precision_log = ['%.3f' % v for v in class_precision.detach().cpu().tolist()]
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [loss_sem_sup %.4f loss_change %.4f loss_sem_u %.4f lambda_sem_u %.4f pseudo_ratio %.3f bn_loss %.4f sc_loss %.4f acc %.2f] precision=%s' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    metric_snapshot['sem_sup']['last'], metric_snapshot['change']['last'],
                    metric_snapshot['pseudo']['last'], metric_snapshot['lambda_sem_u']['last'],
                    metric_snapshot['pseudo_ratio']['last'], metric_snapshot['bn']['last'],
                    metric_snapshot['sc']['last'], metric_snapshot['accuracy']['last'] * 100,
                    precision_log))
                writer.add_scalar('train/loss_sem_sup', metric_snapshot['sem_sup']['last'], running_iter)
                writer.add_scalar('train/loss_change', metric_snapshot['change']['last'], running_iter)
                writer.add_scalar('train/loss_bn', metric_snapshot['bn']['last'], running_iter)
                writer.add_scalar('train/loss_sc', metric_snapshot['sc']['last'], running_iter)
                writer.add_scalar('train/loss_sem_u', metric_snapshot['pseudo']['last'], running_iter)
                writer.add_scalar('train/lambda_sem_u_current', metric_snapshot['lambda_sem_u']['last'], running_iter)
                writer.add_scalar('train/pseudo_ratio', metric_snapshot['pseudo_ratio']['last'], running_iter)
                if graph_runtime is not None:
                    print('[sam-graph-stage1] loss_graph_gt={:.4f} loss_edge_gt={:.4f} gt_nodes={:.1f}'.format(
                        metric_snapshot['graph_gt']['last'], metric_snapshot['edge_gt']['last'],
                        metric_snapshot['gt_nodes']['last']))
                    writer.add_scalar('train/graph/loss_graph_gt', metric_snapshot['graph_gt']['last'], running_iter)
                    writer.add_scalar('train/graph/loss_edge_gt', metric_snapshot['edge_gt']['last'], running_iter)
                    writer.add_scalar('train/graph/gt_nodes', metric_snapshot['gt_nodes']['last'], running_iter)
                if stage2_active:
                    pseudo_histogram = (
                        graph_veto.class_histogram.detach().cpu().tolist()
                        if graph_veto is not None else [0] * RS.num_classes)
                    print('[sam-graph-stage2] loss_gt={:.4f} loss_temp={:.4f} '
                          'loss_pixel={:.4f} weight={:.4f} nodes={:.1f} '
                          'GT/U/Ignore={:.1f}/{:.1f}/{:.1f} edges={:.1f} '
                          'prop/spatial/temp={:.1f}/{:.1f}/{:.1f} edge_w={:.3f} '
                          'gt_acc={:.3f} graph_conf={:.3f} graph_valid={:.3f} '
                          'teacher_candidate={:.3f} veto={:.3f} fallback={:.3f} '
                          'accepted={:.3f} grad_student/graph={:.3f}/{:.3f} '
                          'nonfinite_skips={} classes={}'.format(
                              metric_snapshot['graph_stage2_gt']['last'],
                              metric_snapshot['graph_temp']['last'],
                              metric_snapshot['graph_pixel_pseudo']['last'],
                              metric_snapshot['graph_pseudo_weight']['last'],
                              metric_snapshot['graph_nodes']['last'],
                              metric_snapshot['gt_nodes']['last'],
                              metric_snapshot['u_nodes']['last'],
                              metric_snapshot['ignore_nodes']['last'],
                              metric_snapshot['graph_edges']['last'],
                              metric_snapshot['propagation_edges']['last'],
                              metric_snapshot['spatial_edges']['last'],
                              metric_snapshot['temporal_edges']['last'],
                              metric_snapshot['edge_reliability']['last'],
                              metric_snapshot['graph_gt_accuracy']['last'],
                              metric_snapshot['graph_confidence']['last'],
                              metric_snapshot['graph_valid_ratio']['last'],
                              metric_snapshot['teacher_candidate_ratio']['last'],
                              metric_snapshot['graph_veto_ratio']['last'],
                              metric_snapshot['graph_fallback_ratio']['last'],
                              metric_snapshot['graph_accepted_ratio']['last'],
                              metric_snapshot['student_grad_norm']['last'],
                              metric_snapshot['graph_grad_norm']['last'],
                              graph_runtime.stage2_counters['nonfinite_skips'],
                              pseudo_histogram))
                    for name in (
                            'graph_stage2_gt', 'graph_temp', 'graph_pixel_pseudo',
                            'graph_pseudo_weight', 'graph_nodes', 'graph_edges',
                            'graph_valid_ratio', 'teacher_candidate_ratio',
                            'graph_veto_ratio', 'graph_fallback_ratio',
                            'graph_accepted_ratio', 'graph_confidence',
                            'edge_reliability', 'graph_gt_accuracy',
                            'student_grad_norm', 'graph_grad_norm'):
                        writer.add_scalar(
                            'train/graph_stage2/' + name,
                            metric_snapshot[name]['last'], running_iter)
                for cls_idx, cls_precision in enumerate(class_precision.detach().cpu().tolist()):
                    writer.add_scalar('train/class_precision/class_%d' % cls_idx, cls_precision, running_iter)
                writer.add_scalar('train accuracy', metric_snapshot['accuracy']['last'], running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        scheduler.step()
        if pseudo_alpha_diagnostics is not None:
            print('[pseudo-alpha sweep] measured=GT classes 1..{}; unlabeled=coverage only'.format(
                RS.num_classes - 1))
            rows = pseudo_alpha_diagnostic_rows(pseudo_alpha_diagnostics)
            for cls in range(1, RS.num_classes):
                class_rows = [row for row in rows if row['class_id'] == cls]
                first = class_rows[0]
                print('  class={} pred_measured={} pred_unlabeled={} entropy[min/mean/max]={}/{}/{}'.format(
                    cls, first['measured_candidate'], first['unlabeled_candidate'],
                    'NA' if first['entropy_min'] is None else '{:.4f}'.format(first['entropy_min']),
                    'NA' if first['entropy_mean'] is None else '{:.4f}'.format(first['entropy_mean']),
                    'NA' if first['entropy_max'] is None else '{:.4f}'.format(first['entropy_max'])))
                for row in class_rows:
                    precision = ('NA' if row['measured_precision'] is None
                                 else '{:.4f}'.format(row['measured_precision']))
                    cutoff = ('NA' if row['entropy_cutoff_mean'] is None
                              else '{:.4f}'.format(row['entropy_cutoff_mean']))
                    print('    alpha={:.2f} cutoff_mean={} measured={}/{} tp={} fp={} precision={} unlabeled={}/{} coverage={:.4f}'.format(
                        row['ratio'], cutoff, row['measured_accepted'],
                        row['measured_candidate'], row['measured_tp'],
                        row['measured_fp'], precision, row['unlabeled_accepted'],
                        row['unlabeled_candidate'], row['unlabeled_coverage']))
        if opt.use_ema_teacher and opt.use_semantic_pseudo and curr_epoch >= opt.pseudo_start_epoch:
            class_precision = update_class_precision_from_stats(
                epoch_precision_tp, epoch_precision_fp, class_precision,
                opt.pseudo_precision_momentum)
            class_precision[0] = 0.0
        print('Epoch %d class_precision=%s lambda_sem_u=%.4f' % (
            curr_epoch,
            ['%.3f' % v for v in class_precision.detach().cpu().tolist()],
            get_lambda_sem_u(opt, curr_epoch)))

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        epoch_train_accuracy = train_metrics.snapshot()['accuracy']['average']
        if epoch_train_accuracy > bestaccT: bestaccT = epoch_train_accuracy
        is_best = Fscd_v > bestFscdV
        if is_best:
            bestFscdV = Fscd_v
            bestaccV = acc_v
            bestloss = loss_v
        checkpoint_payload = build_training_state(
            student=net, teacher=net_psd, optimizer=optimizer, scheduler=scheduler,
            epoch=curr_epoch + 1, optimizer_step=optimizer_step,
            class_precision=class_precision,
            best_metrics={"train_acc": bestaccT, "Fscd": bestFscdV, "mIoU": mIoU_v,
                          "Sek": Sek_v, "accuracy": bestaccV, "loss": bestloss},
            run_config=run_config)
        if graph_runtime is not None:
            graph_runtime.add_to_checkpoint_payload(checkpoint_payload)
        checkpoint_name = "{}_{}_seed{}_epoch{:03d}_Fscd{:.4f}.pth".format(
            run_config["dataset"], run_config["mode"], run_config["seed"], curr_epoch, Fscd_v)
        if is_best or (curr_epoch + 1) % args["predict_step"] == 0:
            torch.save(checkpoint_payload, os.path.join(args["chkpt_dir"], checkpoint_name))
        print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f' % (
        time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    invalid_val_batches = 0
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda(non_blocking=True).float()
            imgs_B = imgs_B.cuda(non_blocking=True).float()
            labels_A = labels_A.cuda(non_blocking=True).long()
            labels_B = labels_B.cuda(non_blocking=True).long()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=opt.amp):
            imgs_A, imgs_B = scale.scale_input((imgs_A, imgs_B))
            outs = net(imgs_A, imgs_B)
            outs = scale.scale_output(outs)
            outputs_A, outputs_B, out_change = outs

            loss = supervised_semantic_loss(outputs_A, outputs_B, labels_A, labels_B)
            if not all(torch.isfinite(tensor).all().item() for tensor in (outputs_A, outputs_B, out_change, loss)):
                invalid_val_batches += 1
                repaired = _repair_nonfinite_state(net)
                print("[Warning] non-finite validation batch {} at epoch {}; batch skipped repaired_state={}".format(vi, curr_epoch, repaired))
                continue
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = (out_change.detach() > 0).cpu()
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze(1).long()).numpy()
        preds_B = (preds_B * change_mask.squeeze(1).long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
            print('Prediction saved!')

    if invalid_val_batches:
        print("[Warning] validation skipped {} non-finite batches at epoch {}".format(invalid_val_batches, curr_epoch))
    if not preds_all:
        print("[Warning] validation produced no finite predictions at epoch {}; metrics skipped.".format(curr_epoch))
        return 0.0, 0.0, 0.0, 0.0, float("inf")
    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(curr_time, val_loss.average(), Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Change Detection train')
    parser.add_argument("--dataset", choices=("landsat", "second"),
                        default=_selected_dataset.dataset)
    parser.add_argument("--num-classes", type=int,
                        default=training_config.model_num_classes)
    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str, default="")  
    parser.add_argument("--resume", type=str, default="",
                        help="Load model weights from a .pth checkpoint.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch index to continue from after loading --resume.")
    parser.add_argument("--cuda", type=int, default=0,
                        help="Logical CUDA index after CUDA_VISIBLE_DEVICES remapping.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Physical micro-batch size; 2 fits a 24 GB RTX 3090.")
    parser.add_argument("--effective-batch-size", type=int, default=8,
                        help="Target batch size after gradient accumulation.")
    parser.add_argument("--val-batch-size", type=int, default=args['val_batch_size'])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum gradient norm; set <= 0 to disable clipping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Use CUDA automatic mixed precision to reduce memory usage.")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable CUDA automatic mixed precision.")
    add_performance_arguments(parser)
    parser.add_argument("--debug-nonfinite", action="store_true",
                        help="Stop at the first module that produces NaN/Inf and print tensor stats.")
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--use-ema-teacher", dest="use_ema_teacher", action="store_true", default=False,
                        help="Enable EMA teacher model for pseudo-label generation.")
    parser.add_argument("--no-use-ema-teacher", dest="use_ema_teacher", action="store_false",
                        help="Disable EMA teacher updates.")
    parser.add_argument("--use-semantic-pseudo", dest="use_semantic_pseudo", action="store_true", default=False,
                        help="Enable entropy-filtered semantic pseudo labels on unchanged regions.")
    parser.add_argument("--no-use-semantic-pseudo", dest="use_semantic_pseudo", action="store_false",
                        help="Disable semantic pseudo labels for the supervised baseline.")
    parser.add_argument("--ema-decay", type=float, default=args['ema_decay'],
                        help="EMA decay used to update the teacher model from the student.")
    parser.add_argument("--pseudo-start-epoch", type=int, default=30,
                        help="Epoch to start semantic pseudo-label supervision.")
    parser.add_argument("--pseudo-alpha-start", type=float, default=0.5,
                        help="Initial per-class pseudo-label keep ratio.")
    parser.add_argument("--pseudo-alpha-end", type=float, default=0.8,
                        help="Final per-class pseudo-label keep ratio.")
    parser.add_argument("--pseudo-min-ratio", type=float, default=0.05,
                        help="Minimum class-wise keep ratio before precision scaling.")
    parser.add_argument("--pseudo-ignore-index", type=int, default=255,
                        help="Ignore index used for unreliable pseudo-label pixels.")
    parser.add_argument("--pseudo-change-thr", type=float, default=0.3,
                        help="Teacher change probability threshold for unchanged pseudo-label candidates.")
    parser.add_argument("--lambda-sem-u", type=float, default=args['lambda_sem_u'],
                        help="Weight of semantic pseudo-label loss.")
    parser.add_argument("--lambda-sem-sup", type=float, default=args['lambda_sem_sup'],
                        help="Weight of supervised semantic loss.")
    parser.add_argument("--pseudo-rampup-epoch", type=int, default=5,
                        help="Epochs used to ramp up semantic pseudo-label loss weight.")
    parser.add_argument("--pseudo-precision-init", type=float, default=0.8,
                        help="Initial teacher class precision used for class-wise keep ratios.")
    parser.add_argument("--pseudo-precision-momentum", type=float, default=0.9,
                        help="Momentum for EMA class precision updates.")
    parser.add_argument("--pseudo-visualization-interval", type=int, default=200,
                        help="Save pseudo-label visualization every N iterations; 0 disables it.")
    parser.add_argument("--pseudo-diagnostic-interval", type=int, default=50,
                        help="Sample the alpha sweep every N iterations; 0 disables it.")
    add_graph_stage1_arguments(parser)

    opt = parser.parse_args()

    if opt.batch_size < 1 or opt.effective_batch_size < opt.batch_size:
        parser.error("batch sizes must be positive and effective batch must be >= physical batch")
    if opt.effective_batch_size % opt.batch_size != 0:
        parser.error("--effective-batch-size must be divisible by --batch-size")
    if opt.use_semantic_pseudo and not opt.use_ema_teacher:
        parser.error("--use-semantic-pseudo requires --use-ema-teacher")
    if opt.use_sam_graph_stage2 and not opt.use_sam_graph_stage1:
        parser.error("--use-sam-graph-stage2 requires --use-sam-graph-stage1")
    if opt.use_sam_graph_stage2 and not opt.use_ema_teacher:
        parser.error("--use-sam-graph-stage2 requires --use-ema-teacher")
    if opt.grad_clip <= 0:
        opt.grad_clip = float("inf")
    if opt.start_epoch < 0:
        parser.error("--start-epoch must be >= 0")
    if opt.start_epoch > 0 and not opt.resume:
        parser.error("--start-epoch > 0 requires --resume")
    if opt.start_epoch >= opt.epochs:
        parser.error("--start-epoch must be smaller than --epochs")
    if opt.pseudo_visualization_interval < 0 or opt.pseudo_diagnostic_interval < 0:
        parser.error("pseudo sampling intervals must be non-negative")
    if opt.lambda_graph_gt < 0 or opt.lambda_edge_gt < 0:
        parser.error("graph loss weights must be non-negative")
    if not 0.0 <= opt.sam_node_min_area_ratio <= opt.sam_node_max_area_ratio <= 1.0:
        parser.error("SAM node area ratios must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= opt.graph_hard_negative_ratio <= 1.0:
        parser.error("--graph-hard-negative-ratio must be in [0, 1]")
    if opt.graph_stage2_start_epoch < 0 or opt.graph_local_topk < 0:
        parser.error("Stage2 start epoch and local Top-K must be non-negative")
    if opt.graph_gat_heads <= 0 or opt.graph_node_dim % opt.graph_gat_heads != 0:
        parser.error("--graph-node-dim must be divisible by --graph-gat-heads")
    if not 0.0 <= opt.graph_gat_dropout < 1.0:
        parser.error("--graph-gat-dropout must be in [0, 1)")
    if not 0.0 <= opt.graph_edge_threshold <= 1.0:
        parser.error("--graph-edge-threshold must be in [0, 1]")
    if not 0.0 <= opt.graph_confidence_threshold <= 1.0:
        parser.error("--graph-confidence-threshold must be in [0, 1]")
    if not 0.0 <= opt.graph_teacher_threshold <= opt.teacher_fallback_threshold <= 1.0:
        parser.error("graph teacher thresholds must satisfy 0 <= base <= fallback <= 1")
    if (opt.graph_temp_weight < 0 or opt.graph_pixel_pseudo_weight < 0
            or opt.graph_pseudo_ramp_epochs < 0):
        parser.error("Stage2 loss weights and ramp epochs must be non-negative")
    if opt.graph_grad_clip <= 0:
        opt.graph_grad_clip = float("inf")

    main(opt)
