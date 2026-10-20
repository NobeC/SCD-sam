import torch


@torch.no_grad()
def generate_entropy_pseudo_label(
    logits,
    candidate_mask,
    num_classes,
    gamma_per_class,
    ignore_index=255,
):
    """
    Build class-wise entropy filtered pseudo labels from teacher logits.

    Args:
        logits: Tensor [B, C, H, W].
        candidate_mask: Bool tensor [B, H, W], pixels allowed for pseudo labels.
        num_classes: Number of semantic classes in logits.
        gamma_per_class: Per-class keep ratio. Set a class ratio to 0 to skip it.
        ignore_index: Target value for unreliable pixels.
    """
    # Entropy must stay in FP32: FP16 underflows log(eps) and can create NaNs.
    prob = torch.softmax(logits.float(), dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    pred = torch.argmax(prob, dim=1)

    if not torch.is_tensor(gamma_per_class):
        gamma_per_class = torch.as_tensor(gamma_per_class, device=logits.device)
    gamma_per_class = gamma_per_class.to(device=logits.device, dtype=torch.float32)

    candidate_mask = candidate_mask.bool()
    pseudo_target = torch.full_like(pred, fill_value=ignore_index)
    reliable_mask = torch.zeros_like(pred, dtype=torch.bool)

    for cls in range(num_classes):
        cls_mask = (pred == cls) & candidate_mask
        num_valid = int(cls_mask.sum().item())
        if num_valid <= 0:
            continue

        ratio = float(gamma_per_class[cls].clamp(0.0, 1.0).item())
        num_keep = int(num_valid * ratio)
        if num_keep <= 0:
            continue

        cls_entropy = entropy[cls_mask]
        if num_keep >= num_valid:
            threshold = cls_entropy.max()
        else:
            threshold = torch.topk(cls_entropy, k=num_keep, largest=False).values.max()

        keep_mask = cls_mask & (entropy <= threshold)
        pseudo_target[keep_mask] = pred[keep_mask]
        reliable_mask[keep_mask] = True

    return pseudo_target.long(), reliable_mask, entropy


@torch.no_grad()
def accumulate_class_precision_stats(
    logits,
    labels,
    valid_mask,
    num_classes,
    true_positive,
    false_positive,
):
    """Accumulate teacher TP/FP counts for epoch-level class precision."""
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    valid_mask = valid_mask.bool()

    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid_mask
        if pred_cls.any():
            true_positive[cls] += ((labels == cls) & pred_cls).sum().float()
            false_positive[cls] += ((labels != cls) & pred_cls).sum().float()

    return true_positive, false_positive


@torch.no_grad()
def update_class_precision_from_stats(
    true_positive,
    false_positive,
    old_precision,
    momentum=0.9,
    eps=1e-6,
):
    """Update class precision once per epoch from accumulated TP/FP counts.

    Classes that were never predicted during the epoch keep their previous
    precision instead of being treated as perfect or zero precision.
    """
    precision = old_precision.clone().to(device=true_positive.device)
    predicted = (true_positive + false_positive) > 0
    if predicted.any():
        epoch_precision = true_positive[predicted] / (
            true_positive[predicted] + false_positive[predicted] + eps)
        precision[predicted] = momentum * precision[predicted] + (1.0 - momentum) * epoch_precision

    return precision.clamp(min=0.0, max=1.0)

@torch.no_grad()
def create_pseudo_alpha_diagnostics(
    num_classes,
    ratios=(0.03, 0.05, 0.10, 0.15, 0.20, 0.30),
    device=None,
):
    """Create additive epoch statistics for an entropy-ranked alpha sweep."""
    device = torch.device("cpu") if device is None else device
    ratios = tuple(float(ratio) for ratio in ratios)
    shape = (len(ratios), int(num_classes))
    return {
        "num_classes": int(num_classes),
        "ratios": ratios,
        "measured_candidate": torch.zeros(num_classes, dtype=torch.long, device=device),
        "measured_accepted": torch.zeros(shape, dtype=torch.long, device=device),
        "measured_tp": torch.zeros(shape, dtype=torch.long, device=device),
        "measured_fp": torch.zeros(shape, dtype=torch.long, device=device),
        "unlabeled_candidate": torch.zeros(num_classes, dtype=torch.long, device=device),
        "unlabeled_accepted": torch.zeros(shape, dtype=torch.long, device=device),
        "entropy_count": torch.zeros(num_classes, dtype=torch.long, device=device),
        "entropy_sum": torch.zeros(num_classes, dtype=torch.float64, device=device),
        "entropy_min": torch.full((num_classes,), float("inf"), device=device),
        "entropy_max": torch.full((num_classes,), float("-inf"), device=device),
        "cutoff_sum": torch.zeros(shape, dtype=torch.float64, device=device),
        "cutoff_count": torch.zeros(shape, dtype=torch.long, device=device),
    }


def _lowest_entropy_indices(entropy, mask, ratio):
    flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
    num_keep = int(flat_indices.numel() * ratio)
    if num_keep <= 0:
        return flat_indices[:0], None
    flat_entropy = entropy.reshape(-1)
    order = torch.argsort(flat_entropy[flat_indices], stable=True)
    selected = flat_indices[order[:num_keep]]
    return selected, float(flat_entropy[selected].max().item())


@torch.no_grad()
def accumulate_pseudo_alpha_diagnostics(
    state, logits, labels, measured_mask, unlabeled_mask,
):
    """Accumulate valid-GT precision separately from unlabeled coverage."""
    probabilities = torch.softmax(logits.float(), dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum(dim=1)
    prediction = probabilities.argmax(dim=1)
    labels_flat = labels.reshape(-1)

    for cls in range(1, state["num_classes"]):
        predicted_cls = prediction == cls
        measured_cls = predicted_cls & measured_mask.bool()
        unlabeled_cls = predicted_cls & unlabeled_mask.bool()
        state["measured_candidate"][cls] += measured_cls.sum()
        state["unlabeled_candidate"][cls] += unlabeled_cls.sum()

        cls_entropy = entropy[unlabeled_cls]
        if cls_entropy.numel():
            state["entropy_count"][cls] += cls_entropy.numel()
            state["entropy_sum"][cls] += cls_entropy.double().sum()
            state["entropy_min"][cls] = torch.minimum(
                state["entropy_min"][cls], cls_entropy.min())
            state["entropy_max"][cls] = torch.maximum(
                state["entropy_max"][cls], cls_entropy.max())

        for ratio_idx, ratio in enumerate(state["ratios"]):
            measured_indices, _ = _lowest_entropy_indices(
                entropy, measured_cls, ratio)
            measured_kept = int(measured_indices.numel())
            state["measured_accepted"][ratio_idx, cls] += measured_kept
            if measured_kept:
                selected_labels = labels_flat[measured_indices]
                tp = int((selected_labels == cls).sum().item())
                state["measured_tp"][ratio_idx, cls] += tp
                state["measured_fp"][ratio_idx, cls] += measured_kept - tp

            unlabeled_indices, cutoff = _lowest_entropy_indices(
                entropy, unlabeled_cls, ratio)
            state["unlabeled_accepted"][ratio_idx, cls] += unlabeled_indices.numel()
            if cutoff is not None:
                state["cutoff_sum"][ratio_idx, cls] += cutoff
                state["cutoff_count"][ratio_idx, cls] += 1
    return state


def pseudo_alpha_diagnostic_rows(state):
    """Return class-sorted, ratio-sorted printable diagnostic rows."""
    rows = []
    for cls in range(1, state["num_classes"]):
        entropy_count = int(state["entropy_count"][cls].item())
        measured_candidate = int(state["measured_candidate"][cls].item())
        unlabeled_candidate = int(state["unlabeled_candidate"][cls].item())
        for ratio_idx, ratio in enumerate(state["ratios"]):
            measured_accepted = int(state["measured_accepted"][ratio_idx, cls].item())
            measured_tp = int(state["measured_tp"][ratio_idx, cls].item())
            measured_fp = int(state["measured_fp"][ratio_idx, cls].item())
            unlabeled_accepted = int(state["unlabeled_accepted"][ratio_idx, cls].item())
            cutoff_count = int(state["cutoff_count"][ratio_idx, cls].item())
            rows.append({
                "class_id": cls,
                "ratio": ratio,
                "measured_candidate": measured_candidate,
                "measured_accepted": measured_accepted,
                "measured_tp": measured_tp,
                "measured_fp": measured_fp,
                "measured_precision": measured_tp / measured_accepted if measured_accepted else None,
                "unlabeled_candidate": unlabeled_candidate,
                "unlabeled_accepted": unlabeled_accepted,
                "unlabeled_coverage": unlabeled_accepted / unlabeled_candidate if unlabeled_candidate else 0.0,
                "entropy_min": float(state["entropy_min"][cls].item()) if entropy_count else None,
                "entropy_mean": float(state["entropy_sum"][cls].item()) / entropy_count if entropy_count else None,
                "entropy_max": float(state["entropy_max"][cls].item()) if entropy_count else None,
                "entropy_cutoff_mean": float(state["cutoff_sum"][ratio_idx, cls].item()) / cutoff_count if cutoff_count else None,
            })
    return rows

