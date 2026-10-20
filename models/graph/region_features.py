import torch
import torch.nn as nn
import torch.nn.functional as F


class SamGraphFeatureProvider(nn.Module):
    """Select stable MobileSAM stage3 and encoder-neck outputs for A/B."""

    def __init__(self, backbone, detach=True):
        super().__init__()
        self.backbone = backbone
        self.detach = bool(detach)

    def _select(self, image):
        outputs = self.backbone(image)
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
            raise ValueError("MobileSAM backbone must return at least five feature tensors")
        stage3, neck = outputs[2], outputs[4]
        if stage3.ndim != 4 or neck.ndim != 4:
            raise ValueError("stage3 and neck features must have shape [B, C, H, W]")
        if stage3.shape[0] != neck.shape[0]:
            raise ValueError("stage3 and neck batch dimensions must match")
        if not torch.isfinite(stage3).all() or not torch.isfinite(neck).all():
            raise FloatingPointError("MobileSAM graph features contain NaN or Inf")
        if self.detach:
            stage3, neck = stage3.detach(), neck.detach()
        return stage3, neck

    def forward(self, image_a, image_b):
        stage_a, neck_a = self._select(image_a)
        stage_b, neck_b = self._select(image_b)
        if stage_a.shape != stage_b.shape or neck_a.shape != neck_b.shape:
            raise ValueError("A/B graph feature shapes must match")
        return stage_a, neck_a, stage_b, neck_b


def pool_mask_regions(feature, masks):
    """Return FP32 masked mean/max for one sample and validity per source mask."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("feature must have shape [1, C, H, W]")
    masks = torch.as_tensor(masks, dtype=torch.bool, device=feature.device)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    channels = int(feature.shape[1])
    if masks.shape[0] == 0:
        empty = torch.empty((0, channels), device=feature.device, dtype=torch.float32)
        return empty, empty.clone(), torch.empty((0,), device=feature.device, dtype=torch.bool)

    mapped = F.interpolate(
        masks[:, None].float(), size=feature.shape[-2:], mode="nearest")[:, 0].bool()
    valid = mapped.flatten(1).any(dim=1)
    weights = mapped[:, None].float()
    values = feature[0].float().unsqueeze(0)
    counts = weights.flatten(2).sum(dim=2).clamp_min(1.0)
    means = (values * weights).flatten(2).sum(dim=2) / counts
    fill = torch.finfo(torch.float32).min
    maxima = values.expand(mapped.shape[0], -1, -1, -1).masked_fill(~mapped[:, None], fill)
    maxima = maxima.flatten(2).max(dim=2).values
    means = torch.where(valid[:, None], means, torch.zeros_like(means))
    maxima = torch.where(valid[:, None], maxima, torch.zeros_like(maxima))
    return means, maxima, valid


def mask_geometry(masks):
    """Compute normalized center, box size, and area for `[N,H,W]` masks."""
    masks = torch.as_tensor(masks, dtype=torch.bool)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    count, height, width = masks.shape
    output = torch.zeros((count, 5), dtype=torch.float32, device=masks.device)
    for index in range(count):
        coordinates = torch.nonzero(masks[index], as_tuple=False)
        if coordinates.numel() == 0:
            continue
        y_min, x_min = coordinates.min(dim=0).values
        y_max, x_max = coordinates.max(dim=0).values
        center_x = (x_min.float() + x_max.float()) * 0.5 / float(max(width - 1, 1))
        center_y = (y_min.float() + y_max.float()) * 0.5 / float(max(height - 1, 1))
        box_width = (x_max - x_min + 1).float() / float(width)
        box_height = (y_max - y_min + 1).float() / float(height)
        area = masks[index].sum().float() / float(height * width)
        output[index] = torch.stack((center_x, center_y, box_width, box_height, area))
    return output
