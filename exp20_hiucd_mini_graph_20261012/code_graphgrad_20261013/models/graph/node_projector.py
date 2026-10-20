from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from models.graph.region_features import mask_geometry, pool_mask_regions


@dataclass
class TemporalNodeOutput:
    nodes_a: torch.Tensor
    nodes_b: torch.Tensor
    geometry: torch.Tensor
    valid: torch.Tensor


class TemporalNodeProjector(nn.Module):
    """Pool shared masks over A/B stage+neck features and project temporal nodes."""

    def __init__(
        self,
        stage_channels=320,
        neck_channels=256,
        node_dim=128,
        embedding_dim=8,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.stage_channels = int(stage_channels)
        self.neck_channels = int(neck_channels)
        self.node_dim = int(node_dim)
        self.embedding_dim = int(embedding_dim)
        if min(self.stage_channels, self.neck_channels, self.node_dim, self.embedding_dim) <= 0:
            raise ValueError("projector dimensions must be positive")
        pooled_dim = 2 * self.stage_channels + 2 * self.neck_channels
        input_dim = 3 * pooled_dim + 5 + 2 * self.embedding_dim
        self.time_embedding = nn.Embedding(2, self.embedding_dim)
        self.scale_embedding = nn.Embedding(3, self.embedding_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.node_dim),
        )

    def _validate_features(self, stage_a, neck_a, stage_b, neck_b, masks_list):
        tensors = (stage_a, neck_a, stage_b, neck_b)
        if any(tensor.ndim != 4 for tensor in tensors):
            raise ValueError("graph features must have shape [B, C, H, W]")
        batch = stage_a.shape[0]
        if any(tensor.shape[0] != batch for tensor in tensors) or len(masks_list) != batch:
            raise ValueError("feature batches and masks_list length must match")
        if stage_a.shape[1] != self.stage_channels or stage_b.shape[1] != self.stage_channels:
            raise ValueError("stage feature channel count does not match projector configuration")
        if neck_a.shape[1] != self.neck_channels or neck_b.shape[1] != self.neck_channels:
            raise ValueError("neck feature channel count does not match projector configuration")
        if stage_a.shape != stage_b.shape or neck_a.shape != neck_b.shape:
            raise ValueError("A/B feature shapes must match")

    def _project_time(self, feature, difference, product, geometry, time_index):
        count = feature.shape[0]
        if count == 0:
            return feature.new_empty((0, self.node_dim), dtype=torch.float32)
        time = self.time_embedding.weight[time_index].float().expand(count, -1)
        scale = self.scale_embedding.weight[1].float().expand(count, -1)
        raw = torch.cat((feature, difference, product, geometry.float(), time, scale), dim=1)
        return self.projector(raw.float())

    def forward(self, stage_a, neck_a, stage_b, neck_b, masks_list) -> List[TemporalNodeOutput]:
        self._validate_features(stage_a, neck_a, stage_b, neck_b, masks_list)
        outputs = []
        for index, source_masks in enumerate(masks_list):
            masks = torch.as_tensor(source_masks, dtype=torch.bool, device=stage_a.device)
            stage_mean_a, stage_max_a, stage_valid_a = pool_mask_regions(stage_a[index:index + 1], masks)
            stage_mean_b, stage_max_b, stage_valid_b = pool_mask_regions(stage_b[index:index + 1], masks)
            neck_mean_a, neck_max_a, neck_valid_a = pool_mask_regions(neck_a[index:index + 1], masks)
            neck_mean_b, neck_max_b, neck_valid_b = pool_mask_regions(neck_b[index:index + 1], masks)
            feature_a = torch.cat((stage_mean_a, stage_max_a, neck_mean_a, neck_max_a), dim=1)
            feature_b = torch.cat((stage_mean_b, stage_max_b, neck_mean_b, neck_max_b), dim=1)
            difference = (feature_a - feature_b).abs()
            product = feature_a * feature_b
            geometry = mask_geometry(masks).to(device=stage_a.device, dtype=torch.float32)
            nodes_a = self._project_time(feature_a, difference, product, geometry, 0)
            nodes_b = self._project_time(feature_b, difference, product, geometry, 1)
            valid = stage_valid_a & stage_valid_b & neck_valid_a & neck_valid_b
            outputs.append(TemporalNodeOutput(nodes_a, nodes_b, geometry, valid))
        return outputs
