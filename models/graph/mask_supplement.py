import torch
import torch.nn as nn
import torch.nn.functional as F


class SamMaskSupplement(nn.Module):
    """Supplement semantic-neck features with SAM mask region pooling.

    For each spatial scale, each pixel is replaced with its SAM-region's
    pooled feature (soft scatter), then gated by temporal uncertainty so
    the supplement only contributes where the change signal is ambiguous.

    Args:
        channels_list: list of channel counts for each supplemented scale,
                       e.g. [96, 192, 384] for inplanes=96.
    """

    def __init__(self, channels_list):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, max(c // 8, 1), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(c // 8, 1), 1, 1),
            )
            for c in channels_list
        ])
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, c, 1, bias=False) for c in channels_list
        ])
        self._init_weights()

    def _init_weights(self):
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
        for lat in self.laterals:
            nn.init.eye_(lat.weight.view(lat.weight.shape[0], -1))

    @staticmethod
    @torch.no_grad()
    def _region_scatter(feat, masks):
        """Pool feature over SAM mask regions and broadcast back.

        Gradients are detached here (matching the existing graph detach
        convention); the gate and lateral conv still receive gradients.

        feat:  [1, C, H', W']
        masks: [N, H,  W ] bool  (original resolution)
        returns [1, C, H', W']
        """
        _, C, H, W = feat.shape
        N = masks.shape[0]
        if N == 0:
            return torch.zeros_like(feat)

        masks_r = F.interpolate(
            masks.float().unsqueeze(1), size=(H, W), mode="nearest"
        ).squeeze(1)  # [N, H', W']

        area = masks_r.flatten(1).sum(1).clamp_min(1.0)  # [N]
        means = torch.einsum(
            "nhw,chw->nc", masks_r, feat[0].float()
        ) / area.unsqueeze(1)  # [N, C]

        coverage = masks_r.sum(0).clamp_min(1.0)  # [H', W']
        scattered = torch.einsum("nc,nhw->chw", means, masks_r) / coverage

        return scattered.unsqueeze(0).to(feat.dtype)  # [1, C, H', W']

    def forward(self, feats_a, feats_b, masks_list):
        """
        feats_a, feats_b: lists of [B, C_i, H_i, W_i], one tensor per scale
        masks_list:       list of B elements, each a bool Tensor [N_i, H, W]
                          or None when no masks are available for that sample
        Returns: (out_a, out_b) same structure as feats_a/feats_b
        """
        B = feats_a[0].shape[0]
        out_a = list(feats_a)
        out_b = list(feats_b)

        for i, (gate_conv, lateral) in enumerate(zip(self.gates, self.laterals)):
            fa, fb = feats_a[i], feats_b[i]

            supp_a_parts, supp_b_parts = [], []
            for b in range(B):
                m = masks_list[b]
                if m is None or (torch.is_tensor(m) and m.shape[0] == 0):
                    empty = torch.zeros(0, *fa.shape[-2:],
                                        dtype=torch.bool, device=fa.device)
                    supp_a_parts.append(self._region_scatter(fa[b:b+1], empty))
                    supp_b_parts.append(self._region_scatter(fb[b:b+1], empty))
                else:
                    masks_t = torch.as_tensor(m, dtype=torch.bool, device=fa.device)
                    supp_a_parts.append(self._region_scatter(fa[b:b+1], masks_t))
                    supp_b_parts.append(self._region_scatter(fb[b:b+1], masks_t))

            supp_a = torch.cat(supp_a_parts, dim=0)  # [B, C, H', W']
            supp_b = torch.cat(supp_b_parts, dim=0)

            diff = (fa - fb).abs()
            gate = torch.sigmoid(gate_conv(diff))  # [B, 1, H', W']

            out_a[i] = fa + gate * lateral(supp_a)
            out_b[i] = fb + gate * lateral(supp_b)

        return out_a, out_b
