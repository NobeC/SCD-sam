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
    def _scatter_preresized(feat, masks_r):
        """Pool and scatter with already-resized masks.

        feat:    [1, C, H', W']
        masks_r: [N, H', W'] float, already at feature resolution
        returns  [1, C, H', W']
        """
        N = masks_r.shape[0]
        if N == 0:
            return torch.zeros_like(feat)
        _, C, H, W = feat.shape
        # [N, H'*W'] and [C, H'*W']
        m_flat = masks_r.flatten(1)                          # [N, HW]
        f_flat = feat[0].float().flatten(1)                  # [C, HW]
        area = m_flat.sum(1).clamp_min(1.0)                  # [N]
        means = (m_flat @ f_flat.T) / area.unsqueeze(1)      # [N, C]
        coverage = m_flat.sum(0).clamp_min(1.0)              # [HW]
        scattered = (means.T @ m_flat) / coverage            # [C, HW]
        return scattered.view(1, C, H, W).to(feat.dtype)

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

        # Pre-convert masks to float tensors once (device transfer + bool→float)
        masks_gpu = []
        for b in range(B):
            m = masks_list[b]
            if m is None or (torch.is_tensor(m) and m.shape[0] == 0):
                masks_gpu.append(None)
            else:
                masks_gpu.append(torch.as_tensor(m, dtype=torch.float32,
                                                  device=feats_a[0].device))

        for i, (gate_conv, lateral) in enumerate(zip(self.gates, self.laterals)):
            fa, fb = feats_a[i], feats_b[i]
            _, _, H, W = fa.shape

            supp_a_parts, supp_b_parts = [], []
            for b in range(B):
                mf = masks_gpu[b]
                if mf is None:
                    supp_a_parts.append(torch.zeros_like(fa[b:b + 1]))
                    supp_b_parts.append(torch.zeros_like(fb[b:b + 1]))
                else:
                    # Resize once per scale (shared for A and B)
                    orig_H, orig_W = mf.shape[-2], mf.shape[-1]
                    if orig_H != H or orig_W != W:
                        mr = F.interpolate(mf.unsqueeze(1), size=(H, W),
                                           mode="nearest").squeeze(1)
                    else:
                        mr = mf
                    supp_a_parts.append(self._scatter_preresized(fa[b:b + 1], mr))
                    supp_b_parts.append(self._scatter_preresized(fb[b:b + 1], mr))

            supp_a = torch.cat(supp_a_parts, dim=0)
            supp_b = torch.cat(supp_b_parts, dim=0)

            diff = (fa - fb).abs()
            gate = torch.sigmoid(gate_conv(diff))

            out_a[i] = fa + gate * lateral(supp_a)
            out_b[i] = fb + gate * lateral(supp_b)

        return out_a, out_b
