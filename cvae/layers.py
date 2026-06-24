"""
Layers ported from ADV-BMT for VectorMap-CVAE.
- build_mlps: common MLP builder
- PointNetPolylineEncoder: vector map feature encoder
- FourierEmbedding: relative positional encoding for map-agent interactions
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# build_mlps  (from bmt/models/layers/common_layers.py)
# ---------------------------------------------------------------------------
def build_mlps(c_in, mlp_channels, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)
    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k]))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k]), nn.ReLU()])
            else:
                layers.extend([
                    nn.Linear(c_in, mlp_channels[k], bias=False),
                    nn.LayerNorm(mlp_channels[k]),
                    nn.ReLU(),
                ])
            c_in = mlp_channels[k]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# PointNetPolylineEncoder  (from bmt/models/layers/polyline_encoder.py)
# ---------------------------------------------------------------------------
class PointNetPolylineEncoder(nn.Module):
    """Encode variable-length polylines into fixed-size feature vectors.

    Input
    -----
    polylines : (B, M, V, C)   – M polylines, each with V vectors of dim C
    polylines_mask : (B, M, V) – boolean validity mask

    Output
    ------
    feature : (B, M, D)
    """

    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1,
                 out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False,
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False,
        )
        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim,
                mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True,
                without_norm=True,
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        B, M, V, C = polylines.shape

        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])
        polylines_feature = polylines_feature_valid.new_zeros(B, M, V, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # global feature via max-pool over vectors
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat(
            (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, V, 1)), dim=-1
        )

        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(B, M, V, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # second max-pool -> (B, M, D)
        feature_buffers = feature_buffers.max(dim=2)[0]

        if self.out_mlps is not None:
            valid_mask = polylines_mask.sum(dim=-1) > 0
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(B, M, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid.to(polylines_feature.dtype)

        return feature_buffers


# ---------------------------------------------------------------------------
# FourierEmbedding  (from bmt/models/layers/fourier_embedding.py)
# ---------------------------------------------------------------------------
class FourierEmbedding(nn.Module):
    """Learnable Fourier embedding for continuous inputs (e.g. relative position)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(input_dim)
        ])
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[..., i, :])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)

