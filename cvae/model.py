"""
VectorMapCVAE – Conditional VAE for trajectory generation with vector-map
encoding and social attention.

Supports both single-state (OB_HORIZON=0) and multi-state (OB_HORIZON>=1)
observation encoders.  The decoder is autoregressive with dynamic map
cross-attention and social attention recomputed at every decode step.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from .layers import PointNetPolylineEncoder, FourierEmbedding, build_mlps
except ImportError:  # Allow running this file directly during development.
    from layers import PointNetPolylineEncoder, FourierEmbedding, build_mlps


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class DecoderZH(nn.Module):
    """Decode displacement from (z, h)."""
    def __init__(self, z_dim, hidden_dim, embed_dim, output_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, embed_dim),
            nn.ReLU6(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU6(),
        )
        self.mu = nn.Linear(embed_dim, output_dim)

    def forward(self, z, h):
        return self.mu(self.embed(torch.cat((z, h), -1)))


class P_Z(nn.Module):
    """Prior p(z | h_fx)."""
    def __init__(self, hidden_dim, embed_dim, z_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.ReLU6(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU6(),
        )
        self.mu = nn.Linear(embed_dim, z_dim)
        self.std = nn.Sequential(nn.Linear(embed_dim, z_dim), nn.Softplus())

    def forward(self, x):
        x = self.embed(x)
        return torch.distributions.Normal(self.mu(x), self.std(x))


class Q_Z(nn.Module):
    """Posterior q(z | h_fx, h_by)."""
    def __init__(self, hidden_dim_fy, hidden_dim_by, embed_dim, z_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim_fy + hidden_dim_by, embed_dim), nn.ReLU6(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU6(),
        )
        self.mu = nn.Linear(embed_dim, z_dim)
        self.std = nn.Sequential(nn.Linear(embed_dim, z_dim), nn.Softplus())

    def forward(self, x, y):
        xy = self.embed(torch.cat((x, y), -1))
        return torch.distributions.Normal(self.mu(xy), self.std(xy))


class EmbedZDMS(nn.Module):
    """Embed (z, d, map_ctx, social_ctx) for GRU input at each decoder step."""
    def __init__(self, z_dim, d_dim, map_ctx_dim, social_ctx_dim, output_dim):
        super().__init__()
        in_dim = z_dim + d_dim + map_ctx_dim + social_ctx_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, output_dim), nn.ReLU6(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, z, d, map_ctx, social_ctx):
        return self.net(torch.cat([z, d, map_ctx, social_ctx], -1))


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (used for social and map cross-attention)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, query, key, value, mask=None):
        B, Lq = query.size(0), query.size(1)
        Lk = key.size(1)
        Q = self.W_q(query).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = attn.matmul(V).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.W_o(ctx), attn.mean(dim=1)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class VectorMapCVAE(nn.Module):
    """
    CVAE for trajectory generation with vector-map encoding.

    Supports OB_HORIZON=0 (single-state MLP encoder) and OB_HORIZON>=1
    (multi-state GRU encoder).  The decoder is autoregressive with dynamic
    map cross-attention and social attention recomputed at every step.

    Parameters
    ----------
    horizon : int
        Prediction horizon (number of decode steps).
    d_model : int
        Hidden dimension for map tokens and attention.
    ob_radius : float
        Neighbour observation radius (metres).
    map_feature_dim : int
        Input dimension per map vector.
    max_vectors : int
        Max vectors per polyline.
    n_attention_heads : int
        Number of attention heads.
    num_freq_bands : int
        Fourier embedding frequency bands.
    dt : float
        Time-step duration (seconds).
    z_dim : int
        Latent dimension.
    hidden_dim_fx : int
        Forward encoder hidden dimension.
    hidden_dim_fy : int
        Decoder GRU hidden dimension.
    hidden_dim_by : int
        Backward encoder hidden dimension.
    feature_dim : int
        Social attention feature dimension.
    self_embed_dim : int
        Ego self-embedding dimension.
    neighbor_embed_dim : int
        Neighbour embedding dimension.
    gru_input_dim : int
        Decoder GRU input dimension.
    use_acc : bool
        Whether to include acceleration in the ego state vector.
    """

    def __init__(
        self,
        horizon: int = 10,
        d_model: int = 256,
        ob_radius: float = 50.0,
        map_feature_dim: int = 27,
        max_vectors: int = 128,
        n_attention_heads: int = 8,
        num_freq_bands: int = 64,
        dt: float = 0.5,
        z_dim: int = 32,
        hidden_dim_fx: int = 512,
        hidden_dim_fy: int = 512,
        hidden_dim_by: int = 256,
        feature_dim: int = 256,
        self_embed_dim: int = 128,
        neighbor_embed_dim: int = 128,
        gru_input_dim: int = 128,
        use_acc: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.ob_radius = ob_radius
        self.d_model = d_model
        self.dt = dt
        self.use_acc = use_acc

        d_dim = 2
        map_ctx_dim = d_model
        social_ctx_dim = neighbor_embed_dim

        self._hidden_dim_fy = hidden_dim_fy
        self._social_ctx_dim = social_ctx_dim

        # ====================== Map Encoder ======================
        self.map_polyline_encoder = PointNetPolylineEncoder(
            in_channels=map_feature_dim,
            hidden_dim=d_model,
            num_layers=3,
            num_pre_layers=1,
            out_channels=d_model,
        )
        self.map_relation_embed = FourierEmbedding(
            input_dim=3,
            hidden_dim=d_model,
            num_freq_bands=num_freq_bands,
        )
        self.map_cross_attn = MultiHeadAttention(d_model, n_attention_heads, dropout=0.1)
        self.map_ctx_proj_enc = nn.Linear(d_model, hidden_dim_fx // 2)
        self.map_ctx_proj_dec = nn.Linear(d_model, map_ctx_dim)

        # ====================== Observation Encoder ======================
        _s_input = 4 if use_acc else 2
        self.embed_s = nn.Sequential(
            nn.Linear(_s_input, 64), nn.ReLU6(),
            nn.Linear(64, self_embed_dim),
        )
        _n_input = 4
        self.embed_n = nn.Sequential(
            nn.Linear(_n_input, 64), nn.ReLU6(),
            nn.Linear(64, neighbor_embed_dim), nn.ReLU6(),
            nn.Linear(neighbor_embed_dim, neighbor_embed_dim),
        )
        self.embed_neighbor_features = nn.Sequential(
            nn.Linear(3, feature_dim), nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim), nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.embed_ego_query = nn.Sequential(
            nn.Linear(hidden_dim_fx, feature_dim), nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.social_attention_module = MultiHeadAttention(feature_dim, n_attention_heads, dropout=0.1)
        self.attention_projection = nn.Sequential(nn.Linear(feature_dim, neighbor_embed_dim), nn.ReLU6())

        # GRU encoder (observation)
        self.rnn_fx = nn.GRU(self_embed_dim + neighbor_embed_dim, hidden_dim_fx)
        init_dim = hidden_dim_fx // 2
        self.rnn_fx_init = nn.Sequential(
            nn.Linear(2, init_dim), nn.ReLU6(),
            nn.Linear(init_dim, init_dim * self.rnn_fx.num_layers), nn.ReLU6(),
            nn.Linear(init_dim * self.rnn_fx.num_layers, init_dim * self.rnn_fx.num_layers),
        )
        self.rnn_fx_init_map = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU6(),
            nn.Linear(d_model, d_model), nn.ReLU6(),
            nn.Linear(d_model, (hidden_dim_fx - init_dim) * self.rnn_fx.num_layers),
        )

        # GRU encoder (backward, for training posterior)
        self.rnn_by = nn.GRU(self_embed_dim + neighbor_embed_dim, hidden_dim_by)

        # ====================== Latent Space ======================
        self.q_z = Q_Z(hidden_dim_fy, hidden_dim_by, hidden_dim_fy, z_dim)
        self.p_z = P_Z(hidden_dim_fy, hidden_dim_fy, z_dim)

        # ====================== Decoder ======================
        self.dec = DecoderZH(z_dim, hidden_dim_fy, hidden_dim_fy, d_dim)
        self.embed_zdms = EmbedZDMS(z_dim, d_dim, map_ctx_dim, social_ctx_dim, gru_input_dim)
        self.rnn_fy = nn.GRU(gru_input_dim, hidden_dim_fy)
        self.rnn_fy_init = nn.Sequential(
            nn.Linear(hidden_dim_fx, hidden_dim_fy * self.rnn_fy.num_layers), nn.ReLU6(),
            nn.Linear(hidden_dim_fy * self.rnn_fy.num_layers,
                      hidden_dim_fy * self.rnn_fy.num_layers),
        )
        self.dec_map_query = nn.Sequential(
            nn.Linear(hidden_dim_fy + 2, d_model), nn.ReLU6(),
            nn.Linear(d_model, d_model),
        )

        # ====================== Decoder Social Attention ======================
        self.dec_neighbor_embed = nn.Sequential(
            nn.Linear(_n_input, feature_dim), nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim), nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.dec_social_query = nn.Sequential(
            nn.Linear(hidden_dim_fy, feature_dim), nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.dec_social_attention = MultiHeadAttention(feature_dim, n_attention_heads, dropout=0.1)
        self.social_ctx_proj_dec = nn.Sequential(
            nn.Linear(feature_dim, social_ctx_dim), nn.ReLU6(),
        )

        # ====================== Single-State Fusion (OB_HORIZON=0) ======================
        _fusion_dim = hidden_dim_fx + self_embed_dim + neighbor_embed_dim
        self.single_state_fusion = nn.Sequential(
            nn.Linear(_fusion_dim, hidden_dim_fx),
            nn.ReLU6(),
            nn.Linear(hidden_dim_fx, hidden_dim_fx),
        )

    # ------------------------------------------------------------------
    # Map Encoding
    # ------------------------------------------------------------------
    def encode_map(self, map_feature, map_mask, map_position, map_heading):
        """
        Args
            map_feature:  (B, M, V, feat_dim)
            map_mask:     (B, M, V) bool
            map_position: (B, M, 3)
            map_heading:  (B, M)
        Returns
            map_tokens:   (B, M, d_model)
            map_valid:    (B, M) bool
        """
        map_tokens = self.map_polyline_encoder(map_feature, map_mask)
        map_valid = map_mask.any(dim=-1)
        return map_tokens, map_valid

    def _compute_map_kv(self, map_tokens, map_position, map_heading,
                        query_pos, query_heading):
        """
        Compute position-aware map key/value by adding Fourier relative-
        position embeddings.  Called at every decode step with the *current*
        agent position.
        """
        B, M, D = map_tokens.shape

        if query_pos.shape[-1] == 2:
            qp3 = torch.cat([query_pos, query_pos.new_zeros(B, 1)], dim=-1)
        else:
            qp3 = query_pos
        dx = map_position[..., 0] - qp3[:, 0:1]
        dy = map_position[..., 1] - qp3[:, 1:2]
        dh = map_heading - query_heading.unsqueeze(-1)
        rel = torch.stack([dx, dy, dh], dim=-1)

        rel_flat = rel.reshape(B * M, 3)
        rel_emb = self.map_relation_embed(rel_flat).reshape(B, M, D)

        return map_tokens + rel_emb

    def _map_cross_attn_query(self, agent_query, kv, map_valid):
        """Execute cross-attention with an explicit agent query."""
        q = agent_query.unsqueeze(1)
        mask = map_valid.unsqueeze(1)
        ctx, _ = self.map_cross_attn(q, kv, kv, mask=mask)
        ctx = ctx.squeeze(1)
        return ctx.nan_to_num(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Social attention (encoder)
    # ------------------------------------------------------------------
    def social_attention(self, ego_query, social_features, mask):
        ego_query = ego_query.unsqueeze(1)
        attended, _ = self.social_attention_module(
            ego_query, social_features, social_features,
            mask=mask.unsqueeze(1),
        )
        attended = attended.nan_to_num(0.0, 0.0, 0.0)
        return attended.squeeze(1)

    # ------------------------------------------------------------------
    # Observation encoder
    # ------------------------------------------------------------------
    def enc(self, x, neighbor, *, y=None, map_tokens=None, map_kv=None,
            map_valid=None, seq_len=None):
        """
        Encode observation (and optionally the future for training).

        Args
            x:          (L1+1, N, 6)  – ego history states [x, y, vx, vy, ax, ay]
            neighbor:   (L+1, N, Nn, 6)
            y:          (L2, N, 2) or None
            map_tokens: (N, M, d_model)
            map_kv:     (N, M, d_model) – map tokens + relation embedding
            map_valid:  (N, M) bool
        Returns
            h_fx: (N, hidden_dim_fx)
            h_by: (L2, N, hidden_dim_by) [only if y is not None]
        """
        use_map = map_kv is not None

        with torch.no_grad():
            L1 = x.size(0) - 1
            N = neighbor.size(1)
            Nn = neighbor.size(2)
            state = x

            pos = state[..., :2]
            if y is not None:
                L2 = y.size(0)
                pos = torch.cat((pos, y), 0)
            else:
                L2 = 0

            v = pos[1:] - pos[:-1]
            a = v[1:] - v[:-1]
            if v.size(0) > 0:
                a = torch.cat((state[-1:, ..., 4:6], a))

            neighbor_x = neighbor[..., :2]
            neighbor_v = neighbor[1:, ..., 2:4] * self.dt

            dp = neighbor_x - pos.unsqueeze(-2)
            dv = neighbor_v - v.unsqueeze(-2)

            dist = dp.norm(dim=-1)
            mask = dist <= self.ob_radius
            dp0, mask0 = dp[0], mask[0]
            dp, mask = dp[1:], mask[1:]
            dist = dist[1:]

            dot_dp_v = (dp @ v.unsqueeze(-1)).squeeze(-1)
            bearing = dot_dp_v / (dist * v.norm(dim=-1).unsqueeze(-1))
            bearing = bearing.nan_to_num(0, 0, 0)
            dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).view(dp.size(0), N, Nn)
            tau = -dot_dp_dv / dv.norm(dim=-1)
            tau = tau.nan_to_num(0, 0, 0).clip(0, 7)
            mpd = (dp + tau.unsqueeze(-1) * dv).norm(dim=-1)
            features = torch.stack((dist, bearing, mpd), -1)

        k = self.embed_neighbor_features(features)
        s = self.embed_s(torch.cat((v, a), -1) if self.use_acc else v)
        n = self.embed_n(torch.cat((dp, dv), -1))

        # ==============================================================
        # Forward encoder: single-state (L1 == 0)  vs  GRU (L1 >= 1)
        # ==============================================================
        if L1 == 0:
            with torch.no_grad():
                vel_disp = state[0, :, 2:4]
            if self.use_acc:
                with torch.no_grad():
                    accel = state[0, :, 4:6]
                s_feat = self.embed_s(torch.cat([vel_disp, accel], -1))
            else:
                s_feat = self.embed_s(vel_disp)

            with torch.no_grad():
                n_vel_0 = neighbor[0, :, :, 2:4] * self.dt
                dv_0 = n_vel_0 - vel_disp.unsqueeze(1)
                dist_0 = dp0.norm(dim=-1)
                dot0 = (dp0 * vel_disp.unsqueeze(1)).sum(-1)
                v_norm = vel_disp.norm(dim=-1, keepdim=True)
                bearing_0 = dot0 / (dist_0 * v_norm).clamp(min=1e-8)
                bearing_0 = bearing_0.nan_to_num(0, 0, 0)
                dot_dv0 = (dp0 * dv_0).sum(-1)
                tau_0 = -dot_dv0 / dv_0.norm(dim=-1).clamp(min=1e-8)
                tau_0 = tau_0.nan_to_num(0, 0, 0).clip(0, 7)
                mpd_0 = (dp0 + tau_0.unsqueeze(-1) * dv_0).norm(dim=-1)
                feat_0 = torch.stack((dist_0, bearing_0, mpd_0), -1)

            k_0 = self.embed_neighbor_features(feat_0)

            h_dp = self.rnn_fx_init(dp0)
            h_dp = (mask0.unsqueeze(-1) * h_dp).sum(-2)
            h_dp = h_dp.view(N, -1, self.rnn_fx.num_layers)

            if use_map:
                map_global = (map_kv * map_valid.unsqueeze(-1).float()).sum(dim=1)
                denom = map_valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                map_global = map_global / denom
                h_map = self.rnn_fx_init_map(map_global)
                h_map = h_map.view(N, -1, self.rnn_fx.num_layers)
                h_init = torch.cat((h_dp, h_map), 1)
            else:
                pad = h_dp.new_zeros(N, h_dp.size(1), self.rnn_fx.num_layers)
                h_init = torch.cat((h_dp, pad), 1)

            h_init = h_init.squeeze(-1)

            q = self.embed_ego_query(h_init)
            attended = self.social_attention(q, k_0, mask0)
            n_feat = self.attention_projection(attended)

            h_fx = self.single_state_fusion(
                torch.cat([h_init, s_feat, n_feat], -1)
            )
        else:
            h_dp = self.rnn_fx_init(dp0)
            h_dp = (mask0.unsqueeze(-1) * h_dp).sum(-2)
            h_dp = h_dp.view(N, -1, self.rnn_fx.num_layers)

            if use_map:
                map_global = (map_kv * map_valid.unsqueeze(-1).float()).sum(dim=1)
                denom = map_valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                map_global = map_global / denom
                h_map = self.rnn_fx_init_map(map_global)
                h_map = h_map.view(N, -1, self.rnn_fx.num_layers)
                h = torch.cat((h_dp, h_map), 1)
            else:
                pad = h_dp.new_zeros(N, h_dp.size(1), self.rnn_fx.num_layers)
                h = torch.cat((h_dp, pad), 1)

            h = h.permute(2, 0, 1).contiguous()

            H = []
            for t in range(L1):
                q = self.embed_ego_query(h[-1])
                attended_neighbors = self.social_attention(q, k[t], mask[t])
                attended_neighbors = self.attention_projection(attended_neighbors)
                x_t = torch.cat((attended_neighbors, s[t]), -1).unsqueeze(0)
                _, h = self.rnn_fx(x_t, h)
                H.append(h[-1])

            if seq_len is None:
                h_fx = H[-1]
            else:
                dyn_mask = torch.arange(L1, device=seq_len.device).unsqueeze(0) == (seq_len - 2).unsqueeze(-1)
                h_fx = torch.stack(H, 1)[dyn_mask]

        if y is None:
            return h_fx

        # Backward encoder
        mask_t = mask[L1:L1 + L2].unsqueeze(-1)
        n_t = n[L1:L1 + L2]
        n_t = (mask_t * n_t).sum(-2)
        s_t = s[L1:L1 + L2]
        x_t = torch.cat((n_t, s_t), -1)
        x_t = torch.flip(x_t, (0,))
        b, _ = self.rnn_by(x_t)
        if self.rnn_by.num_layers > 1:
            b = b[..., -b.size(-1) // self.rnn_by.num_layers:]
        b = torch.flip(b, (0,))
        return h_fx, b

    # ------------------------------------------------------------------
    # Decoder map cross-attention (dynamic K/V at each step)
    # ------------------------------------------------------------------
    def _decoder_map_ctx(self, h_fy, current_pos, current_heading,
                         map_tokens, map_position, map_heading, map_valid):
        """
        Compute map context for the current decode step.  The Fourier
        relative-position embedding is recomputed from the agent's *current
        predicted position*, so K/V reflects the changing distance to road
        boundaries.
        """
        kv = self._compute_map_kv(map_tokens, map_position, map_heading,
                                  current_pos, current_heading)
        q_input = torch.cat([h_fy, current_pos], dim=-1)
        q = self.dec_map_query(q_input)
        ctx = self._map_cross_attn_query(q, kv, map_valid)
        return self.map_ctx_proj_dec(ctx)

    # ------------------------------------------------------------------
    # Decoder social attention (at each step)
    # ------------------------------------------------------------------
    def _decoder_social_ctx(self, h_fy, current_pos, current_vel,
                            neighbor_pos_t, neighbor_vel_t, neighbor_mask_t):
        """
        Compute social context for the current decode step by attending to
        neighbour vehicles at the corresponding time step.
        """
        dp = neighbor_pos_t - current_pos.unsqueeze(1)
        n_vel_adj = neighbor_vel_t * self.dt
        dv = n_vel_adj - current_vel.unsqueeze(1)
        rel_feat = torch.cat([dp, dv], dim=-1)

        dist = dp.norm(dim=-1)
        mask = neighbor_mask_t & (dist <= self.ob_radius)

        k = self.dec_neighbor_embed(rel_feat)

        q = self.dec_social_query(h_fy)
        q = q.unsqueeze(1)

        attended, _ = self.dec_social_attention(
            q, k, k, mask=mask.unsqueeze(1),
        )
        attended = attended.squeeze(1).nan_to_num(0.0, 0.0, 0.0)

        return self.social_ctx_proj_dec(attended)

    # ------------------------------------------------------------------
    # Forward (unified train / inference)
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        self.rnn_fx.flatten_parameters()
        self.rnn_fy.flatten_parameters()

        _sentinel = object()

        def _next_or(it, key, default=_sentinel):
            val = kwargs.get(key, _sentinel)
            if val is not _sentinel:
                return val
            try:
                return next(it)
            except StopIteration:
                if default is _sentinel:
                    raise ValueError(f"Missing required argument: {key}")
                return default

        if self.training:
            self.rnn_by.flatten_parameters()
            args_iter = iter(args)
            x = _next_or(args_iter, "x")
            y = _next_or(args_iter, "y")
            neighbor = _next_or(args_iter, "neighbor")
            map_feature = kwargs.get("map_feature", None)
            map_mask = kwargs.get("map_mask", None)
            map_position = kwargs.get("map_position", None)
            map_heading = kwargs.get("map_heading", None)
            seq_len = kwargs.get("seq_len", None)
            scheduled_sampling_ratio = kwargs.get("scheduled_sampling_ratio", 0.0)
            return self.learn(x, y, neighbor,
                              map_feature=map_feature, map_mask=map_mask,
                              map_position=map_position, map_heading=map_heading,
                              seq_len=seq_len,
                              scheduled_sampling_ratio=scheduled_sampling_ratio)

        args_iter = iter(args)
        x = _next_or(args_iter, "x")
        neighbor = _next_or(args_iter, "neighbor", None)
        map_feature = kwargs.get("map_feature", None)
        map_mask = kwargs.get("map_mask", None)
        map_position = kwargs.get("map_position", None)
        map_heading = kwargs.get("map_heading", None)
        seq_len = kwargs.get("seq_len", None)
        n_predictions = kwargs.get("n_predictions", 0)
        stochastic = n_predictions > 0

        if neighbor is None:
            sh = list(x.shape)
            sh.insert(-1, 0)
            neighbor = torch.empty(sh, dtype=x.dtype, device=x.device)

        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1)
        N = x.size(1)
        L1 = x.size(0) - 1

        # --- Encode map ---
        map_tokens, map_valid = None, None
        has_map = map_feature is not None and map_mask is not None
        if has_map:
            map_tokens, map_valid = self.encode_map(map_feature, map_mask,
                                                     map_position, map_heading)
            if seq_len is None:
                enc_pos = x[-1, :, :2]
            else:
                enc_pos = x.gather(0, (seq_len - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2)).squeeze(0)
            if x.size(0) >= 2:
                _dp_enc = x[-1, :, :2] - x[-2, :, :2]
                enc_heading = torch.atan2(_dp_enc[:, 1], _dp_enc[:, 0])
            else:
                enc_heading = torch.atan2(x[0, :, 3], x[0, :, 2])
            map_kv_enc = self._compute_map_kv(map_tokens, map_position, map_heading,
                                              enc_pos, enc_heading)
        else:
            map_kv_enc = None

        # --- Encode observation ---
        neighbor_enc = neighbor[:x.size(0)]
        h_fx = self.enc(x, neighbor_enc,
                        map_tokens=map_tokens if has_map else None,
                        map_kv=map_kv_enc, map_valid=map_valid, seq_len=seq_len)

        # Current position tracker
        if seq_len is None:
            x_T = x[-1, ..., :2].unsqueeze(0)
        else:
            x_T = x.gather(0, (seq_len - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2))

        # --- Autoregressive Decoder ---
        current_pos = x_T.squeeze(0)

        h = self.rnn_fy_init(h_fx)
        h = h.view(N, -1, self.rnn_fy.num_layers).permute(2, 0, 1)
        if stochastic:
            h = h.repeat(1, n_predictions, 1)
            if has_map:
                map_tokens = map_tokens.repeat(n_predictions, 1, 1)
                map_position = map_position.repeat(n_predictions, 1, 1)
                map_heading = map_heading.repeat(n_predictions, 1)
                map_valid = map_valid.repeat(n_predictions, 1)
        h = h.contiguous()

        if x.size(0) >= 2:
            dp_init = x[-1, :, :2] - x[-2, :, :2]
            current_heading = torch.atan2(dp_init[:, 1], dp_init[:, 0])
        else:
            current_heading = torch.atan2(x[0, :, 3], x[0, :, 2])

        current_vel = x[-1, :, 2:4]

        if stochastic:
            current_pos = current_pos.repeat(n_predictions, 1)
            current_heading = current_heading.repeat(n_predictions)
            current_vel = current_vel.repeat(n_predictions, 1)

        L_neigh = neighbor.size(0)
        _decode_steps = kwargs.get('decode_steps', self.horizon)

        D = []
        for t in range(_decode_steps):
            p_z = self.p_z(h[-1])
            z = p_z.sample() if stochastic else p_z.mean
            d = self.dec(z, h[-1])

            D.append(d)
            if t == _decode_steps - 1:
                break

            if has_map:
                map_ctx = self._decoder_map_ctx(
                    h[-1], current_pos, current_heading,
                    map_tokens, map_position, map_heading, map_valid,
                )
            else:
                map_ctx = h[-1].new_zeros(h[-1].size(0), self.d_model)

            neigh_idx = L1 + t + 1
            if neigh_idx < L_neigh:
                neigh_t = neighbor[neigh_idx]
                if stochastic:
                    neigh_t = neigh_t.repeat(n_predictions, 1, 1)
                n_pos_t = neigh_t[..., :2]
                n_vel_t = neigh_t[..., 2:4]
                n_mask_t = n_pos_t.norm(dim=-1) < 1e8
                social_ctx = self._decoder_social_ctx(
                    h[-1], current_pos, current_vel,
                    n_pos_t, n_vel_t, n_mask_t,
                )
            else:
                social_ctx = h[-1].new_zeros(h[-1].size(0), self._social_ctx_dim)

            zdms = self.embed_zdms(z, d, map_ctx, social_ctx)
            _, h = self.rnn_fy(zdms.unsqueeze(0), h)

            current_pos = current_pos + d
            new_heading = torch.atan2(d[:, 1], d[:, 0])
            moving = d.norm(dim=-1) > 1e-4
            current_heading = torch.where(moving, new_heading, current_heading)
            current_vel = -d

        d = torch.stack(D)
        pred = torch.cumsum(d, 0)
        if stochastic:
            pred = pred.view(pred.size(0), n_predictions, -1, pred.size(-1)).permute(1, 0, 2, 3)
        pred = pred + x_T
        if C < 3:
            pred = pred.squeeze(-2) if pred.dim() > 3 else pred.squeeze(1)
        return pred

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def learn(self, x, y, neighbor=None, *, map_feature=None, map_mask=None,
              map_position=None, map_heading=None, seq_len=None,
              scheduled_sampling_ratio=0.0):
        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1) if neighbor is not None else None
            if y is not None:
                y = y.unsqueeze(1)
        N = x.size(1)
        L1 = x.size(0) - 1

        if y is not None and y.size(0) != self.horizon:
            print(f"[Warn] Unmatched seq length: y={y.size(0)} vs horizon={self.horizon}")

        # --- Encode map ---
        map_tokens, map_valid = None, None
        has_map = map_feature is not None and map_mask is not None
        if has_map:
            map_tokens, map_valid = self.encode_map(map_feature, map_mask,
                                                     map_position, map_heading)
            if seq_len is None:
                enc_pos = x[-1, :, :2]
            else:
                enc_pos = x.gather(0, (seq_len - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2)).squeeze(0)
            if x.size(0) >= 2:
                _dp_enc = x[-1, :, :2] - x[-2, :, :2]
                enc_heading = torch.atan2(_dp_enc[:, 1], _dp_enc[:, 0])
            else:
                enc_heading = torch.atan2(x[0, :, 3], x[0, :, 2])
            map_kv_enc = self._compute_map_kv(map_tokens, map_position, map_heading,
                                              enc_pos, enc_heading)
        else:
            map_kv_enc = None

        # --- Encode observation + backward ---
        h_fx, b = self.enc(x, neighbor, y=y,
                           map_tokens=map_tokens if has_map else None,
                           map_kv=map_kv_enc, map_valid=map_valid, seq_len=seq_len)

        # Current position tracker
        if seq_len is None:
            x_T = x[-1, ..., :2].unsqueeze(0)
        else:
            x_T = x.gather(0, (seq_len - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 2))

        # --- Autoregressive Decoder ---
        current_pos = x_T.squeeze(0)

        h = self.rnn_fy_init(h_fx)
        h = h.view(N, -1, self.rnn_fy.num_layers).permute(2, 0, 1).contiguous()

        if x.size(0) >= 2:
            dp_init = x[-1, :, :2] - x[-2, :, :2]
            current_heading = torch.atan2(dp_init[:, 1], dp_init[:, 0])
        else:
            current_heading = torch.atan2(x[0, :, 3], x[0, :, 2])

        current_vel = x[-1, :, 2:4]

        with torch.no_grad():
            gt_disps = torch.zeros_like(y)
            gt_disps[0] = y[0] - x_T.squeeze(0)
            if y.size(0) > 1:
                gt_disps[1:] = y[1:] - y[:-1]
            gt_headings = torch.atan2(gt_disps[..., 1], gt_disps[..., 0])

        L_neigh = neighbor.size(0)

        P, Q, D_list, Z = [], [], [], []
        for t in range(self.horizon):
            p_z = self.p_z(h[-1])
            q_z = self.q_z(h[-1], b[t])
            z = q_z.rsample()
            d = self.dec(z, h[-1])

            P.append(p_z)
            Q.append(q_z)
            D_list.append(d)
            Z.append(z)

            if t == self.horizon - 1:
                break

            use_model_pred = (scheduled_sampling_ratio > 0.0 and
                              torch.rand(1).item() < scheduled_sampling_ratio)

            # Dynamic map cross-attention
            if has_map:
                map_ctx = self._decoder_map_ctx(
                    h[-1], current_pos, current_heading,
                    map_tokens, map_position, map_heading, map_valid,
                )
            else:
                map_ctx = h[-1].new_zeros(h[-1].size(0), self.d_model)

            # Decoder social attention
            neigh_idx = L1 + t + 1
            if neigh_idx < L_neigh:
                neigh_t = neighbor[neigh_idx]
                n_pos_t = neigh_t[..., :2]
                n_vel_t = neigh_t[..., 2:4]
                n_mask_t = n_pos_t.norm(dim=-1) < 1e8
                social_ctx = self._decoder_social_ctx(
                    h[-1], current_pos, current_vel,
                    n_pos_t, n_vel_t, n_mask_t,
                )
            else:
                social_ctx = h[-1].new_zeros(h[-1].size(0), self._social_ctx_dim)

            zdms = self.embed_zdms(z, d, map_ctx, social_ctx)
            _, h = self.rnn_fy(zdms.unsqueeze(0), h)

            # Scheduled sampling: position / heading / velocity update
            if use_model_pred:
                pred_pos = current_pos + d.detach()
                current_pos = pred_pos
                current_vel = -d.detach()
                d_norm = d.detach().norm(dim=-1)
                d_heading = torch.atan2(d.detach()[:, 1], d.detach()[:, 0])
                moving = d_norm > 1e-4
                current_heading = torch.where(moving, d_heading, current_heading)
            else:
                current_pos = y[t]
                current_vel = -gt_disps[t]
                gt_step_norm = gt_disps[t].norm(dim=-1)
                moving = gt_step_norm > 1e-4
                current_heading = torch.where(moving, gt_headings[t], current_heading)

        d = torch.stack(D_list)
        with torch.no_grad():
            y_rel = y - x_T
        pred = torch.cumsum(d, 0)

        err = (pred - y_rel).square()
        kl = []
        for p, q, z in zip(P, Q, Z):
            kl.append(q.log_prob(z) - p.log_prob(z))
        kl = torch.stack(kl)

        return err, kl

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(self, err, kl, beta=1.0, time_weight_exp=0.0, free_bits=0.0):
        """
        Compute total loss = reconstruction + β * KL.

        Args
            err:              (H, N, 2) squared displacement errors
            kl:               (H, N, z_dim) KL divergence per step
            beta:             KL weight (β-VAE)
            time_weight_exp:  exponential time weighting (0 = uniform)
            free_bits:        minimum KL per dimension (0 = no free bits)
        """
        H = err.size(0)
        rec_per_point = err

        if time_weight_exp > 0.0:
            t_idx = torch.arange(H, device=err.device, dtype=err.dtype)
            weights = (1.0 + t_idx / H) ** time_weight_exp
            weights = weights / weights.mean()
            weights = weights.view(H, 1, 1)
            rec_per_point = rec_per_point * weights

        rec = rec_per_point.mean()

        if free_bits > 0.0:
            kl_val = kl.clamp(min=free_bits).mean()
        else:
            kl_val = kl.mean()

        total = beta * kl_val + rec
        return {"loss": total, "rec": rec, "kl": kl_val}
