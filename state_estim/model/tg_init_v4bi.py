"""
StateEstim Init Model — Categorical v4 with Bias Init (V4BI).

Categorical discretisation of all continuous attributes:
  - Heading:  144 bins in [-π/2, π/2]
  - Speed:    zero-inflated + 60 log-scale bins in [0, 30] m/s
  - Position: 40 bins each (longitudinal, lateral), optional NUB
  - Size:     60 length bins [0,15m], 40 width bins [0,4m]

Loaded from the StateEstim training codebase checkpoint.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from random import choices
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

from model.model_utils import MLP_3, CG_stacked
from utils.data_utils import WaymoAgent, get_agent_pos_from_vec

copy_func = copy.deepcopy


class initializer_categorical_v4(pl.LightningModule):
    """StateEstim initializer — categorical v4 (V4BI)."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.hidden_dim = cfg['hidden_dim']
        self.CG_agent = CG_stacked(5, self.hidden_dim)
        self.CG_line = CG_stacked(5, self.hidden_dim)
        self.agent_encode = MLP_3([17, 256, 512, self.hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, self.hidden_dim])
        self.type_embedding = nn.Embedding(20, self.hidden_dim)
        self.traf_embedding = nn.Embedding(4, self.hidden_dim)

        middle_layer_shape = [self.hidden_dim * 2, self.hidden_dim, 256]

        self.prob_head = MLP_3([*middle_layer_shape, 1])

        self.n_heading_bins = cfg.get('n_heading_bins', 144)
        self.heading_min = -math.pi / 2
        self.heading_max = math.pi / 2
        self.heading_head = MLP_3([*middle_layer_shape, self.n_heading_bins])

        self.n_speed_bins = cfg.get('n_speed_bins', 60)
        self.speed_min = 0.0
        self.speed_max = cfg.get('max_speed', 30.0)
        self.speed_stopped_threshold = cfg.get('speed_stopped_threshold', 0.5)
        self.speed_stopped_head = MLP_3([*middle_layer_shape, 1])
        self.speed_head = MLP_3([*middle_layer_shape, self.n_speed_bins])

        self._init_speed_bias(cfg)

        self.log_speed_min = 0.0
        self.log_speed_max = math.log(1 + self.speed_max)

        self.n_pos_bins = cfg.get('n_pos_bins', 40)
        self.pos_min = -0.5
        self.pos_max = 0.5
        self.pos_temperature = cfg.get('pos_temperature', 1.0)
        self.pos_mode = cfg.get('pos_mode', 'independent')
        self.pos_bin_concentration = cfg.get('pos_bin_concentration', 1.0)

        if self.pos_mode == 'joint':
            self.n_pos_joint_x = cfg.get('n_pos_joint_x', 12)
            self.n_pos_joint_y = cfg.get('n_pos_joint_y', 12)
            self.pos_joint_head = MLP_3(
                [*middle_layer_shape, self.n_pos_joint_x * self.n_pos_joint_y])
        else:
            self.pos_long_head = MLP_3([*middle_layer_shape, self.n_pos_bins])
            self.pos_lat_head = MLP_3([*middle_layer_shape, self.n_pos_bins])

        self.n_length_bins = cfg.get('n_length_bins', 60)
        self.length_min = 0.0
        self.length_max = cfg.get('max_length', 15.0)
        self.length_head = MLP_3([*middle_layer_shape, self.n_length_bins])

        self.n_width_bins = cfg.get('n_width_bins', 40)
        self.width_min = 0.0
        self.width_max = cfg.get('max_width', 4.0)
        self.width_head = MLP_3([*middle_layer_shape, self.n_width_bins])

        self.use_vel_heading = cfg.get('use_vel_heading', True)
        if self.use_vel_heading:
            self.vel_heading_head = MLP_3([*middle_layer_shape, 1])

        self.ls_heading = cfg.get('ls_heading', 0.1)
        self.ls_speed = cfg.get('ls_speed', 0.0)
        self.ls_size = cfg.get('ls_size', 0.1)
        self.ls_position = cfg.get('ls_position', 0.0)

        self.register_buffer('heading_centers', self._make_centers(
            self.heading_min, self.heading_max, self.n_heading_bins))
        self.register_buffer('speed_centers', self._make_log_speed_centers(
            self.speed_max, self.n_speed_bins))

        _mk = (self._make_concentrated_centers if self.pos_bin_concentration != 1.0
               else lambda a, b, n, c=None: self._make_centers(a, b, n))
        if self.pos_mode == 'joint':
            self.register_buffer('pos_x_centers', _mk(
                self.pos_min, self.pos_max, self.n_pos_joint_x, self.pos_bin_concentration))
            self.register_buffer('pos_y_centers', _mk(
                self.pos_min, self.pos_max, self.n_pos_joint_y, self.pos_bin_concentration))
        else:
            self.register_buffer('pos_centers', _mk(
                self.pos_min, self.pos_max, self.n_pos_bins, self.pos_bin_concentration))

        self.register_buffer('length_centers', self._make_centers(
            self.length_min, self.length_max, self.n_length_bins))
        self.register_buffer('width_centers', self._make_centers(
            self.width_min, self.width_max, self.n_width_bins))

        self._enable_head_freeze = cfg.get('enable_head_freeze', False)
        self._freeze_patience = cfg.get('head_freeze_patience', 10)

        if self.pos_mode == 'joint':
            self._HEAD_MAP = {
                'prob':        ('prob_head',           'prob_loss'),
                'heading':     ('heading_head',        'heading_loss'),
                'stopped':     ('speed_stopped_head',  'stopped_loss'),
                'speed':       ('speed_head',          'speed_loss'),
                'pos_joint':   ('pos_joint_head',      'pos_joint_loss'),
                'length':      ('length_head',         'length_loss'),
                'width':       ('width_head',          'width_loss'),
            }
        else:
            self._HEAD_MAP = {
                'prob':        ('prob_head',           'prob_loss'),
                'heading':     ('heading_head',        'heading_loss'),
                'stopped':     ('speed_stopped_head',  'stopped_loss'),
                'speed':       ('speed_head',          'speed_loss'),
                'pos_long':    ('pos_long_head',       'pos_long_loss'),
                'pos_lat':     ('pos_lat_head',        'pos_lat_loss'),
                'length':      ('length_head',         'length_loss'),
                'width':       ('width_head',          'width_loss'),
            }
        if self.use_vel_heading:
            self._HEAD_MAP['vel_heading'] = ('vel_heading_head', 'vel_heading_loss')
        self._frozen_heads: set = set()
        self._head_best_loss = {n: float('inf') for n in self._HEAD_MAP}
        self._head_patience_ctr = {n: 0 for n in self._HEAD_MAP}
        self._val_loss_accum = {}
        self._val_loss_count = 0

        self.use_uncertainty_weighting = cfg.get('use_uncertainty_weighting', False)
        if self.use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1)) for name in self._HEAD_MAP
            })

    def _init_speed_bias(self, cfg):
        init_stopped_logit = cfg.get('init_stopped_logit', 3.0)
        init_speed_logit = cfg.get('init_speed_logit', 5.0)
        init_heading_logit = cfg.get('init_heading_logit', 0.0)
        n_warm_bins = cfg.get('init_speed_warm_bins', 3)

        if init_stopped_logit > 0:
            last_layer = self.speed_stopped_head.mlp[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.constant_(last_layer.bias, init_stopped_logit)

        if init_speed_logit > 0:
            last_layer = self.speed_head.mlp[-1]
            with torch.no_grad():
                last_layer.bias.zero_()
                last_layer.bias[:n_warm_bins] = init_speed_logit

        if init_heading_logit > 0:
            center_bin = self.n_heading_bins // 2
            last_layer = self.heading_head.mlp[-1]
            with torch.no_grad():
                last_layer.bias.zero_()
                last_layer.bias[center_bin] = init_heading_logit

    @staticmethod
    def _make_centers(vmin, vmax, n_bins):
        bw = (vmax - vmin) / n_bins
        return torch.linspace(vmin + bw / 2, vmax - bw / 2, n_bins)

    @staticmethod
    def _make_concentrated_centers(vmin, vmax, n_bins, concentration):
        bw = (vmax - vmin) / n_bins
        t_centers = torch.linspace(vmin + bw / 2, vmax - bw / 2, n_bins)
        mid = (vmin + vmax) / 2
        half = (vmax - vmin) / 2
        t_norm = (t_centers - mid) / half
        c_norm = torch.sign(t_norm) * torch.abs(t_norm).pow(concentration)
        return mid + c_norm * half

    @staticmethod
    def _make_log_speed_centers(speed_max, n_bins):
        log_max = math.log(1 + speed_max)
        bw = log_max / n_bins
        log_centers = torch.linspace(bw / 2, log_max - bw / 2, n_bins)
        return torch.exp(log_centers) - 1

    # ── Bin conversion helpers ──

    def _val_to_bin(self, val, vmin, vmax, n_bins):
        bw = (vmax - vmin) / n_bins
        return ((val - vmin) / bw).long().clamp(0, n_bins - 1)

    def heading_to_bin(self, v):
        return self._val_to_bin(v, self.heading_min, self.heading_max, self.n_heading_bins)

    def speed_to_bin_log(self, v):
        log_v = torch.log(1 + v.clamp(min=0))
        return self._val_to_bin(log_v, self.log_speed_min, self.log_speed_max, self.n_speed_bins)

    def _pos_transform(self, v):
        if self.pos_bin_concentration == 1.0:
            return v
        mid = (self.pos_min + self.pos_max) / 2
        half = (self.pos_max - self.pos_min) / 2
        v_norm = (v - mid) / half
        v_t = torch.sign(v_norm) * (torch.abs(v_norm) + 1e-8).pow(
            1.0 / self.pos_bin_concentration)
        return mid + v_t * half

    def pos_to_bin(self, v):
        return self._val_to_bin(self._pos_transform(v),
                                self.pos_min, self.pos_max, self.n_pos_bins)

    def length_to_bin(self, v):
        return self._val_to_bin(v, self.length_min, self.length_max, self.n_length_bins)

    def width_to_bin(self, v):
        return self._val_to_bin(v, self.width_min, self.width_max, self.n_width_bins)

    # ── Training (kept for checkpoint compatibility) ──

    def training_step(self, batch, batch_idx):
        context_agent = self.agent_feature_extract(batch['agent_feat'], batch['agent_mask'], True)
        feature = self.map_feature_extract(batch['lane_inp'], batch['lane_mask'], context_agent)
        center_num = batch['center'].shape[1]
        feature = feature[:, :center_num]
        pred = self.feature_to_dists(feature)
        losses, total_loss = self.compute_loss(batch, pred)
        return total_loss

    def validation_step(self, batch, batch_idx):
        context_agent = self.agent_feature_extract(batch['agent_feat'], batch['agent_mask'], True)
        feature = self.map_feature_extract(batch['lane_inp'], batch['lane_mask'], context_agent)
        center_num = batch['center'].shape[1]
        feature = feature[:, :center_num]
        pred = self.feature_to_dists(feature)
        losses, total_loss = self.compute_loss(batch, pred)
        return total_loss

    def configure_optimizers(self):
        lr = self.cfg.get('lr', 3e-4)
        lr_min = self.cfg.get('lr_min', 1e-6)
        max_epoch = self.cfg.get('max_epoch', 100)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                     weight_decay=self.cfg.get('weight_decay', 4e-4))
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=lr_min)
        return [optimizer], [scheduler]

    # ── Feature → predictions ──

    def feature_to_dists(self, feature):
        d = {
            'prob': self.prob_head(feature).squeeze(-1),
            'heading_logits': self.heading_head(feature),
            'speed_stopped_logits': self.speed_stopped_head(feature).squeeze(-1),
            'speed_logits': self.speed_head(feature),
            'length_logits': self.length_head(feature),
            'width_logits': self.width_head(feature),
        }
        if self.use_vel_heading:
            d['vel_heading'] = self.vel_heading_head(feature).squeeze(-1)
        if self.pos_mode == 'joint':
            d['pos_joint_logits'] = self.pos_joint_head(feature)
        else:
            d['pos_long_logits'] = self.pos_long_head(feature)
            d['pos_lat_logits'] = self.pos_lat_head(feature)
        return d

    # ── Loss ──

    def _categorical_loss_with_ls(self, logits, gt_vals, n_bins, val_to_bin_fn, gt_mask, gt_sum,
                                   label_smoothing=0.0):
        gt_bins = val_to_bin_fn(gt_vals)
        loss = F.cross_entropy(
            logits.reshape(-1, n_bins),
            gt_bins.reshape(-1),
            reduction='none',
            label_smoothing=label_smoothing,
        ).reshape(gt_mask.shape)
        return (torch.sum(loss * gt_mask, dim=1) / gt_sum).mean()

    def compute_loss(self, data, pred):
        BCE = torch.nn.BCEWithLogitsLoss()
        prob_loss = BCE(pred['prob'], data['gt_distribution'])
        line_mask = data['center_mask']
        prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)

        gt_mask = data['gt_distribution']
        gt_sum = torch.clip(torch.sum(gt_mask, dim=1).unsqueeze(-1), min=1)

        heading_loss = self._categorical_loss_with_ls(
            pred['heading_logits'], data['gt_heading'],
            self.n_heading_bins, self.heading_to_bin, gt_mask, gt_sum,
            label_smoothing=self.ls_heading)

        is_stopped_gt = (data['gt_speed'] < self.speed_stopped_threshold).float()
        stopped_bce = F.binary_cross_entropy_with_logits(
            pred['speed_stopped_logits'], is_stopped_gt, reduction='none')
        stopped_loss = (torch.sum(stopped_bce * gt_mask, dim=1) / gt_sum).mean()

        moving_mask = gt_mask * (1 - is_stopped_gt)
        moving_sum = torch.clip(torch.sum(moving_mask, dim=1).unsqueeze(-1), min=1)
        gt_speed_bins = self.speed_to_bin_log(data['gt_speed'])
        speed_ce = F.cross_entropy(
            pred['speed_logits'].reshape(-1, self.n_speed_bins),
            gt_speed_bins.reshape(-1),
            reduction='none',
            label_smoothing=self.ls_speed,
        ).reshape(gt_mask.shape)
        speed_loss = (torch.sum(speed_ce * moving_mask, dim=1) / moving_sum).mean()

        pos_long_loss = self._categorical_loss_with_ls(
            pred['pos_long_logits'], data['gt_long_lat'][..., 0],
            self.n_pos_bins, self.pos_to_bin, gt_mask, gt_sum,
            label_smoothing=self.ls_position)
        pos_lat_loss = self._categorical_loss_with_ls(
            pred['pos_lat_logits'], data['gt_long_lat'][..., 1],
            self.n_pos_bins, self.pos_to_bin, gt_mask, gt_sum,
            label_smoothing=self.ls_position)

        length_loss = self._categorical_loss_with_ls(
            pred['length_logits'], data['gt_bbox'][..., 0],
            self.n_length_bins, self.length_to_bin, gt_mask, gt_sum,
            label_smoothing=self.ls_size)
        width_loss = self._categorical_loss_with_ls(
            pred['width_logits'], data['gt_bbox'][..., 1],
            self.n_width_bins, self.width_to_bin, gt_mask, gt_sum,
            label_smoothing=self.ls_size)

        if self.use_vel_heading:
            L1 = torch.nn.L1Loss(reduction='none')
            vel_heading_loss = L1(pred['vel_heading'], data['gt_vel_heading'])
            vel_heading_loss = (torch.sum(vel_heading_loss * gt_mask, dim=1) / gt_sum).mean()

        losses = {
            'prob_loss': prob_loss, 'heading_loss': heading_loss,
            'stopped_loss': stopped_loss, 'speed_loss': speed_loss,
            'pos_long_loss': pos_long_loss, 'pos_lat_loss': pos_lat_loss,
            'length_loss': length_loss, 'width_loss': width_loss,
        }
        if self.use_vel_heading:
            losses['vel_heading_loss'] = vel_heading_loss
        total_loss = sum(losses.values())
        return losses, total_loss

    # ── Sampling helpers ──

    def _sample_categorical(self, logits, centers, temperature=1.0):
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        bin_idx = dist.sample()
        value = centers[bin_idx]
        log_prob = dist.log_prob(bin_idx)
        return value, log_prob

    def _sample_speed_zero_inflated(self, stopped_logits, speed_logits):
        stopped_prob = torch.sigmoid(stopped_logits)
        is_stopped = torch.bernoulli(stopped_prob)
        speed_cond, speed_cond_lp = self._sample_categorical(
            speed_logits, self.speed_centers)
        speed = torch.where(is_stopped.bool(), torch.zeros_like(speed_cond), speed_cond)
        log_stopped = torch.log(stopped_prob + 1e-8)
        log_moving = torch.log(1 - stopped_prob + 1e-8) + speed_cond_lp
        log_prob = torch.where(is_stopped.bool(), log_stopped, log_moving)
        return speed, log_prob

    def _expected_value(self, logits, centers):
        """Compute expected value (soft argmax) from categorical logits."""
        probs = F.softmax(logits, dim=-1)
        return (probs * centers).sum(dim=-1)

    def sample_from_distribution(self, pred, center_lane, repeat_num=5):
        prob = pred['prob'][0]
        max_prob = 0

        for i in range(2):
            indx = choices(list(range(prob.shape[-1])), prob)[0]
            vec_logprob_ = prob[indx]
            if vec_logprob_ > max_prob:
                the_indx = indx
                max_prob = max(vec_logprob_, max_prob)

        prob_list = []
        agents_list = []

        for i in range(repeat_num):
            pos_long, pos_long_lp = self._sample_categorical(
                pred['pos_long_logits'], self.pos_centers, self.pos_temperature)
            pos_lat, pos_lat_lp = self._sample_categorical(
                pred['pos_lat_logits'], self.pos_centers, self.pos_temperature)
            pos = torch.stack([pos_long, pos_lat], dim=-1)
            pos_lp = pos_long_lp + pos_lat_lp

            heading, heading_lp = self._sample_categorical(pred['heading_logits'], self.heading_centers)
            speed, speed_lp = self._sample_speed_zero_inflated(
                pred['speed_stopped_logits'], pred['speed_logits'])

            if self.use_vel_heading:
                vel_heading = pred['vel_heading']
            else:
                vel_heading = torch.zeros_like(heading)

            length, length_lp = self._sample_categorical(pred['length_logits'], self.length_centers)
            width, width_lp = self._sample_categorical(pred['width_logits'], self.width_centers)
            bbox = torch.stack([length, width], dim=-1)

            agents = get_agent_pos_from_vec(
                center_lane, pos[0], speed[0], vel_heading[0], heading[0], bbox[0])
            agents_list.append(agents)

            all_prob = (pos_lp[0, the_indx] +
                        heading_lp[0, the_indx] + speed_lp[0, the_indx] +
                        length_lp[0, the_indx] + width_lp[0, the_indx])
            prob_list.append(all_prob)

        max_indx = torch.stack(prob_list).argmax().item()
        max_agents = agents_list[max_indx]
        return max_agents, prob, the_indx

    # ── Encoder ──

    def agent_feature_extract(self, agent_feat, agent_mask, random_mask):
        agent = agent_feat[..., :-2]
        agent_line_type = agent_feat[..., -2].to(int)
        agent_line_traf = agent_feat[..., -1].to(int)

        agent_line_type_embed = self.type_embedding(agent_line_type)
        agent_line_traf_embed = self.traf_embedding(agent_line_traf)

        min_agent_num = self.cfg['min_agent']
        if random_mask:
            agent_mask[:, 0] = 1
            for i in range(agent_mask.shape[0]):
                masked_num = i % min_agent_num
                agent_mask[i, 1 + masked_num:] = 0

        agent_enc = self.agent_encode(agent) + agent_line_type_embed + agent_line_traf_embed
        b, a, d = agent_enc.shape
        context_agent = torch.ones([b, d], device=agent_feat.device)
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)
        return context_agent

    def map_feature_extract(self, lane_inp, line_mask, context_agent):
        polyline = lane_inp[..., :4]
        polyline_type = lane_inp[..., 4].to(int)
        polyline_traf = lane_inp[..., 5].to(int)

        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)

        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)
        context_line = context_line.unsqueeze(1).repeat(1, line_enc.shape[1], 1)
        feature = torch.cat([line_enc, context_line], dim=-1)
        return feature

    # ── Inference ──

    def inference(self, data, context_num=1):
        agent_num = data['agent_mask'].sum().item()
        idx_list, pred_list, prob_list, shapes = [], [], [], []

        for i in range(context_num):
            context_agent = data['agent'][0, [i]].cpu().numpy()
            context_agent = WaymoAgent(context_agent)
            shapes.append(context_agent.get_polygon()[0])
            pred_list.append(context_agent)
            idx_list.append(data['vec_based_rep'][..., 0][0, i].item())

        minimum_agent = self.cfg['pad_num']
        center = data['center'][0]

        for i in range(context_num, max(agent_num, minimum_agent)):
            data['agent_mask'][:, :i] = 1
            data['agent_mask'][:, i:] = 0
            pred = self.forward(data, False)
            pred['prob'][:, idx_list] = 0

            cnt = 0
            while cnt < 3:
                agents, prob, indx = self.sample_from_distribution(pred, center)
                the_agent = agents.get_agent(indx)
                poly = the_agent.get_polygon()[0]
                if not any(poly.intersects(s) for s in shapes):
                    shapes.append(poly)
                    break
                cnt += 1

            pred_list.append(the_agent)
            data['agent_feat'][:, i] = Tensor(the_agent.get_inp())
            idx_list.append(indx)
            prob_list.append(prob)

        return {'agent': pred_list, 'prob': prob_list}

    # ── Forward ──

    def forward(self, data, random_mask=True):
        context_agent = self.agent_feature_extract(data['agent_feat'], data['agent_mask'], random_mask)
        feature = self.map_feature_extract(data['lane_inp'], data['lane_mask'], context_agent)
        center_num = data['center'].shape[1]
        feature = feature[:, :center_num]

        pred = self.feature_to_dists(feature)
        pred['prob'] = nn.Sigmoid()(pred['prob'])

        return pred
