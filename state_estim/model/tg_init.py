"""
StateEstim Stage-1: Vehicle Placement Model (initializer).

Predicts per-lane-vector distributions for:
  prob, position, bounding-box, heading, speed, velocity-heading.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from random import choices
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl

from model.model_utils import MLP_3, CG_stacked
from utils.data_utils import WaymoAgent, get_agent_pos_from_vec

copy_func = copy.deepcopy


class initializer(pl.LightningModule):
    """A transformer model with wider latent space"""

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
        self.K = cfg['gaussian_comp']

        self.prob_head = MLP_3([*middle_layer_shape, 1])
        self.speed_head = MLP_3([*middle_layer_shape, 1])
        self.vel_heading_head = MLP_3([*middle_layer_shape, 1])
        self.pos_head = MLP_3([*middle_layer_shape, self.K * (1 + 5)])
        self.bbox_head = MLP_3([*middle_layer_shape, self.K * (1 + 5)])
        self.heading_head = MLP_3([*middle_layer_shape, self.K * (1 + 2)])

    # ------------------------------------------------------------------
    # Training / validation  (kept for checkpoint compatibility)
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        context_agent = self.agent_feature_extract(batch['agent_feat'], batch['agent_mask'], True)
        feature = self.map_feature_extract(batch['lane_inp'], batch['lane_mask'], context_agent)
        center_num = batch['center'].shape[1]
        feature = feature[:, :center_num]
        pred_dists = self.feature_to_dists(feature)
        losses, total_loss = self.compute_loss(batch, pred_dists)
        log_ = {'train': losses}
        self.logger.log_metrics(log_)
        return total_loss

    def validation_step(self, batch, batch_idx):
        context_agent = self.agent_feature_extract(batch['agent_feat'], batch['agent_mask'], True)
        feature = self.map_feature_extract(batch['lane_inp'], batch['lane_mask'], context_agent)
        center_num = batch['center'].shape[1]
        feature = feature[:, :center_num]
        pred_dists = self.feature_to_dists(feature)
        losses, total_loss = self.compute_loss(batch, pred_dists)
        log_ = {'valid': losses}
        self.logger.log_metrics(log_)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1, verbose=True)
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def output_to_dist(self, para, n):
        if n == 2:
            loc, tril, diag = para[..., :2], para[..., 2], para[..., 3:]
            sigma_1 = torch.exp(diag[..., 0])
            sigma_2 = torch.exp(diag[..., 1])
            rho = torch.tanh(tril)
            cov = torch.stack(
                [sigma_1**2, rho * sigma_1 * sigma_2,
                 rho * sigma_1 * sigma_2, sigma_2**2],
                dim=-1,
            ).view(*loc.shape[:-1], 2, 2)
            return torch.distributions.multivariate_normal.MultivariateNormal(
                loc=loc, covariance_matrix=cov)
        if n == 1:
            loc, scale = para[..., 0], para[..., 1]
            scale = torch.exp(scale)
            return torch.distributions.Normal(loc, scale)

    def feature_to_dists(self, feature):
        K = self.K
        prob_pred = self.prob_head(feature).squeeze(-1)
        speed_out = nn.ReLU()(self.speed_head(feature))
        vel_heading_out = self.vel_heading_head(feature)

        # Position GMM
        pos_out = self.pos_head(feature).view([*feature.shape[:-1], K, -1])
        pos_weight = pos_out[..., 0]
        pos_param = pos_out[..., 1:]
        pos_distri = self.output_to_dist(pos_param, 2)
        pos_weight = torch.distributions.Categorical(logits=pos_weight)
        pos_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(pos_weight, pos_distri)

        # BBox GMM
        bbox_out = self.bbox_head(feature).view([*feature.shape[:-1], K, -1])
        bbox_weight = bbox_out[..., 0]
        bbox_param = bbox_out[..., 1:]
        bbox_distri = self.output_to_dist(bbox_param, 2)
        bbox_weight = torch.distributions.Categorical(logits=bbox_weight)
        bbox_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(bbox_weight, bbox_distri)

        # Heading GMM
        heading_out = self.heading_head(feature).view([*feature.shape[:-1], K, -1])
        heading_weight = heading_out[..., 0]
        heading_param = heading_out[..., 1:]
        heading_distri = self.output_to_dist(heading_param, 1)
        heading_weight = torch.distributions.Categorical(logits=heading_weight)
        heading_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(heading_weight, heading_distri)

        return {
            'prob': prob_pred,
            'pos': pos_gmm,
            'bbox': bbox_gmm,
            'heading': heading_gmm,
            'speed': speed_out.squeeze(-1),
            'vel_heading': vel_heading_out.squeeze(-1),
        }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Loss (kept for checkpoint compatibility)
    # ------------------------------------------------------------------
    def compute_loss(self, data, pred_dists):
        BCE = torch.nn.BCEWithLogitsLoss()
        MSE = torch.nn.MSELoss(reduction='none')
        L1 = torch.nn.L1Loss(reduction='none')

        prob_loss = BCE(pred_dists['prob'], data['gt_distribution'])
        line_mask = data['center_mask']
        prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)

        gt_mask = data['gt_distribution']
        gt_sum = torch.clip(torch.sum(gt_mask, dim=1).unsqueeze(-1), min=1)

        pos_loss = -pred_dists['pos'].log_prob(data['gt_long_lat'])
        pos_loss = (torch.sum(pos_loss * gt_mask, dim=1) / gt_sum).mean()

        speed_loss = MSE(pred_dists['speed'], data['gt_speed'])
        speed_loss = (torch.sum(speed_loss * gt_mask, dim=1) / gt_sum).mean()

        bbox_loss = -pred_dists['bbox'].log_prob(data['gt_bbox'])
        bbox_loss = (torch.sum(bbox_loss * gt_mask, dim=1) / gt_sum).mean()

        vel_heading_loss = L1(pred_dists['vel_heading'], data['gt_vel_heading'])
        vel_heading_loss = (torch.sum(vel_heading_loss * gt_mask, dim=1) / gt_sum).mean()

        heading_loss = -pred_dists['heading'].log_prob(data['gt_heading'])
        heading_loss = (torch.sum(heading_loss * gt_mask, dim=1) / gt_sum).mean()

        losses = {
            'prob_loss': prob_loss,
            'pos_loss': pos_loss,
            'heading_loss': heading_loss,
            'speed_loss': speed_loss,
            'vel_heading_loss': vel_heading_loss,
            'bbox_loss': bbox_loss,
        }
        total_loss = prob_loss + pos_loss + heading_loss + speed_loss + vel_heading_loss + bbox_loss
        return losses, total_loss

    # ------------------------------------------------------------------
    # Sampling / inference helpers (kept for checkpoint compatibility)
    # ------------------------------------------------------------------
    def sample_from_distribution(self, pred, center_lane, repeat_num=5):
        prob = pred['prob'][0]
        max_prob = 0
        for i in range(2):
            indx = choices(list(range(prob.shape[-1])), prob)[0]
            vec_logprob_ = prob[indx]
            if vec_logprob_ > max_prob:
                the_indx = indx
                max_prob = max(vec_logprob_, max_prob)

        prob_list, agents_list = [], []
        speed = torch.clip(pred['speed'], min=0)
        vel_heading = pred['vel_heading']

        for i in range(repeat_num):
            pos = torch.clip(pred['pos'].sample(), min=-0.5, max=0.5)
            pos_logprob = pred['pos'].log_prob(pos)
            heading = torch.clip(pred['heading'].sample(), min=-np.pi / 2, max=np.pi / 2)
            heading_logprob = pred['heading'].log_prob(heading)
            bbox = torch.clip(pred['bbox'].sample(), min=1.5)
            bbox_logprob = pred['bbox'].log_prob(bbox)

            agents = get_agent_pos_from_vec(center_lane, pos[0], speed[0], vel_heading[0], heading[0], bbox[0])
            agents_list.append(agents)
            all_prob = heading_logprob[0, the_indx] + bbox_logprob[0, the_indx] + pos_logprob[0, the_indx]
            prob_list.append(all_prob)

        max_indx = torch.stack(prob_list).argmax().item()
        return agents_list[max_indx], prob, the_indx

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data, random_mask=True):
        context_agent = self.agent_feature_extract(data['agent_feat'], data['agent_mask'], random_mask)
        feature = self.map_feature_extract(data['lane_inp'], data['lane_mask'], context_agent)
        center_num = data['center'].shape[1]
        feature = feature[:, :center_num]

        pred_dists = self.feature_to_dists(feature)
        pred_dists['prob'] = nn.Sigmoid()(pred_dists['prob'])
        return pred_dists


