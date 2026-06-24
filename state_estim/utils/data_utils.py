"""
Data-processing utilities for the init model.

Contains WaymoAgent, get_vec_rep, process_map_inp, get_agent_pos_from_vec.
"""

import copy

import numpy as np
import torch
from torch import Tensor
from shapely.geometry import Polygon

from utils.utils import cal_rel_dir, rotate


# ===========================================================================
# get_agent_pos_from_vec  (used by model sampling)
# ===========================================================================

def get_agent_pos_from_vec(vec, long_lat, speed, vel_heading, heading, bbox):
    x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
    vec_len = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len * long_lat[..., 0]
    lat_pos = vec_len * long_lat[..., 1]
    coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)
    coord[:, 0] += x_center
    coord[:, 1] += y_center

    agent_dir = vec_dir + heading
    v_dir = vel_heading + agent_dir
    vel = torch.stack([torch.cos(v_dir) * speed, torch.sin(v_dir) * speed], axis=-1)
    agent_num, _ = vel.shape

    type_col = Tensor([[1]]).repeat(agent_num, 1).to(coord.device)
    agent = torch.cat([coord, vel, agent_dir.unsqueeze(1), bbox, type_col], dim=-1).detach().cpu().numpy()

    vec_based_rep = torch.cat(
        [long_lat, speed.unsqueeze(-1), vel_heading.unsqueeze(-1),
         heading.unsqueeze(-1), vec], dim=-1
    ).detach().cpu().numpy()

    return WaymoAgent(agent, vec_based_rep)


# ===========================================================================
# get_vec_rep
# ===========================================================================

def get_vec_rep(case_info):
    """Assign each agent to its nearest centre-lane vector and compute
    vector-based representation features."""
    thres = 5
    max_agent_num = 32

    agent = case_info['agent']
    vectors = case_info['center']
    agent_mask = case_info['agent_mask']

    vec_x = (vectors[..., 0] + vectors[..., 2]) / 2
    vec_y = (vectors[..., 1] + vectors[..., 3]) / 2
    agent_x = agent[..., 0]
    agent_y = agent[..., 1]

    b, vec_num = vec_y.shape
    _, agent_num = agent_x.shape

    vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
    vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)
    agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
    agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

    dist = np.sqrt((vec_x - agent_x)**2 + (vec_y - agent_y)**2)
    cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
    dist[cent_mask == 0] = 10e5
    vec_index = np.argmin(dist, -1)
    min_dist_to_lane = np.min(dist, -1)
    min_dist_mask = min_dist_to_lane < thres

    selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

    vx, vy = agent[..., 2], agent[..., 3]
    v_value = np.sqrt(vx**2 + vy**2)
    low_vel = v_value < 0.1
    dir_v = np.arctan2(vy, vx)

    x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
    dir_ = np.arctan2(y2 - y1, x2 - x1)
    agent_dir = agent[..., 4]

    v_relative_dir = cal_rel_dir(dir_v, agent_dir)
    relative_dir = cal_rel_dir(agent_dir, dir_)
    v_relative_dir[low_vel] = 0

    v_dir_mask = abs(v_relative_dir) < np.pi / 6
    dir_mask = abs(relative_dir) < np.pi / 4

    agent_x2 = agent[..., 0]
    agent_y2 = agent[..., 1]
    vec_x2 = (x1 + x2) / 2
    vec_y2 = (y1 + y2) / 2
    cent_to_agent_x = agent_x2 - vec_x2
    cent_to_agent_y = agent_y2 - vec_y2
    coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir_)

    vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)
    lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
    long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len

    total_mask = min_dist_mask * agent_mask * v_dir_mask * dir_mask
    total_mask[:, 0] = 1
    total_mask = total_mask.astype(bool)

    b_s, agent_num_orig, agent_dim = agent.shape
    agent_ = np.zeros([b_s, max_agent_num, agent_dim])
    agent_mask_ = np.zeros([b_s, max_agent_num]).astype(bool)

    the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
    info = np.concatenate([
        vec_index[..., np.newaxis],
        long_perc[..., np.newaxis],
        lat_perc[..., np.newaxis],
        v_value[..., np.newaxis],
        v_relative_dir[..., np.newaxis],
        relative_dir[..., np.newaxis],
        the_vec,
    ], -1)
    info_ = np.zeros([b_s, max_agent_num, info.shape[-1]])

    for i in range(agent.shape[0]):
        agent_i = agent[i][total_mask[i]]
        info_i = info[i][total_mask[i]]
        agent_i = agent_i[:max_agent_num]
        info_i = info_i[:max_agent_num]
        valid_num = agent_i.shape[0]
        agent_i = np.pad(agent_i, [[0, max_agent_num - agent_i.shape[0]], [0, 0]])
        info_i = np.pad(info_i, [[0, max_agent_num - info_i.shape[0]], [0, 0]])
        agent_[i] = agent_i
        info_[i] = info_i
        agent_mask_[i, :valid_num] = True

    case_info['vec_based_rep'] = info_[..., 1:]
    case_info['agent_vec_indx'] = info_[..., 0].astype(int)
    case_info['agent_mask'] = agent_mask_
    case_info['agent'] = agent_


# ===========================================================================
# process_map_inp
# ===========================================================================

def process_map_inp(case_info, map_size=50):
    """Normalise lane-vector coordinates by map_size and concatenate."""
    center = copy.deepcopy(case_info['center'])
    center[..., :4] /= map_size
    edge = copy.deepcopy(case_info['bound'])
    edge[..., :4] /= map_size
    cross = copy.deepcopy(case_info['cross'])
    cross[..., :4] /= map_size
    rest = copy.deepcopy(case_info['rest'])
    rest[..., :4] /= map_size

    case_info['lane_inp'] = np.concatenate([center, edge, cross, rest], axis=1)
    case_info['lane_mask'] = np.concatenate(
        [case_info['center_mask'], case_info['bound_mask'],
         case_info['cross_mask'], case_info['rest_mask']], axis=1
    )


# ===========================================================================
# WaymoAgent
# ===========================================================================

class WaymoAgent:
    def __init__(self, feature, vec_based_info=None, range=50, max_speed=30, from_inp=False):
        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:
            self.position = feature[..., :2] * self.RANGE
            self.velocity = feature[..., 2:4] * self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5], feature[..., 4])[..., np.newaxis]
            self.length_width = feature[..., 6:8]
            type_col = np.ones_like(self.heading)
            self.feature = np.concatenate(
                [self.position, self.velocity, self.heading, self.length_width, type_col], axis=-1)
            if vec_based_info is not None:
                vec_based_rep = copy.deepcopy(vec_based_info)
                vec_based_rep[..., 5:9] *= self.RANGE
                vec_based_rep[..., 2] *= self.MAX_SPEED
                self.vec_based_info = vec_based_rep
        else:
            self.feature = feature
            self.position = feature[..., :2]
            self.velocity = feature[..., 2:4]
            self.heading = feature[..., [4]]
            self.length_width = feature[..., 5:7]
            self.type = feature[..., [7]]
            self.vec_based_info = vec_based_info

    @staticmethod
    def from_list_to_array(inp_list):
        MAX_AGENT = 32
        agent = np.concatenate([x.get_inp(act=True) for x in inp_list], axis=0)
        agent = agent[:MAX_AGENT]
        agent_num = agent.shape[0]
        agent = np.pad(agent, ([0, MAX_AGENT - agent_num], [0, 0]))
        agent_mask = np.zeros([agent_num])
        agent_mask = np.pad(agent_mask, ([0, MAX_AGENT - agent_num]))
        agent_mask[:agent_num] = 1
        return agent, agent_mask.astype(bool)

    def get_agent(self, index):
        return WaymoAgent(self.feature[[index]], self.vec_based_info[[index]])

    def get_list(self):
        bs, agent_num, feature_dim = self.feature.shape
        vec_dim = self.vec_based_info.shape[-1]
        feature = self.feature.reshape([-1, feature_dim])
        vec_rep = self.vec_based_info.reshape([-1, vec_dim])
        return [WaymoAgent(feature[[i]], vec_rep[[i]]) for i in range(feature.shape[0])]

    def get_inp(self, act=False, act_inp=False):
        if act:
            return np.concatenate([self.position, self.velocity, self.heading, self.length_width], axis=-1)

        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED
        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        if act_inp:
            return np.concatenate([pos, velo, cos_head, sin_head, self.length_width], axis=-1)

        vec_based_rep = copy.deepcopy(self.vec_based_info)
        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED
        return np.concatenate([pos, velo, cos_head, sin_head, self.length_width, vec_based_rep], axis=-1)

    def get_rect(self, pad=0):
        l, w = (self.length_width[..., 0] + pad) / 2, (self.length_width[..., 1] + pad) / 2
        x1, y1 = l, w
        x2, y2 = l, -w
        point1 = rotate(x1, y1, self.heading[..., 0])
        point2 = rotate(x2, y2, self.heading[..., 0])
        center = self.position
        x1, y1 = point1[..., [0]], point1[..., [1]]
        x2, y2 = point2[..., [0]], point2[..., [1]]
        p1 = np.concatenate([center[..., [0]] + x1, center[..., [1]] + y1], axis=-1)
        p2 = np.concatenate([center[..., [0]] + x2, center[..., [1]] + y2], axis=-1)
        p3 = np.concatenate([center[..., [0]] - x1, center[..., [1]] - y1], axis=-1)
        p4 = np.concatenate([center[..., [0]] - x2, center[..., [1]] - y2], axis=-1)
        p1 = p1.reshape(-1, p1.shape[-1])
        p2 = p2.reshape(-1, p1.shape[-1])
        p3 = p3.reshape(-1, p1.shape[-1])
        p4 = p4.reshape(-1, p1.shape[-1])
        return [np.stack([p1[i], p2[i], p3[i], p4[i]]) for i in range(p1.shape[0])]

    def get_polygon(self):
        rect_list = self.get_rect(pad=0.25)
        return [Polygon([r[0], r[1], r[2], r[3]]) for r in rect_list]


