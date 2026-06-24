"""
Core utilities: coordinate rotation, lane processing, map processing.
"""

import numpy as np
import torch


def cal_rel_dir(dir1, dir2):
    """Signed angular difference, result in (-pi, pi]."""
    dist = dir1 - dir2
    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2
    dist[dist > np.pi] -= np.pi * 2
    return dist


def rotate(x, y, angle):
    """Rotate (x, y) by *angle* (rad).  Works for both numpy and torch."""
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        return torch.stack((other_x_trans, other_y_trans), axis=-1)
    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        return np.stack((other_x_trans, other_y_trans), axis=-1)


# ------------------------------------------------------------------
# Lane / map vector processing
# ------------------------------------------------------------------

def process_lane(lane, max_vec, lane_range, offset=-40):
    """Convert raw lane points into fixed-size vector arrays."""
    vec_dim = 6

    lane_point_mask = (
        (abs(lane[..., 0] + offset) < lane_range) *
        (abs(lane[..., 1]) < lane_range)
    )
    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list, vec_mask_list = [], []
    b_s, _, lane_dim = lane.shape

    for lid in lane_id:
        id_set = lane[..., -2] == lid
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        vector[..., 4] = points[:, 1:, 2]       # type
        vector[..., 5] = points[:, 1:, 4]       # traffic state
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]
        dist = vector_t[..., 0]**2 + vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])
        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]
        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        all_vec[t] = vector_t
        all_mask[t] = mask_t

    return all_vec, all_mask.astype(bool)


def process_map(lane, traf, center_num=384, edge_num=128,
                lane_range=60, offest=-40):
    """
    Separate raw lane data into centre / boundary / crosswalk / rest vectors
    and attach traffic-light state.
    """
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]
    for i in range(b_s):
        traf_t = traf[i]
        lane_id_t = lane_id[i]
        for a_traf in traf_t:
            control_lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_id_t == control_lane_id)
            lane_with_traf[i, lane_idx, -1] = state

    lane = lane_with_traf
    lane_type = lane[0, :, 2]

    center_ind = (lane_type == 1) | (lane_type == 2) | (lane_type == 3)
    bound_ind = (lane_type == 15) | (lane_type == 16)
    cross_ind = (lane_type == 18) | (lane_type == 19)
    rest = ~(center_ind | bound_ind | cross_ind)

    cent, cent_mask = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest_v, rest_mask = process_lane(lane[:, rest], 192, lane_range, offest)

    return (cent, cent_mask, bound, bound_mask,
            cross, cross_mask, rest_v, rest_mask)


