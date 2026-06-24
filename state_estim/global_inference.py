"""
Global inference pipeline for StateEstim's init model (Stage 1).

Runs vehicle placement prediction across the entire map by generating anchor
points along center lanes and running the init model for each local region.

Supports both ScenarioNet and StateEstim pkl formats natively.
"""

import numpy as np
import torch
import torch.nn.functional as F

from model.tg_init import initializer
from model.tg_init_v4bi import initializer_categorical_v4
from utils.data_utils import WaymoAgent, get_vec_rep, process_map_inp
from utils.scenario_loader import load_scenario
from utils.utils import rotate, process_map

MAX_AGENT = 32
CENTER_NUM = 384
EDGE_NUM = 128


# ===========================================================================
# Anchor Generation
# ===========================================================================

def generate_anchors(lane_data, min_spacing=40.0):
    """
    Generate anchor points along center lanes for global map coverage.

    Uses a hybrid approach:
      1. Place candidate anchors at regular intervals along each center lane.
      2. Greedy de-duplication so no two anchors are closer than *min_spacing*.

    Args:
        lane_data:    np.ndarray [N_points, 4]  (x, y, type, lane_id)
        min_spacing:  float – minimum distance (m) between anchors

    Returns:
        list[dict] – each dict has keys  x, y, heading, lane_id
    """
    center_mask = np.isin(lane_data[:, 2].astype(int), [1, 2, 3])
    center_pts = lane_data[center_mask]
    if len(center_pts) == 0:
        return []

    lane_ids = np.unique(center_pts[:, 3]).astype(int)
    raw_anchors = []

    for lid in lane_ids:
        pts = center_pts[center_pts[:, 3] == lid][:, :2]
        if len(pts) < 2:
            continue

        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        if total_len < 1.0:
            continue

        if total_len < min_spacing * 1.5:
            positions = [total_len / 2]
        else:
            positions = list(np.arange(min_spacing / 2, total_len, min_spacing))

        for d in positions:
            idx = min(np.searchsorted(cum_len, d), len(pts) - 1)
            prev = max(idx - 1, 0)
            nxt = min(idx + 1, len(pts) - 1)
            if prev == nxt:
                nxt = min(nxt + 1, len(pts) - 1)
            dx = pts[nxt, 0] - pts[prev, 0]
            dy = pts[nxt, 1] - pts[prev, 1]
            heading = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-6 else 0.0

            raw_anchors.append({
                'x': float(pts[idx, 0]),
                'y': float(pts[idx, 1]),
                'heading': heading,
                'lane_id': int(lid),
            })

    # Greedy de-duplication
    selected = []
    for a in raw_anchors:
        if all(
            np.hypot(a['x'] - s['x'], a['y'] - s['y']) >= min_spacing
            for s in selected
        ):
            selected.append(a)

    return selected


# ===========================================================================
# Per-anchor data preparation
# ===========================================================================

def _transform_lane_to_anchor(lane_raw, anchor):
    lane = lane_raw.copy()
    lane[:, :2] -= np.array([anchor['x'], anchor['y']])
    coords = rotate(lane[:, 0].copy(), lane[:, 1].copy(), -anchor['heading'])
    lane[:, :2] = coords
    return lane


def _transform_agents_to_anchor(agents, anchor):
    a = agents.copy()
    a[:, :2] -= np.array([anchor['x'], anchor['y']])
    a[:, :2] = rotate(a[:, 0].copy(), a[:, 1].copy(), -anchor['heading'])
    a[:, 2:4] = rotate(a[:, 2].copy(), a[:, 3].copy(), -anchor['heading'])
    a[:, 4] -= anchor['heading']
    return a


def prepare_anchor_input(lane_raw, agents_t0, traf_t0, anchor, map_size=50):
    """Build the model-input dict for one anchor point."""
    lane_local = _transform_lane_to_anchor(lane_raw, anchor)
    agents_local = _transform_agents_to_anchor(agents_t0, anchor)

    valid = agents_local[:, -1].astype(bool)
    is_vehicle = agents_local[:, -2] == 1
    in_range = (np.abs(agents_local[:, 0]) < map_size) & (np.abs(agents_local[:, 1]) < map_size)
    keep = valid & is_vehicle & in_range
    kept_agents = agents_local[keep][:, :8]
    n = min(kept_agents.shape[0], MAX_AGENT)

    padded = np.zeros([1, MAX_AGENT, 8])
    mask = np.zeros([1, MAX_AGENT], dtype=bool)
    if n > 0:
        padded[0, :n] = kept_agents[:n]
        mask[0, :n] = True

    lane_batch = lane_local[np.newaxis]
    center, center_mask, bound, bound_mask, cross, cross_mask, rest, rest_mask = \
        process_map(lane_batch, [traf_t0],
                    center_num=CENTER_NUM, edge_num=EDGE_NUM,
                    lane_range=map_size, offest=0)

    if not center_mask.any():
        return None

    case = {
        'agent': padded,
        'agent_mask': mask,
        'center': center,
        'center_mask': center_mask,
        'bound': bound,
        'bound_mask': bound_mask,
        'cross': cross,
        'cross_mask': cross_mask,
        'rest': rest,
        'rest_mask': rest_mask,
    }

    get_vec_rep(case)

    agent_obj = WaymoAgent(case['agent'], case['vec_based_rep'])
    case['agent_feat'] = agent_obj.get_inp()

    process_map_inp(case, map_size)

    for k in case:
        if isinstance(case[k], np.ndarray) and case[k].dtype == np.float64:
            case[k] = case[k].astype(np.float32)

    return case


# ===========================================================================
# Main inference class
# ===========================================================================

class GlobalInitInference:
    """Run the init model over the full map of a Waymo scenario."""

    BATCH_KEYS = ['agent_feat', 'agent_mask', 'lane_inp', 'lane_mask',
                  'center', 'center_mask']

    def __init__(self, ckpt_path, device='cpu', map_size=50,
                 anchor_min_spacing=40, model_type='auto'):
        self.device = device
        self.map_size = map_size
        self.anchor_min_spacing = anchor_min_spacing

        print(f'Loading init model from {ckpt_path} …')
        self.model_type = self._detect_model_type(ckpt_path, model_type)
        if self.model_type == 'v4bi':
            self.model = initializer_categorical_v4.load_from_checkpoint(
                ckpt_path, map_location=device)
        else:
            self.model = initializer.load_from_checkpoint(
                ckpt_path, map_location=device)
        self.model.to(device)
        self.model.eval()
        print(f'Model loaded (type={self.model_type}).\n')

    @staticmethod
    def _detect_model_type(ckpt_path, hint='auto'):
        """Detect model type from checkpoint keys if hint is 'auto'."""
        if hint != 'auto':
            return hint
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_keys = set(ckpt.get('state_dict', {}).keys())
            if any('heading_head' in k and 'speed_stopped_head' in k
                   for k in [' '.join(state_keys)]):
                return 'v4bi'
            if 'hyper_parameters' in ckpt:
                hp = ckpt['hyper_parameters']
                cfg = hp.get('cfg', {})
                if 'n_heading_bins' in cfg:
                    return 'v4bi'
            if any('speed_stopped_head' in k for k in state_keys):
                return 'v4bi'
        except Exception:
            pass
        return 'gmm'

    # ------------------------------------------------------------------
    def run(self, scenario_path, timestep=0):
        """
        Run global inference on a scenario.

        Accepts both ScenarioNet and StateEstim pkl formats natively.

        Args:
            scenario_path: path to a .pkl file (either format).

        Returns:
            results   – list[dict] per anchor (predictions in global coords)
            lane_raw  – [N, 4] lane data (global coords)
            agents_t0 – [M, 9] agents at selected timestep (global coords)
        """
        lane_raw, agents_t0, traf_t0 = load_scenario(scenario_path, timestep=timestep)

        anchors = generate_anchors(lane_raw,
                                   min_spacing=self.anchor_min_spacing)
        print(f'  Anchors generated : {len(anchors)}')

        valid_anchors, inputs = [], []
        for a in anchors:
            inp = prepare_anchor_input(lane_raw, agents_t0, traf_t0,
                                       a, self.map_size)
            if inp is not None:
                valid_anchors.append(a)
                inputs.append(inp)

        print(f'  Valid anchors     : {len(valid_anchors)}')
        if not inputs:
            return [], lane_raw, agents_t0

        batch = self._collate(inputs)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)

        with torch.no_grad():
            pred = self.model(batch, random_mask=False)

        extractor = (self._extract_result_v4bi if self.model_type == 'v4bi'
                     else self._extract_result_gmm)
        results = [
            extractor(pred, inputs[i], valid_anchors[i], i)
            for i in range(len(valid_anchors))
        ]
        return results, lane_raw, agents_t0

    # ------------------------------------------------------------------
    def _collate(self, input_list):
        batch = {}
        for key in self.BATCH_KEYS:
            stacked = np.concatenate([inp[key] for inp in input_list], axis=0)
            t = torch.tensor(stacked)
            if t.dtype == torch.float64:
                t = t.float()
            if 'mask' in key:
                t = t.bool()
            batch[key] = t
        return batch

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_result_gmm(pred, case_info, anchor, idx):
        center = case_info['center'][0]
        cmask = case_info['center_mask'][0]
        n = int(cmask.sum())

        vecs = center[cmask]
        mx = (vecs[:, 0] + vecs[:, 2]) / 2
        my = (vecs[:, 1] + vecs[:, 3]) / 2
        lane_dir_local = np.arctan2(vecs[:, 3] - vecs[:, 1],
                                    vecs[:, 2] - vecs[:, 0])

        # Convert vector endpoints to global coordinates as well
        p1 = rotate(vecs[:, 0], vecs[:, 1], anchor['heading'])
        p2 = rotate(vecs[:, 2], vecs[:, 3], anchor['heading'])
        vec_global = np.stack([
            p1[:, 0] + anchor['x'],
            p1[:, 1] + anchor['y'],
            p2[:, 0] + anchor['x'],
            p2[:, 1] + anchor['y'],
        ], axis=-1)

        gxy = rotate(mx, my, anchor['heading'])
        gx = gxy[:, 0] + anchor['x']
        gy = gxy[:, 1] + anchor['y']
        lane_dir_global = lane_dir_local + anchor['heading']

        prob = pred['prob'][idx, :n].cpu().numpy()
        speed = pred['speed'][idx, :n].cpu().numpy()
        vel_heading = pred['vel_heading'][idx, :n].cpu().numpy()
        pos_mean = pred['pos'].mean[idx, :n].cpu().numpy()
        pos_logits = pred['pos'].mixture_distribution.logits[idx, :n].cpu().numpy()
        pos_loc = pred['pos'].component_distribution.loc[idx, :n].cpu().numpy()
        pos_cov = pred['pos'].component_distribution.covariance_matrix[idx, :n].cpu().numpy()
        bbox_mean = pred['bbox'].mean[idx, :n].cpu().numpy()
        heading_mean = pred['heading'].mean[idx, :n].cpu().numpy()
        speed_probs = np.ones((n, 1), dtype=np.float32)
        speed_centers = np.zeros((n, 1), dtype=np.float32)

        return {
            'anchor': anchor,
            'global_positions': np.stack([gx, gy], axis=-1),
            'vec_global': vec_global,
            'lane_directions': lane_dir_global,
            'prob': prob,
            'speed': speed,
            'speed_probs': speed_probs,
            'speed_centers': speed_centers,
            'vel_heading': vel_heading,
            'pos_mean': pos_mean,
            'pos_logits': pos_logits,
            'pos_loc': pos_loc,
            'pos_cov': pos_cov,
            'bbox_mean': bbox_mean,
            'heading_mean': heading_mean,
        }

    # ------------------------------------------------------------------
    def _extract_result_v4bi(self, pred, case_info, anchor, idx):
        """Extract results from V4BI categorical model output.

        Converts categorical logits to point estimates matching the GMM
        result dict format for downstream compatibility.
        """
        center = case_info['center'][0]
        cmask = case_info['center_mask'][0]
        n = int(cmask.sum())

        vecs = center[cmask]
        mx = (vecs[:, 0] + vecs[:, 2]) / 2
        my = (vecs[:, 1] + vecs[:, 3]) / 2
        lane_dir_local = np.arctan2(vecs[:, 3] - vecs[:, 1],
                                    vecs[:, 2] - vecs[:, 0])

        p1 = rotate(vecs[:, 0], vecs[:, 1], anchor['heading'])
        p2 = rotate(vecs[:, 2], vecs[:, 3], anchor['heading'])
        vec_global = np.stack([
            p1[:, 0] + anchor['x'],
            p1[:, 1] + anchor['y'],
            p2[:, 0] + anchor['x'],
            p2[:, 1] + anchor['y'],
        ], axis=-1)

        gxy = rotate(mx, my, anchor['heading'])
        gx = gxy[:, 0] + anchor['x']
        gy = gxy[:, 1] + anchor['y']
        lane_dir_global = lane_dir_local + anchor['heading']

        mdl = self.model
        prob = pred['prob'][idx, :n].cpu().numpy()

        # Speed: zero-inflated categorical distribution.
        stopped_p = torch.sigmoid(pred['speed_stopped_logits'][idx, :n]).cpu()
        speed_bin_probs_t = F.softmax(pred['speed_logits'][idx, :n], dim=-1).cpu()
        speed_cond = (speed_bin_probs_t * mdl.speed_centers.cpu()).sum(-1)
        speed = ((1 - stopped_p) * speed_cond).numpy()
        speed_probs = torch.cat([
            stopped_p.unsqueeze(-1),
            (1 - stopped_p).unsqueeze(-1) * speed_bin_probs_t,
        ], dim=-1).numpy()
        speed_centers = np.tile(
            np.concatenate([
                np.array([0.0], dtype=np.float32),
                mdl.speed_centers.detach().cpu().numpy().astype(np.float32),
            ])[None, :],
            (n, 1),
        )

        # vel_heading
        if mdl.use_vel_heading:
            vel_heading = pred['vel_heading'][idx, :n].cpu().numpy()
        else:
            vel_heading = np.zeros(n, dtype=np.float32)

        # Position: independent categorical distributions in local long/lat coords.
        pos_long_probs_t = F.softmax(pred['pos_long_logits'][idx, :n], dim=-1).cpu()
        pos_lat_probs_t = F.softmax(pred['pos_lat_logits'][idx, :n], dim=-1).cpu()
        pos_long = (pos_long_probs_t * mdl.pos_centers.cpu()).sum(-1).numpy()
        pos_lat = (pos_lat_probs_t * mdl.pos_centers.cpu()).sum(-1).numpy()
        pos_mean = np.stack([pos_long, pos_lat], axis=-1)
        pos_long_probs = pos_long_probs_t.numpy()
        pos_lat_probs = pos_lat_probs_t.numpy()
        pos_centers = np.tile(
            mdl.pos_centers.detach().cpu().numpy().astype(np.float32)[None, :],
            (n, 1),
        )

        # Bbox: categorical distributions.
        length_probs_t = F.softmax(pred['length_logits'][idx, :n], dim=-1).cpu()
        width_probs_t = F.softmax(pred['width_logits'][idx, :n], dim=-1).cpu()
        length_mean = (length_probs_t * mdl.length_centers.cpu()).sum(-1).numpy()
        width_mean = (width_probs_t * mdl.width_centers.cpu()).sum(-1).numpy()
        bbox_mean = np.stack([length_mean, width_mean], axis=-1)
        length_probs = length_probs_t.numpy()
        width_probs = width_probs_t.numpy()
        length_centers = np.tile(
            mdl.length_centers.detach().cpu().numpy().astype(np.float32)[None, :],
            (n, 1),
        )
        width_centers = np.tile(
            mdl.width_centers.detach().cpu().numpy().astype(np.float32)[None, :],
            (n, 1),
        )

        # Heading: categorical distribution relative to the lane direction.
        heading_probs_t = F.softmax(pred['heading_logits'][idx, :n], dim=-1).cpu()
        heading_mean = (heading_probs_t * mdl.heading_centers.cpu()).sum(-1).numpy()
        heading_probs = heading_probs_t.numpy()
        heading_centers = np.tile(
            mdl.heading_centers.detach().cpu().numpy().astype(np.float32)[None, :],
            (n, 1),
        )

        # Provide dummy GMM fields for compatibility with pos_loglik calcs
        pos_logits = np.zeros((n, 1), dtype=np.float32)
        pos_loc = pos_mean[:, np.newaxis, :]
        pos_cov = np.tile(np.eye(2, dtype=np.float32) * 0.01, (n, 1, 1, 1))

        return {
            'anchor': anchor,
            'global_positions': np.stack([gx, gy], axis=-1),
            'vec_global': vec_global,
            'lane_directions': lane_dir_global,
            'prob': prob,
            'speed': speed,
            'speed_probs': speed_probs,
            'speed_centers': speed_centers,
            'vel_heading': vel_heading,
            'pos_mean': pos_mean,
            'pos_long_probs': pos_long_probs,
            'pos_lat_probs': pos_lat_probs,
            'pos_centers': pos_centers,
            'pos_logits': pos_logits,
            'pos_loc': pos_loc,
            'pos_cov': pos_cov,
            'bbox_mean': bbox_mean,
            'length_probs': length_probs,
            'length_centers': length_centers,
            'width_probs': width_probs,
            'width_centers': width_centers,
            'heading_mean': heading_mean,
            'heading_probs': heading_probs,
            'heading_centers': heading_centers,
        }
