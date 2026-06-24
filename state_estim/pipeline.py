"""pipeline.py – StateEstim → CVAE end-to-end pipeline

Given a Waymo scenario (StateEstim PKL format):
  1. select_timestep → best timestep t*
  2. StateEstim Bernoulli/position distributions → collision snapshot
  3. Convert lane data → ADV-BMT 27D map features for CVAE
  4. CVAE reverse-time inference → past trajectory per inserted vehicle
  5. Export scenario dumps and optional Reverse Waymo plots

Usage:
    python pipeline.py --data_dir /path/to/state_estim_pkls --n_scenarios 3
    python pipeline.py --data_dir /path/to/state_estim_pkls --n_scenarios 1 --device cuda
"""

import argparse
import glob
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# ── path setup ──────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_COSTER_DIR = _THIS_DIR.parent          # …/COSTER
_CVAE_DIR = _COSTER_DIR / "cvae"

import importlib.util
from shapely.geometry import Point
from shapely.affinity import (rotate as shapely_rotate,
                               scale as shapely_scale,
                               translate as shapely_translate)

# Package imports
_REFINE_DIR = _COSTER_DIR / "refine"
if str(_COSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_COSTER_DIR))
from refine.refine import refine_trajectory
from refine.reject import PhysicsLimits, full_rejection_check

# StateEstim module imports (must come first – state_estim_module/ has a model/ package)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from utils.scenario_loader import load_scenario, _ensure_pickle_compat
from global_inference import GlobalInitInference
from select_timestep import select_best_timestep
from ego_lane_overlap import (
    extract_center_vectors, find_overlapping_mask, build_inference_lookup,
    _agent_polygon, _lane_vec_polygon,
    EGO_SEMI_LONG, EGO_SEMI_LAT,
)

from cvae.model import VectorMapCVAE


# =========================================================================
# Constants
# =========================================================================

# CVAE settings (from config.py defaults)
CVAE_OB_HORIZON = 0
CVAE_PRED_HORIZON = 10
CVAE_SKIP = 5              # 10Hz → 2Hz (dt = 0.5s)
CVAE_DT = CVAE_SKIP * 0.1  # 0.5s per step
CVAE_MAX_VECTORS = 128
CVAE_MAX_MAP_FEATURES = 256
CVAE_MAX_NEIGHBORS = 32
CVAE_OB_RADIUS = 50.0
CVAE_N_PREDICTIONS = 500   # stochastic samples for refine + cluster

# Refine / representative-selection settings
REFINE_VOCAB_PATH = str(_COSTER_DIR / "refine" / "data" / "vocab_t0_K384.npz")
KMEANS_N_CLUSTERS = 5
KMEANS_MAX_ITERS = 50

# StateEstim type code → ADV-BMT 27D type flags (dims 13-24)
#   (is_lane, is_sidewalk, is_road_boundary, is_road_line,
#    is_broken, is_solid, is_yellow, is_white, is_driveway,
#    is_crosswalk, is_speed_bump, is_stop_sign)
_TYPE_FLAGS = {
    1:  (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # LANE_FREEWAY
    2:  (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # LANE_SURFACE_STREET
    3:  (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # LANE_BIKE_LANE
    6:  (0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0),  # BROKEN_SINGLE_WHITE
    7:  (0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0),  # SOLID_SINGLE_WHITE
    8:  (0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0),  # SOLID_DOUBLE_WHITE
    9:  (0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0),  # BROKEN_SINGLE_YELLOW
    10: (0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0),  # BROKEN_DOUBLE_YELLOW
    11: (0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0),  # SOLID_SINGLE_YELLOW
    12: (0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0),  # SOLID_DOUBLE_YELLOW
    13: (0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0),  # PASSING_DOUBLE_YELLOW
    15: (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # ROAD_EDGE_BOUNDARY
    16: (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # ROAD_EDGE_MEDIAN
    17: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # STOP_SIGN
    18: (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),  # CROSSWALK
    19: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),  # SPEED_BUMP
}


def _safe_prob(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = float(probs.sum())
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(probs, dtype=np.float64) / max(len(probs), 1)
    return probs / total


def _sample_categorical_value(probs, centers, fallback):
    if probs is None or centers is None:
        return float(fallback)
    probs = _safe_prob(probs)
    centers = np.asarray(centers, dtype=np.float64)
    if len(probs) != len(centers) or len(probs) == 0:
        return float(fallback)
    idx = int(np.random.choice(len(probs), p=probs))
    return float(centers[idx])


def _trajectory_lateral_arc_features(trajs_fwd, collision_heading):
    """Return [signed lateral displacement, arc length] per trajectory."""
    trajs = np.asarray(trajs_fwd, dtype=np.float64)
    if trajs.ndim != 3 or trajs.shape[1] < 2:
        return np.zeros((len(trajs), 2), dtype=np.float64)

    heading_axis = np.array(
        [np.cos(collision_heading), np.sin(collision_heading)], dtype=np.float64)
    lateral_axis = np.array([-heading_axis[1], heading_axis[0]], dtype=np.float64)

    displacement = trajs[:, -1, :2] - trajs[:, 0, :2]
    lateral_disp = displacement @ lateral_axis
    arc_length = np.linalg.norm(np.diff(trajs[:, :, :2], axis=1), axis=-1).sum(axis=1)
    return np.stack([lateral_disp, arc_length], axis=-1)


def _standardize_features(features):
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - mean) / std


def _kmeans_labels(features, n_clusters, max_iters=KMEANS_MAX_ITERS):
    """Small deterministic K-means implementation for 2D trajectory features."""
    x = _standardize_features(features)
    n = len(x)
    k = min(max(int(n_clusters), 1), n)
    if n == 0:
        return np.empty(0, dtype=int), np.empty((0, x.shape[1]), dtype=np.float64)

    centroids = [x[int(np.argmin(np.linalg.norm(x, axis=1)))]]
    while len(centroids) < k:
        dists = np.min(
            np.linalg.norm(x[:, None, :] - np.stack(centroids)[None, :, :], axis=-1),
            axis=1,
        )
        centroids.append(x[int(np.argmax(dists))])
    centroids = np.stack(centroids, axis=0)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dmat = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = np.argmin(dmat, axis=1)
        new_centroids = centroids.copy()
        for ci in range(k):
            members = x[new_labels == ci]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(dmat, axis=1)))
                new_centroids[ci] = x[farthest]
            else:
                new_centroids[ci] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centroids, new_centroids):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    return labels, centroids


def _kmeans_representatives(features, labels, centroids):
    """Return nearest-to-centroid raw trajectory index for each cluster."""
    x = _standardize_features(features)
    reps = {}
    cluster_sizes = {}
    for cl in sorted(set(labels.tolist())):
        idx = np.where(labels == cl)[0]
        if len(idx) == 0:
            continue
        dists = np.linalg.norm(x[idx] - centroids[cl], axis=1)
        reps[int(cl)] = int(idx[int(np.argmin(dists))])
        cluster_sizes[int(cl)] = int(len(idx))
    return reps, cluster_sizes


def _box_support_along(heading, half_l, half_w, direction):
    rel = direction - heading
    return abs(half_l * np.cos(rel)) + abs(half_w * np.sin(rel))


def _local_long_lat_from_global(point_xy, vec_global):
    """Convert a global point to StateEstim local (long, lat) coordinates."""
    x1, y1, x2, y2 = [float(v) for v in vec_global]
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
    delta = np.asarray(point_xy, dtype=np.float64) - center
    vec_len = max(float(np.hypot(x2 - x1, y2 - y1)), 1e-6)
    vec_dir = float(np.arctan2(y2 - y1, x2 - x1))
    c, s = np.cos(np.pi / 2.0 - vec_dir), np.sin(np.pi / 2.0 - vec_dir)
    lat = (delta[0] * c - delta[1] * s) / vec_len
    long = (delta[0] * s + delta[1] * c) / vec_len
    return float(long), float(lat)


def _global_from_local_long_lat(long_lat, vec_global):
    """Convert StateEstim local (long, lat) coordinates to a global point."""
    x1, y1, x2, y2 = [float(v) for v in vec_global]
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
    vec_len = max(float(np.hypot(x2 - x1, y2 - y1)), 1e-6)
    vec_dir = float(np.arctan2(y2 - y1, x2 - x1))
    long, lat = float(long_lat[0]), float(long_lat[1])
    local = np.array([lat * vec_len, long * vec_len], dtype=np.float64)
    c, s = np.cos(-np.pi / 2.0 + vec_dir), np.sin(-np.pi / 2.0 + vec_dir)
    return center + np.array([
        local[0] * c - local[1] * s,
        local[0] * s + local[1] * c,
    ])


def _position_log_prob(point_xy, state):
    centers = state.get('pos_centers')
    p_long = state.get('pos_long_probs')
    p_lat = state.get('pos_lat_probs')
    vec_global = state.get('vec_global')
    if centers is None or p_long is None or p_lat is None or vec_global is None:
        return 0.0
    long, lat = _local_long_lat_from_global(point_xy, vec_global)
    centers = np.asarray(centers, dtype=np.float64)
    p_long = _safe_prob(p_long)
    p_lat = _safe_prob(p_lat)
    i_long = int(np.argmin(np.abs(centers - long)))
    i_lat = int(np.argmin(np.abs(centers - lat)))
    logp = np.log(max(float(p_long[i_long]), 1e-12))
    logp += np.log(max(float(p_lat[i_lat]), 1e-12))
    outside = max(abs(long) - 0.5, 0.0) + max(abs(lat) - 0.5, 0.0)
    return float(logp - 20.0 * outside)


def _paper_snapshot_from_region(target_state, state):
    """Build the collision snapshot from one selected local region.

    The region is selected by Bernoulli placement probability.  Heading,
    velocity, and size are sampled from the predicted state distributions.
    Among face-contact placements around the target, the center with maximum
    predicted position probability is selected.
    """
    tx, ty = float(target_state[0]), float(target_state[1])
    target_heading = float(target_state[4])
    target_l = max(float(target_state[5]), 0.3)
    target_w = max(float(target_state[6]), 0.3)

    heading_rel = _sample_categorical_value(
        state.get('heading_probs'), state.get('heading_centers'),
        state.get('heading_mean', 0.0))
    heading = float(state.get('lane_directions', 0.0)) + heading_rel
    speed = _sample_categorical_value(
        state.get('speed_probs'), state.get('speed_centers'),
        state.get('speed', 0.0))
    length = max(_sample_categorical_value(
        state.get('length_probs'), state.get('length_centers'),
        state.get('bbox_mean', [4.5, 2.0])[0]), 0.3)
    width = max(_sample_categorical_value(
        state.get('width_probs'), state.get('width_centers'),
        state.get('bbox_mean', [4.5, 2.0])[1]), 0.3)

    candidate_dirs = [
        target_heading,
        target_heading + np.pi / 2.0,
        target_heading + np.pi,
        target_heading - np.pi / 2.0,
    ]
    if state.get('pos_mean') is not None and state.get('vec_global') is not None:
        expected_xy = _global_from_local_long_lat(state['pos_mean'], state['vec_global'])
        expected_dir = float(np.arctan2(expected_xy[1] - ty, expected_xy[0] - tx))
        candidate_dirs.append(expected_dir)

    best = None
    for direction in candidate_dirs:
        target_support = _box_support_along(
            target_heading, target_l / 2.0, target_w / 2.0, direction)
        adv_support = _box_support_along(
            heading, length / 2.0, width / 2.0, direction + np.pi)
        center = np.array([
            tx + np.cos(direction) * (target_support + adv_support),
            ty + np.sin(direction) * (target_support + adv_support),
        ])
        score = _position_log_prob(center, state)
        if best is None or score > best[0]:
            best = (score, center, direction)

    _, center, place_dir = best
    poly = _agent_polygon(center[0], center[1], heading, length, width)
    return {
        'cx': float(center[0]),
        'cy': float(center[1]),
        'heading': heading,
        'speed': speed,
        'vel_heading': float(state.get('vel_heading', 0.0)),
        'length': length,
        'width': width,
        'prob': float(state.get('prob', 0.0)),
        'position_log_prob': _position_log_prob(center, state),
        'place_dir': place_dir,
        'poly': poly,
        'region_x': float(state.get('global_positions', [np.nan, np.nan])[0]),
        'region_y': float(state.get('global_positions', [np.nan, np.nan])[1]),
    }


# ---------------------------------------------------------------------------
# Trajectory quality evaluation (for new Stage 3 filtering)
# ---------------------------------------------------------------------------

_ROAD_LINE_TYPES_FOR_CROSSING = {6, 7, 8, 9, 10, 11, 12, 13}
_ROAD_EDGE_TYPES_FOR_OFFMAP = {15, 16}
_LANE_CENTER_TYPES_FOR_OFFMAP = {1, 2, 3}
_LANE_CROSS_THRESHOLD = 3       # cross ≥3 road-lines → bad
_OFF_MAP_DIST_THRESHOLD = 5.0   # metres from nearest lane centre → off-map


def _evaluate_traj_quality(traj_fwd, lane_raw,
                           cross_threshold=_LANE_CROSS_THRESHOLD,
                           off_map_dist=_OFF_MAP_DIST_THRESHOLD,
                           all_agent=None, best_t=None,
                           veh_length=None, veh_width=None,
                           traj_headings=None):
    """Evaluate a single forward-time trajectory for off-map / lane-crossing /
    collision with existing agents.  (Legacy — kept for backward compat.)

    Returns:
        dict  is_off_map, lane_crossings, has_collision, is_bad
    """
    from shapely.geometry import LineString, Point, MultiLineString

    _empty = {'is_off_map': False, 'lane_crossings': 0,
              'has_collision': False, 'is_bad': False}

    pts = np.asarray(traj_fwd)[:, :2]
    if len(pts) < 2:
        return _empty

    try:
        traj_line = LineString(pts)
    except Exception:
        return _empty

    lane_ids = lane_raw[:, 3].astype(int)
    types = lane_raw[:, 2].astype(int)

    crossing_count = 0
    seen = set()
    for lid_val in lane_ids:
        lid = int(lid_val)
        if lid in seen:
            continue
        seen.add(lid)
        mask = lane_ids == lid
        tc = int(types[mask][0])
        if tc not in _ROAD_LINE_TYPES_FOR_CROSSING:
            continue
        xy = lane_raw[mask][:, :2]
        if len(xy) < 2:
            continue
        try:
            if traj_line.intersects(LineString(xy)):
                crossing_count += 1
        except Exception:
            continue

    is_off_map = False
    seen2 = set()
    for lid_val in lane_ids:
        lid = int(lid_val)
        if lid in seen2:
            continue
        seen2.add(lid)
        mask = lane_ids == lid
        tc = int(types[mask][0])
        if tc not in _ROAD_EDGE_TYPES_FOR_OFFMAP:
            continue
        xy = lane_raw[mask][:, :2]
        if len(xy) < 2:
            continue
        try:
            if traj_line.intersects(LineString(xy)):
                is_off_map = True
                break
        except Exception:
            continue

    if not is_off_map:
        lc_lines = []
        seen3 = set()
        for lid_val in lane_ids:
            lid = int(lid_val)
            if lid in seen3:
                continue
            seen3.add(lid)
            mask = lane_ids == lid
            tc = int(types[mask][0])
            if tc not in _LANE_CENTER_TYPES_FOR_OFFMAP:
                continue
            xy = lane_raw[mask][:, :2]
            if len(xy) >= 2:
                try:
                    lc_lines.append(LineString(xy))
                except Exception:
                    pass
        if lc_lines:
            ml = MultiLineString(lc_lines)
            for p in pts:
                try:
                    if ml.distance(Point(float(p[0]), float(p[1]))) > off_map_dist:
                        is_off_map = True
                        break
                except Exception:
                    continue

    has_collision = False
    if (all_agent is not None and best_t is not None
            and veh_length is not None and veh_width is not None):
        n_pts = len(pts)
        T_total = all_agent.shape[0]
        for i in range(n_pts):
            t_10hz = best_t - (n_pts - 1 - i) * CVAE_SKIP
            if t_10hz < 0 or t_10hz >= T_total:
                continue
            ins_heading = _robust_traj_heading(
                pts, i, traj_headings=traj_headings,
                lane_raw=lane_raw, lookback=3)
            if ins_heading is None:
                continue
            ins_poly = _agent_polygon(
                float(pts[i, 0]), float(pts[i, 1]),
                ins_heading, veh_length, veh_width)
            agents_at_t = all_agent[t_10hz]
            for ai in range(1, agents_at_t.shape[0]):
                ag = agents_at_t[ai]
                if ag[8] < 0.5:
                    continue
                ag_poly = _agent_polygon(
                    float(ag[0]), float(ag[1]), float(ag[4]),
                    max(float(ag[5]), 0.5), max(float(ag[6]), 0.3))
                if ins_poly.intersects(ag_poly):
                    has_collision = True
                    break
            if has_collision:
                break

    is_bad = is_off_map or crossing_count >= cross_threshold or has_collision
    return {'is_off_map': is_off_map, 'lane_crossings': crossing_count,
            'has_collision': has_collision, 'is_bad': is_bad}


# -----------------------------------------------------------------------
# Traffic-light compliance check
# -----------------------------------------------------------------------
_TL_RED_STATE = 1   # LANE_STATE_STOP / LANE_STATE_ARROW_STOP
_TL_STOP_PROXIMITY = 5.0   # metres – trajectory point within this dist of stop line


def _evaluate_tl_compliance(traj_fwd, lane_raw, traffic_light, best_t,
                            traj_headings=None):
    """Check if a forward-time trajectory violates red traffic lights.

    For each trajectory point, we identify the 10 Hz timestep and look up
    which traffic lights are red at that moment.  If the trajectory crosses
    (passes beyond) a red-light stop point on the relevant lane, it counts
    as one violation.

    Args:
        traj_fwd:      (T, 2+) forward-time global positions
        lane_raw:      (N, 4) [x, y, type, lid]
        traffic_light: list[list[ndarray]]  per-timestep TL info
                       each ndarray = [lane_id, stop_x, stop_y, ?, state, active]
        best_t:        collision timestep (10 Hz)
        traj_headings: optional per-point headings

    Returns:
        dict  n_violations (int), is_bad (bool)
    """
    from shapely.geometry import LineString, Point

    pts = np.asarray(traj_fwd)[:, :2]
    n_pts = len(pts)
    if n_pts < 2:
        return {'n_violations': 0, 'is_bad': False}

    if traffic_light is None or len(traffic_light) == 0:
        return {'n_violations': 0, 'is_bad': False}

    T_tl = len(traffic_light)

    # Pre-build lane center polylines (type 1,2,3) for projecting
    lane_ids = lane_raw[:, 3].astype(int)
    types_arr = lane_raw[:, 2].astype(int)
    lane_center_map = {}  # lid → np.array of (x, y)
    seen = set()
    for lid_val in lane_ids:
        lid = int(lid_val)
        if lid in seen:
            continue
        seen.add(lid)
        mask = lane_ids == lid
        tc = int(types_arr[mask][0])
        if tc in _LANE_CENTER_TYPES_FOR_OFFMAP:
            xy = lane_raw[mask][:, :2]
            if len(xy) >= 2:
                lane_center_map[lid] = xy

    # Build trajectory line
    try:
        traj_line = LineString(pts)
    except Exception:
        return {'n_violations': 0, 'is_bad': False}

    violated_tl_ids = set()  # track unique TL violations

    for i in range(n_pts):
        # Map trajectory index → 10 Hz timestep
        t_10hz = best_t - (n_pts - 1 - i) * CVAE_SKIP
        if t_10hz < 0 or t_10hz >= T_tl:
            continue

        # Check all red traffic lights at this timestep
        for tl_info in traffic_light[t_10hz]:
            tl_lane = int(tl_info[0])
            tl_state = int(tl_info[4])
            tl_active = float(tl_info[5])

            if tl_state != _TL_RED_STATE or tl_active < 0.5:
                continue

            stop_pt = np.array([float(tl_info[1]), float(tl_info[2])])

            # Is the trajectory point near this stop line?
            dist_to_stop = np.linalg.norm(pts[i] - stop_pt)
            if dist_to_stop > _TL_STOP_PROXIMITY * 3:
                continue  # far away, skip

            # Check if this lane's center polyline exists
            if tl_lane not in lane_center_map:
                continue

            lane_xy = lane_center_map[tl_lane]
            try:
                lane_line = LineString(lane_xy)
            except Exception:
                continue

            # Project the stop point onto the lane to get a "fraction"
            stop_proj = lane_line.project(Point(stop_pt))
            # Project the trajectory point
            traj_pt_proj = lane_line.project(Point(pts[i]))

            # If the trajectory point is past the stop point on the lane
            # AND close to the lane → violation
            lane_dist = lane_line.distance(Point(pts[i]))
            if lane_dist < _TL_STOP_PROXIMITY and traj_pt_proj > stop_proj:
                tl_key = (tl_lane, t_10hz)
                if tl_key not in violated_tl_ids:
                    violated_tl_ids.add(tl_key)

    n_violations = len(violated_tl_ids)
    return {'n_violations': n_violations, 'is_bad': n_violations > 0}


def _robust_traj_heading(pts, i, traj_headings=None,
                         lane_raw=None, lookback=3):
    """Compute heading at trajectory index *i* with three-level fallback.

    Priority:
      1. ``traj_headings[i]``  (refine-derived or stored heading)
      2. Multi-point average of forward + backward differences
      3. Direction of nearest lane-centre vector
    Returns heading in radians, or *None* if nothing works.
    """
    # ---- Level 1: stored heading ----
    if traj_headings is not None:
        try:
            h = float(traj_headings[i])
            if np.isfinite(h):
                return h
        except (IndexError, TypeError):
            pass

    # ---- Level 2: multi-point average ----
    directions = []
    if i < len(pts) - 1:
        dx = float(pts[i + 1, 0] - pts[i, 0])
        dy = float(pts[i + 1, 1] - pts[i, 1])
        if np.hypot(dx, dy) > 1e-6:
            directions.append(np.arctan2(dy, dx))
    for k in range(1, min(lookback + 1, i + 1)):
        dx = float(pts[i, 0] - pts[i - k, 0])
        dy = float(pts[i, 1] - pts[i - k, 1])
        if np.hypot(dx, dy) > 1e-6:
            directions.append(np.arctan2(dy, dx))
    if directions:
        return float(np.arctan2(
            np.mean(np.sin(directions)),
            np.mean(np.cos(directions))))

    # ---- Level 3: nearest lane-centre vector direction ----
    if lane_raw is not None:
        _lc_types = {1, 2, 3}
        lids = lane_raw[:, 3].astype(int)
        types = lane_raw[:, 2].astype(int)
        best_d, best_h = 1e18, None
        seen = set()
        for lid_val in lids:
            lid = int(lid_val)
            if lid in seen:
                continue
            seen.add(lid)
            mask = lids == lid
            tc = int(types[mask][0])
            if tc not in _lc_types:
                continue
            xy = lane_raw[mask][:, :2]
            for j in range(len(xy) - 1):
                mx = (xy[j, 0] + xy[j + 1, 0]) / 2.0
                my = (xy[j, 1] + xy[j + 1, 1]) / 2.0
                d = np.hypot(pts[i, 0] - mx, pts[i, 1] - my)
                if d < best_d:
                    sdx = float(xy[j + 1, 0] - xy[j, 0])
                    sdy = float(xy[j + 1, 1] - xy[j, 1])
                    if np.hypot(sdx, sdy) > 1e-6:
                        best_d = d
                        best_h = float(np.arctan2(sdy, sdx))
        if best_h is not None:
            return best_h

    return None


_RW_DRAW_FUNCS = None


def _get_reverse_waymo_draw_funcs():
    """Lazy import Reverse_waymo drawing helpers."""
    global _RW_DRAW_FUNCS
    if _RW_DRAW_FUNCS is not None:
        return _RW_DRAW_FUNCS
    if not reverse_waymo_available():
        raise ImportError(_reverse_waymo_missing_message())
    _plot_script_dir = os.path.join(_REVERSE_WAYMO_ROOT, "scripts")
    if _plot_script_dir not in sys.path:
        sys.path.insert(0, _plot_script_dir)
    from plot_waymo_vector_results import (
        draw_vector_map as _draw_vector_map,
        draw_fading_polyline as _draw_fading_polyline,
        draw_oriented_box as _draw_oriented_box,
        apply_ticks_and_grid as _apply_ticks_and_grid,
    )
    _RW_DRAW_FUNCS = (
        _draw_vector_map,
        _draw_fading_polyline,
        _draw_oriented_box,
        _apply_ticks_and_grid,
    )
    return _RW_DRAW_FUNCS


def _draw_snapshot_candidates(
    ax,
    lane_raw: np.ndarray,
    all_agent: np.ndarray,
    best_t: int,
    vectors: np.ndarray,
    overlap_mask: np.ndarray,
    candidates: List[Dict],
    title: str,
    ego_margin: float,
):
    """Draw selected collision snapshot candidates in Reverse Waymo plot style."""
    (_draw_vector_map, _draw_fading_polyline,
     _draw_oriented_box, _apply_ticks_and_grid) = _get_reverse_waymo_draw_funcs()
    light_gray = (171 / 255.0, 193 / 255.0, 175 / 255.0)

    vec_dict = _lane_raw_to_vectors(lane_raw)
    t_now = int(np.clip(best_t, 0, all_agent.shape[0] - 1))
    ego_traj = all_agent[:, 0, :]
    ego_valid = ego_traj[:, 8] > 0.5
    valid_idx = np.where(ego_valid)[0]
    valid_idx = valid_idx[valid_idx <= t_now]
    if len(valid_idx) > 0:
        t_view = int(valid_idx[-1])  # Match Reverse_waymo center (last valid <= best_t)
    else:
        t_view = t_now
    ego_now = all_agent[t_view, 0]
    cx, cy = float(ego_now[0]), float(ego_now[1])
    ego_heading = float(ego_now[4])
    ego_l = max(float(ego_now[5]), 0.5)
    ego_w = max(float(ego_now[6]), 0.3)
    half = float(ego_margin)
    bbox = (cx - half, cy - half, cx + half, cy + half)

    ax.set_facecolor("white")
    ax.set_aspect("equal")
    _draw_vector_map(ax, vec_dict, bbox, alpha=1.0)

    # Overlap region overlay: orange highlight (was red in previous style).
    for i, (x1, y1, x2, y2) in enumerate(vectors):
        if i >= len(overlap_mask) or not overlap_mask[i]:
            continue
        rect = _lane_vec_polygon(x1, y1, x2, y2)
        if rect is None:
            continue
        ax.add_patch(MplPolygon(
            np.array(rect.exterior.coords[:-1]),
            closed=True,
            facecolor="#ffa500",
            edgecolor="#ff8c00",
            alpha=0.22,
            linewidth=1.0,
            zorder=380,
        ))
        ax.plot([x1, x2], [y1, y2], color="#ff8c00", linewidth=1.1, zorder=390)

    # Other agents (up to best_t)
    T_raw, N_agents, _ = all_agent.shape
    for ai in range(1, N_agents):
        ag_now = all_agent[t_view, ai]
        if len(ag_now) < 9 or ag_now[8] < 0.5 or int(ag_now[7]) != 1:
            continue
        ax_now, ay_now = float(ag_now[0]), float(ag_now[1])
        if (ax_now < bbox[0] - 5 or ax_now > bbox[2] + 5 or
            ay_now < bbox[1] - 5 or ay_now > bbox[3] + 5):
            continue
        ag_heading = float(ag_now[4])
        ag_l = max(float(ag_now[5]), 0.5)
        ag_w = max(float(ag_now[6]), 0.3)
        _draw_oriented_box(ax, np.array([ax_now, ay_now]), ag_heading, ag_l, ag_w,
                           color=light_gray, alpha=0.85, zorder=690,
                           edgecolor="black", edgewidth=1.5)

    # Ego at current timestep only
    _draw_oriented_box(ax, np.array([cx, cy]), ego_heading, ego_l, ego_w,
                       color="#1a73e8", alpha=0.85, zorder=700,
                       edgecolor="black", edgewidth=1.5)

    # Candidates (box + heading arrow)
    for pl in candidates:
        pcx, pcy = float(pl['cx']), float(pl['cy'])
        phdg = float(pl['heading'])
        pl_len = max(float(pl['length']), 0.5)
        pl_wid = max(float(pl['width']), 0.3)
        _draw_oriented_box(
            ax,
            np.array([pcx, pcy], dtype=np.float32),
            phdg, pl_len, pl_wid,
            color="#ff0000", alpha=0.85, zorder=690,
            edgecolor="black", edgewidth=1.5,
        )
        # Heading arrow
        arr_len = pl_len * 0.7
        ax.annotate(
            "", xy=(pcx + np.cos(phdg) * arr_len,
                    pcy + np.sin(phdg) * arr_len),
            xytext=(pcx, pcy),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
            zorder=700,
        )

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    _apply_ticks_and_grid(ax)
    ax.set_title(title, fontsize=10, pad=8)


def save_snapshot_plot(
    scenario_id: str,
    snapshot_debug: Dict,
    out_dir: str,
    ego_margin: float = 50.0,
) -> List[str]:
    """Save a plot of the selected collision snapshot."""
    if not snapshot_debug or not snapshot_debug.get('selected_snapshot'):
        return []

    os.makedirs(out_dir, exist_ok=True)
    lane_raw = snapshot_debug['lane_raw']
    all_agent = snapshot_debug['all_agent']
    vectors = snapshot_debug['vectors']
    overlap_mask = snapshot_debug['overlap_mask']
    best_t = snapshot_debug['best_t']
    selected = [snapshot_debug['selected_snapshot']]

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    _draw_snapshot_candidates(
        ax=ax,
        lane_raw=lane_raw,
        all_agent=all_agent,
        best_t=best_t,
        vectors=vectors,
        overlap_mask=overlap_mask,
        candidates=selected,
        title=f"{scenario_id} | t={best_t}\nSelected collision snapshot",
        ego_margin=ego_margin,
    )
    out_path = os.path.join(out_dir, f"{scenario_id}_snapshot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return [out_path]


# =========================================================================
# Map conversion: StateEstim lane_raw → ADV-BMT 27D format
# =========================================================================

def convert_lane_to_map27d(lane_raw, center_xy=None,
                           max_vectors=CVAE_MAX_VECTORS,
                           max_map_features=CVAE_MAX_MAP_FEATURES,
                           crop_range=50.0):
    """
    Convert StateEstim lane_raw (N, 4) → ADV-BMT 27D map features.

    Args:
        lane_raw: (N, 4) array with [x, y, type, lane_id]
        center_xy: (2,) reference point for centering; if None uses centroid
        max_vectors: max vectors per polyline segment
        max_map_features: max polyline segments
        crop_range: crop radius (m) around center_xy

    Returns:
        map_feature:  (M, V, 27) float32
        map_mask:     (M, V)     bool
        map_position: (M, 3)     float32  (avg position per polyline)
        map_heading:  (M,)       float32  (avg heading per polyline)
    """
    if center_xy is None:
        center_xy = lane_raw[:, :2].mean(axis=0)
    cx, cy = float(center_xy[0]), float(center_xy[1])

    # Group points by lane_id, preserving order
    lane_ids = lane_raw[:, 3].astype(int)
    unique_ids = []
    seen = set()
    for lid in lane_ids:
        if lid not in seen:
            unique_ids.append(lid)
            seen.add(lid)

    feat_list = []
    mask_list = []
    pos_list = []
    head_list = []

    for lid in unique_ids:
        pts_raw = lane_raw[lane_ids == lid]
        type_code = int(pts_raw[0, 2])
        pts = pts_raw[:, :2].copy()

        # Center the points
        pts[:, 0] -= cx
        pts[:, 1] -= cy

        if len(pts) < 2:
            # Single-point features (stop signs)
            start = np.array([pts[0, 0], pts[0, 1], 0.0])
            end = start.copy()
            direction = np.zeros(3)
            heading = 0.0
            pt_diff = 0.0

            feat_vec = np.zeros(27, dtype=np.float32)
            feat_vec[0:3] = start
            feat_vec[3:6] = end
            feat_vec[6:9] = direction
            feat_vec[9] = heading
            feat_vec[10] = np.sin(heading)
            feat_vec[11] = np.cos(heading)
            feat_vec[12] = pt_diff
            flags = _TYPE_FLAGS.get(type_code, (0,) * 12)
            feat_vec[13:25] = np.array(flags, dtype=np.float32)
            feat_vec[25] = 0.0
            feat_vec[26] = 1.0

            mf = np.zeros((max_vectors, 27), dtype=np.float32)
            mf[0] = feat_vec
            mm = np.zeros(max_vectors, dtype=bool)
            mm[0] = True

            avg_pos = np.array([start[0], start[1], 0.0], dtype=np.float32)
            if abs(avg_pos[0]) <= crop_range and abs(avg_pos[1]) <= crop_range:
                feat_list.append(mf)
                mask_list.append(mm)
                pos_list.append(avg_pos)
                head_list.append(0.0)
            continue

        # Build vectors from consecutive points
        starts_3d = np.column_stack([pts[:-1], np.zeros(len(pts) - 1)])
        ends_3d = np.column_stack([pts[1:], np.zeros(len(pts) - 1)])
        directions = ends_3d - starts_3d
        headings = np.arctan2(directions[:, 1], directions[:, 0])
        pt_diffs = np.linalg.norm(directions[:, :2], axis=1)

        # Type flags
        flags = np.array(_TYPE_FLAGS.get(type_code, (0,) * 12), dtype=np.float32)

        # Split into segments of max_vectors
        n_vecs = len(starts_3d)
        seg_start = 0
        road_length = 0.0

        while seg_start < n_vecs:
            seg_end = min(seg_start + max_vectors, n_vecs)
            n_valid = seg_end - seg_start

            mf = np.zeros((max_vectors, 27), dtype=np.float32)
            mm = np.zeros(max_vectors, dtype=bool)

            mf[:n_valid, 0:3] = starts_3d[seg_start:seg_end]
            mf[:n_valid, 3:6] = ends_3d[seg_start:seg_end]
            mf[:n_valid, 6:9] = directions[seg_start:seg_end]
            mf[:n_valid, 9] = headings[seg_start:seg_end]
            mf[:n_valid, 10] = np.sin(headings[seg_start:seg_end])
            mf[:n_valid, 11] = np.cos(headings[seg_start:seg_end])
            mf[:n_valid, 12] = pt_diffs[seg_start:seg_end]
            for d in range(12):
                mf[:n_valid, 13 + d] = flags[d]
            road_length += pt_diffs[seg_start:seg_end].sum()
            mf[:n_valid, 25] = road_length
            mf[:n_valid, 26] = 1.0
            mm[:n_valid] = True

            avg_pos = ((mf[:n_valid, 0:3] + mf[:n_valid, 3:6]) / 2).mean(axis=0)
            avg_heading = np.arctan2(
                np.sin(headings[seg_start:seg_end]).mean(),
                np.cos(headings[seg_start:seg_end]).mean(),
            )

            # Crop by range
            if abs(avg_pos[0]) <= crop_range and abs(avg_pos[1]) <= crop_range:
                feat_list.append(mf)
                mask_list.append(mm)
                pos_list.append(avg_pos.astype(np.float32))
                head_list.append(float(avg_heading))

            seg_start = seg_end

    if not feat_list:
        return (np.zeros((0, max_vectors, 27), dtype=np.float32),
                np.zeros((0, max_vectors), dtype=bool),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    # Limit to max_map_features (keep closest to center)
    positions = np.stack(pos_list)
    if len(feat_list) > max_map_features:
        dists = np.linalg.norm(positions[:, :2], axis=1)
        keep = np.argsort(dists)[:max_map_features]
        feat_list = [feat_list[i] for i in keep]
        mask_list = [mask_list[i] for i in keep]
        positions = positions[keep]
        head_list = [head_list[i] for i in keep]

    map_feature = np.stack(feat_list).astype(np.float32)
    map_mask = np.stack(mask_list).astype(bool)
    map_position = positions.astype(np.float32)
    map_heading = np.array(head_list, dtype=np.float32)

    return map_feature, map_mask, map_position, map_heading


# =========================================================================
# Build CVAE input from StateEstim data
# =========================================================================

def build_cvae_input(
    vehicle_state: Dict,
    t_collision: int,
    all_agent: np.ndarray,
    lane_raw: np.ndarray,
    device: torch.device = torch.device('cpu'),
    include_ego: bool = False,
    pred_horizon: int = None,
):
    """
    Build CVAE input tensors for a single inserted vehicle.

    Args:
        vehicle_state: dict with keys cx, cy, heading, length, width,
                       and StateEstim inference: speed, vel_heading
        t_collision: collision timestep index (0-based, in 10Hz)
        all_agent: (T, N, 9) full agent trajectories from StateEstim PKL
        lane_raw: (P, 4) lane data (global coords)
        device: torch device
        pred_horizon: number of 2Hz prediction steps (default: CVAE_PRED_HORIZON)

    Returns:
        dict of tensors ready for CVAE inference
    """
    if pred_horizon is None:
        pred_horizon = CVAE_PRED_HORIZON
    # ── Collision state ──
    col_x = float(vehicle_state['cx'])
    col_y = float(vehicle_state['cy'])
    col_heading = float(vehicle_state['heading'])
    speed = float(vehicle_state.get('speed', 0.0))
    vel_heading = float(vehicle_state.get('vel_heading', 0.0))

    # Velocity: v_dir = vel_heading + agent_heading
    v_dir = vel_heading + col_heading
    col_vx = speed * np.cos(v_dir)
    col_vy = speed * np.sin(v_dir)

    # Acceleration along heading direction (m/s²), read from vehicle_state
    acc_mps2 = float(vehicle_state.get('acc_mps2', 0.0))
    acc_disp_x = acc_mps2 * np.cos(v_dir) * CVAE_DT
    acc_disp_y = acc_mps2 * np.sin(v_dir) * CVAE_DT

    # Use collision point as centering origin
    center_xy = np.array([col_x, col_y])

    # Forward displacement per sub-sampled step (0.5s)
    fwd_disp_x = col_vx * CVAE_DT
    fwd_disp_y = col_vy * CVAE_DT

    # ── Build single-state observation (OB_HORIZON=0) ──
    # State format: [x, y, vx, vy, ax, ay]  (all centered)
    x_full = np.zeros((CVAE_OB_HORIZON + 1, 6), dtype=np.float32)
    # Single collision state (centered → origin)
    x_full[0, 0:2] = [0.0, 0.0]
    x_full[0, 2:4] = [fwd_disp_x, fwd_disp_y]  # forward disp/step
    x_full[0, 4:6] = [acc_disp_x, acc_disp_y]   # acceleration

    # ── Build neighbor trajectories ──
    T_total, N_agents, _ = all_agent.shape
    total_neigh_steps = CVAE_OB_HORIZON + pred_horizon + 1

    # Sub-sampled reverse-time steps (from collision backward)
    rev_steps = []
    for s in range(total_neigh_steps):
        fwd_step = t_collision - s * CVAE_SKIP
        rev_steps.append(max(0, fwd_step))

    # Collect other agents (optionally include ego = index 0)
    neighbor_list = []
    for ai in range(N_agents):
        if ai == 0 and not include_ego:
            continue  # skip ego
        n_states = []
        for fwd_step in rev_steps:
            ag = all_agent[fwd_step, ai]
            if ag[8] < 0.5 or int(ag[7]) != 1:
                n_states.append([1e9] * 6)
                continue
            # Center position, keep velocity as-is (m/s)
            px = float(ag[0]) - col_x
            py = float(ag[1]) - col_y
            vx = float(ag[2])
            vy = float(ag[3])
            n_states.append([px, py, vx, vy, 0.0, 0.0])
        neighbor_list.append(n_states)

    if not neighbor_list:
        neighbors = np.full((total_neigh_steps, 1, 6), 1e9, dtype=np.float32)
    else:
        neighbors = np.array(neighbor_list, dtype=np.float32)  # (Nn, L, 6)
        neighbors = neighbors.transpose(1, 0, 2)               # (L, Nn, 6)

    # Filter neighbors by distance (keep within OB_RADIUS at any step)
    n_pos = neighbors[..., :2]
    ego_pos = np.zeros((total_neigh_steps, 1, 2))
    # ego position trace in centered coords (approximate: constant extrapolation)
    for s in range(total_neigh_steps):
        ego_pos[s, 0] = [0.0 - s * fwd_disp_x, 0.0 - s * fwd_disp_y]
    dist = np.linalg.norm(n_pos - ego_pos, axis=-1)
    within = np.any(dist <= CVAE_OB_RADIUS, axis=0)
    neighbors = neighbors[:, within]
    if neighbors.shape[1] == 0:
        neighbors = np.full((total_neigh_steps, 1, 6), 1e9, dtype=np.float32)
    if neighbors.shape[1] > CVAE_MAX_NEIGHBORS:
        avg_d = np.nanmean(np.where(dist[:, within] < 1e8,
                                     dist[:, within], np.nan), axis=0)
        keep = np.argsort(avg_d)[:CVAE_MAX_NEIGHBORS]
        neighbors = neighbors[:, keep]

    # ── Build 27D map features (centered on collision point) ──
    # Prefer unsampled_lane for better resolution
    map_feature, map_mask, map_position, map_heading = convert_lane_to_map27d(
        lane_raw, center_xy=center_xy, crop_range=CVAE_OB_RADIUS,
    )

    # ── Convert to tensors ──
    # x: (OB+1, 1, 6)
    x_t = torch.tensor(x_full, dtype=torch.float32).unsqueeze(1)
    # neighbor: (L, 1, Nn, 6)
    n_t = torch.tensor(neighbors, dtype=torch.float32).unsqueeze(1)
    # map_feature: (1, M, V, 27)
    mf_t = torch.tensor(map_feature, dtype=torch.float32).unsqueeze(0)
    mm_t = torch.tensor(map_mask, dtype=torch.bool).unsqueeze(0)
    mp_t = torch.tensor(map_position, dtype=torch.float32).unsqueeze(0)
    mh_t = torch.tensor(map_heading, dtype=torch.float32).unsqueeze(0)

    return {
        'x': x_t.to(device),
        'neighbor': n_t.to(device),
        'map_feature': mf_t.to(device),
        'map_mask': mm_t.to(device),
        'map_position': mp_t.to(device),
        'map_heading': mh_t.to(device),
        'center_xy': center_xy,
    }


# =========================================================================
# Pipeline class
# =========================================================================

class StateEstimCVAEPipeline:
    """
    End-to-end pipeline: StateEstim vehicle placement → CVAE backward trajectory.
    """

    def __init__(
        self,
        state_estim_ckpt: str = None,
        cvae_ckpt: str = None,
        cvae_config_path: str = None,
        refine_vocab: str = None,
        device: str = 'cpu',
        max_selected: int = 1,
        min_t: int = 0,
    ):
        self.device = torch.device(device)
        self.max_selected = max_selected
        self.min_t = min_t
        self.last_stage_debug = None

        # Default paths
        if state_estim_ckpt is None:
            state_estim_ckpt = str(_THIS_DIR / 'ckpt' / 'v4bi_ep049.ckpt')
        if cvae_ckpt is None:
            cvae_ckpt = str(_CVAE_DIR / 'ckpt' / 'ckpt-best')
        if cvae_config_path is None:
            cvae_config_path = str(_COSTER_DIR / 'cvae' / 'config.py')
        if refine_vocab is None:
            refine_vocab = os.environ.get("COSTER_REFINE_VOCAB", REFINE_VOCAB_PATH)

        if not os.path.isfile(state_estim_ckpt):
            raise FileNotFoundError(
                f"StateEstim checkpoint not found: {state_estim_ckpt}. "
                "Pass --state_estim_ckpt or place weights under state_estim/ckpt/."
            )
        if not os.path.isfile(cvae_ckpt):
            raise FileNotFoundError(
                f"CVAE checkpoint not found: {cvae_ckpt}. "
                "Pass --cvae_ckpt or train/download weights under cvae/ckpt/."
            )

        # ── Load StateEstim init model ──
        print("Loading StateEstim init model …")
        self.tg_engine = GlobalInitInference(
            ckpt_path=state_estim_ckpt,
            device=device,
            map_size=50,
            anchor_min_spacing=40,
        )

        # ── Load CVAE model ──
        print("Loading CVAE model …")
        spec = importlib.util.spec_from_file_location(
            "cvae_config", cvae_config_path,
            submodule_search_locations=[os.path.dirname(cvae_config_path)],
        )
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        self.cvae_config = config

        self.cvae_model = VectorMapCVAE(**config.model)
        state = torch.load(cvae_ckpt, map_location=self.device,
                           weights_only=False)
        self.cvae_model.load_state_dict(state['model'])
        ade = state.get('ade', '?')
        fde = state.get('fde', '?')
        epoch = state.get('epoch', '?')
        print(f"  CVAE loaded: ADE={ade}, FDE={fde}, epoch={epoch}")
        self.cvae_model.to(self.device)
        self.cvae_model.eval()

        # ── Load Refine vocab (once) ──
        if os.path.isfile(refine_vocab):
            vocab_data = np.load(refine_vocab)
            self._refine_templates = vocab_data["templates"]  # (K, 4)
        else:
            self._refine_templates = None
            print(f"  WARNING: vocab not found at {refine_vocab}; refine snapping will be skipped.")
        self._physics_limits = PhysicsLimits()
        print("Models ready.\n")

    # ------------------------------------------------------------------
    def _run_candidate_pipeline(self, veh, best_t, all_agent,
                                lane_for_cvae, lane_raw,
                                traffic_light=None):
        """Run CVAE → K-means representative selection → refine pipeline.

        CVAE rollouts are summarized by lateral-displacement and arc-length
        features. K-means is applied in that feature space, then only the
        nearest-to-centroid representative of each cluster is refined.

        Returns dict with  accepted_fwd, cluster_info, pred_global,
        pred_centered, penalty, n_total, n_accept, reject_count
        or *None* on failure.
        """
        import math

        collision_pos_np = np.array([veh['cx'], veh['cy']])

        # Adaptive step count: cover all timesteps before collision
        n_steps = max(1, math.ceil(best_t / CVAE_SKIP))

        # Build CVAE input with adaptive pred_horizon
        cvae_inp = build_cvae_input(
            vehicle_state=veh,
            t_collision=best_t,
            all_agent=all_agent,
            lane_raw=lane_for_cvae,
            device=self.device,
            pred_horizon=n_steps,
        )
        center = cvae_inp['center_xy']

        # Run CVAE inference (stochastic samples)
        all_preds_centered = []
        batch_size = 2000
        remaining = CVAE_N_PREDICTIONS
        with torch.no_grad():
            while remaining > 0:
                bs = min(batch_size, remaining)
                pred = self.cvae_model(
                    x=cvae_inp['x'],
                    neighbor=cvae_inp['neighbor'],
                    map_feature=cvae_inp['map_feature'],
                    map_mask=cvae_inp['map_mask'],
                    map_position=cvae_inp['map_position'],
                    map_heading=cvae_inp['map_heading'],
                    n_predictions=bs,
                    decode_steps=n_steps,
                )
                if pred.dim() == 4:
                    pred_np = pred.squeeze(2).cpu().numpy()
                else:
                    pred_np = pred.squeeze(1).cpu().numpy()
                    pred_np = pred_np[None, ...]
                all_preds_centered.append(pred_np)
                remaining -= bs

        all_preds_centered = np.concatenate(
            all_preds_centered, axis=0)[:CVAE_N_PREDICTIONS]
        n_total = len(all_preds_centered)

        templates = self._refine_templates
        physics_limits = self._physics_limits

        # ---- TL compliance helper ----
        def _tl_penalty_for_traj(traj_fwd_pts):
            """Return TL violation count for one forward-time trajectory."""
            if traffic_light is None:
                return 0
            q = _evaluate_tl_compliance(
                traj_fwd_pts, lane_raw, traffic_light, best_t)
            return q['n_violations']

        # ---- Helper: refine a single trajectory by index ----
        def _refine_one_idx(si):
            """Refine trajectory at index *si* in all_preds_centered.
            Returns (ref_pos, ref_head) on success, None on failure."""
            raw_centered = all_preds_centered[si]
            raw_global = raw_centered + center
            fwd_global = raw_global[::-1].copy()
            fwd_with_col = np.concatenate(
                [fwd_global, collision_pos_np[None, :]], axis=0)
            if templates is None:
                return fwd_with_col, None
            ref_pos, ref_head, ref_spd, snap_dists, labels, past_drift = \
                refine_trajectory(
                    fwd_with_col, templates,
                    vehicle_length=veh['length'],
                    vehicle_width=veh['width'],
                    dt=CVAE_DT,
                )
            is_valid, reasons = full_rejection_check(
                snap_dists, ref_pos, ref_head, ref_spd,
                past_drift=past_drift, dt=CVAE_DT,
                limits=physics_limits,
            )
            if not is_valid:
                return None
            return ref_pos, ref_head

        # ================================================================
        # STEP 1: Convert raw CVAE samples to forward-time trajectories.
        # ================================================================
        all_fwd_global = []
        for si in range(n_total):
            raw_centered = all_preds_centered[si]
            raw_global = raw_centered + center
            fwd_global = raw_global[::-1].copy()
            fwd_with_col = np.concatenate(
                [fwd_global, collision_pos_np[None, :]], axis=0)
            all_fwd_global.append(fwd_with_col)

        if n_total < 3:
            if n_total == 0:
                return None

            # Too few for K-means: refine all and use the first valid sample.
            accepted_fwd = []
            accepted_indices = []
            for si in range(n_total):
                result = _refine_one_idx(si)
                if result is not None:
                    ref_pos, _ = result
                    accepted_fwd.append(ref_pos)
                    accepted_indices.append(si)
            n_accept = len(accepted_fwd)
            if n_accept == 0:
                pred_centered = all_preds_centered[0]
                pred_global = pred_centered + center
                fwd_fb = np.concatenate(
                    [pred_global[::-1], collision_pos_np[None, :]], axis=0)
                penalty = _tl_penalty_for_traj(fwd_fb)
                return {
                    'accepted_fwd': [], 'cluster_info': None,
                    'all_representative_globals': [pred_global],
                    'all_representative_centered': [pred_centered],
                    'penalty': penalty, 'n_total': n_total,
                    'n_accept': 0, 'reject_count': n_total,
                }
            ref_traj = accepted_fwd[0]
            pred_global = ref_traj[:-1][::-1].copy()
            pred_centered = pred_global - center
            cluster_info = {
                'n_total': n_total, 'n_accept': n_accept,
                'n_clusters': n_accept,
                'labels': np.arange(n_accept, dtype=int),
                'chosen_idx': 0,
                'representative_indices': list(range(n_accept)),
                'cluster_representatives': {
                    int(i): int(i) for i in range(n_accept)
                },
                'all_accepted_fwd': accepted_fwd,
            }
            penalty = _tl_penalty_for_traj(ref_traj)
            return {
                'accepted_fwd': accepted_fwd, 'cluster_info': cluster_info,
                'all_representative_globals': [pred_global],
                'all_representative_centered': [pred_centered],
                'penalty': penalty, 'n_total': n_total,
                'n_accept': n_accept, 'reject_count': n_total - n_accept,
            }

        # ================================================================
        # STEP 2: K-means on lateral-displacement / arc-length features.
        # ================================================================
        trajs_raw = np.stack(all_fwd_global, axis=0)
        features = _trajectory_lateral_arc_features(trajs_raw, veh['heading'])
        k = min(KMEANS_N_CLUSTERS, n_total)
        kmeans_labels, centroids = _kmeans_labels(features, k)
        representative_indices, cluster_sizes = _kmeans_representatives(
            features, kmeans_labels, centroids)
        unique_cls = sorted(
            representative_indices,
            key=lambda cl: cluster_sizes[cl],
            reverse=True,
        )
        n_clusters = len(unique_cls)

        # ================================================================
        # STEP 3: Refine ONLY the representative of each cluster.
        # ================================================================
        # If the representative fails, try other members nearest to the same
        # centroid in the lateral/arc-length feature space.
        _MAX_FALLBACK = 5  # max retries per cluster if representative refine fails

        from concurrent.futures import ThreadPoolExecutor, as_completed

        representative_to_cluster = {
            representative_indices[cl]: cl for cl in unique_cls
        }
        indices_list = list(representative_to_cluster)

        # Parallel refine of representative candidates
        refined_representatives = {}   # cluster_label -> (ref_pos, ref_head, raw_si)
        failed_clusters = []

        _N_WORKERS = min(16, max(1, len(indices_list)))

        def _refine_worker(si):
            return si, _refine_one_idx(si)

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            futures = {executor.submit(_refine_worker, si): si
                       for si in indices_list}
            refine_results = {}
            for fut in as_completed(futures):
                si, result = fut.result()
                refine_results[si] = result

        # Map results back to clusters
        for si, cl in representative_to_cluster.items():
            if refine_results.get(si) is not None:
                ref_pos, ref_head = refine_results[si]
                refined_representatives[cl] = (ref_pos, ref_head, si)
            else:
                failed_clusters.append(cl)

        # Second pass: for failed clusters, try fallback members
        standardized_features = _standardize_features(features)
        for cl in failed_clusters:
            cl_idx = np.where(kmeans_labels == cl)[0]
            rep_si = representative_indices[cl]
            dists = np.linalg.norm(
                standardized_features[cl_idx] - centroids[cl], axis=1)
            order = cl_idx[np.argsort(dists)]
            candidates_to_try = [
                int(i) for i in order if int(i) != rep_si
            ][:_MAX_FALLBACK]
            for fb_si in candidates_to_try:
                result = _refine_one_idx(fb_si)
                if result is not None:
                    ref_pos, ref_head = result
                    refined_representatives[cl] = (ref_pos, ref_head, fb_si)
                    break

        # Count how many refine attempts total
        n_refine_attempts = len(indices_list) + sum(
            min(len(np.where(kmeans_labels == cl)[0]) - 1, _MAX_FALLBACK)
            for cl in failed_clusters
        )
        n_refine_success = len(refined_representatives)
        reject_count = n_refine_attempts - n_refine_success

        # ================================================================
        # STEP 4: Build output from refined representatives.
        # ================================================================
        all_representative_globals = []
        all_representative_centered = []
        accepted_fwd = []
        penalty = 0

        for cl in unique_cls:
            if cl not in refined_representatives:
                continue
            ref_pos, ref_head, raw_si = refined_representatives[cl]
            accepted_fwd.append(ref_pos)
            mg = ref_pos[:-1][::-1].copy()
            all_representative_globals.append(mg)
            all_representative_centered.append(mg - center)
            penalty += _tl_penalty_for_traj(ref_pos)

        n_accept = len(accepted_fwd)

        if n_accept == 0:
            # All representative refines failed: use the largest-cluster representative.
            best_si = representative_indices[unique_cls[0]]
            pred_centered = all_preds_centered[best_si]
            pred_global = pred_centered + center
            fwd_fallback = np.concatenate(
                [pred_global[::-1], collision_pos_np[None, :]], axis=0)
            penalty = _tl_penalty_for_traj(fwd_fallback)
            return {
                'accepted_fwd': [], 'cluster_info': None,
                'all_representative_globals': [pred_global],
                'all_representative_centered': [pred_centered],
                'penalty': penalty, 'n_total': n_total,
                'n_accept': 0, 'reject_count': reject_count,
            }

        # Build cluster_info compatible with downstream
        cluster_representatives_final = {
            cl: refined_representatives[cl][2]
            for cl in unique_cls
            if cl in refined_representatives
        }
        chosen_idx = cluster_representatives_final.get(unique_cls[0], 0)

        cluster_info = {
            'n_total': n_total, 'n_accept': n_accept,
            'n_clusters': n_clusters,
            'labels': kmeans_labels,
            'features': features,
            'feature_names': ['lateral_displacement', 'arc_length'],
            'representative_indices': [
                representative_indices[cl] for cl in unique_cls
            ],
            'cluster_representatives': cluster_representatives_final,
            'cluster_sizes': cluster_sizes,
            'chosen_idx': chosen_idx,
            'all_accepted_fwd': all_fwd_global,
        }

        return {
            'accepted_fwd': accepted_fwd, 'cluster_info': cluster_info,
            'all_representative_globals': all_representative_globals,
            'all_representative_centered': all_representative_centered,
            'penalty': penalty, 'n_total': n_total,
            'n_accept': n_accept, 'reject_count': reject_count,
        }

    # ------------------------------------------------------------------
    def process_scenario(self, pkl_path: str) -> List[Dict]:
        """
        Process a single Waymo scenario.

        Returns list[dict], each with:
            collision_state:  dict (cx, cy, vx, vy, heading, length, width)
            collision_timestep: int (10Hz index)
            prob:              float
            position_log_prob: float
            trajectory_2hz:    (10, 2) centered positions (collision → past)
            trajectory_global: (10, 2) global positions
            trajectory_10hz:   (T, 2) interpolated 10Hz global positions
        """
        _ensure_pickle_compat()
        self.last_stage_debug = None

        # ── Load raw data ──
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        all_agent = np.array(data['all_agent'])     # (T, N, 9)
        lane_raw = data['lane']                     # (P, 4)
        traffic_light = data.get('traffic_light', None)  # list[list[ndarray]]
        # Use unsampled_lane if available (better resolution for CVAE + plotting)
        lane_for_cvae = data.get('unsampled_lane', lane_raw)
        lane_for_plot = np.array(data.get('unsampled_lane', lane_raw))

        # ── Select collision time by velocity entropy ──
        best_t, scores_arr, _, _ = select_best_timestep(
            pkl_path, engine=self.tg_engine, min_t=self.min_t)
        best_score = float(scores_arr[best_t]) if best_t < len(scores_arr) else 0.0
        print(f"  Selected timestep: t={best_t} (velocity_entropy={best_score:.3f})")

        # ── StateEstim inference at selected timestep ──
        results, lane_vis, agents_t0 = self.tg_engine.run(
            pkl_path, timestep=best_t)
        infer_lookup = build_inference_lookup(results)
        if infer_lookup is None or infer_lookup[0] is None:
            print("  No inference results.")
            return []
        infer_pos, infer_states = infer_lookup

        # ── Ego info ──
        ego = agents_t0[0]
        ex, ey = float(ego[0]), float(ego[1])
        heading = float(ego[4])
        ego_l, ego_w = float(ego[5]), float(ego[6])

        # ego ellipse for overlap detection
        _unit = Point(0, 0).buffer(1.0, resolution=64)
        _scaled = shapely_scale(_unit, xfact=EGO_SEMI_LONG, yfact=EGO_SEMI_LAT)
        _rotated = shapely_rotate(_scaled, np.degrees(heading))
        ego_ellipse = shapely_translate(_rotated, ex, ey)

        vectors = extract_center_vectors(lane_vis)
        overlap_mask = find_overlapping_mask(vectors, ego_ellipse)

        # ── Select collision snapshot by maximum Bernoulli region probability ──
        inside = np.array([
            ego_ellipse.contains(Point(float(x), float(y)))
            for x, y in infer_pos
        ])
        if not inside.any():
            self.last_stage_debug = {
                'lane_raw': lane_for_plot, 'all_agent': all_agent,
                'vectors': vectors, 'overlap_mask': overlap_mask,
                'agents_t0': agents_t0, 'best_t': best_t,
                'selected_snapshot': None,
            }
            print("  No StateEstim regions inside target ellipse.")
            return []

        region_probs = infer_states['prob']
        inside_idx = np.where(inside)[0]
        selected_idx = int(inside_idx[np.argmax(region_probs[inside])])
        selected_state = {k: v[selected_idx] for k, v in infer_states.items()}
        selected_state['global_positions'] = infer_pos[selected_idx]
        veh = _paper_snapshot_from_region(ego, selected_state)
        print(f"  Selected region: prob={veh['prob']:.3f}, "
              f"pos_logp={veh['position_log_prob']:.3f}, "
              f"center=({veh['cx']:.2f}, {veh['cy']:.2f})")

        try:
            res = self._run_candidate_pipeline(
                veh, best_t, all_agent, lane_for_cvae, lane_raw,
                traffic_light=traffic_light)
        except Exception as e:
            print(f"  CVAE/refine pipeline failed: {e}")
            res = None
        if res is None:
            self.last_stage_debug = {
                'lane_raw': lane_for_plot, 'all_agent': all_agent,
                'vectors': vectors, 'overlap_mask': overlap_mask,
                'agents_t0': agents_t0, 'best_t': best_t,
                'selected_snapshot': veh,
            }
            return []

        res['veh'] = veh
        final_selected = [res]
        self.last_stage_debug = {
            'lane_raw': lane_for_plot, 'all_agent': all_agent,
            'vectors': vectors, 'overlap_mask': overlap_mask,
            'agents_t0': agents_t0, 'best_t': best_t,
            'selected_snapshot': veh,
            'snapshot_result': res,
        }

        # ── Build output from pre-computed pipeline results ──
        # Each selected candidate emits one vehicle per refined representative.
        output_vehicles = []
        for rank, res in enumerate(final_selected):
            veh = res['veh']
            cluster_info = res['cluster_info']

            v_dir = veh['vel_heading'] + veh['heading']
            col_vx = veh['speed'] * np.cos(v_dir)
            col_vy = veh['speed'] * np.sin(v_dir)
            collision_pos_np = np.array([veh['cx'], veh['cy']])

            representative_globals = res['all_representative_globals']
            representative_centered = res['all_representative_centered']
            n_representatives = len(representative_globals)
            print(f"  Selected candidate rank {rank+1}: "
                  f"{n_representatives} representative trajectory(ies)")

            # Pick the representative from the largest K-means cluster.
            best_mi = 0
            pred_global = representative_globals[best_mi]
            pred_centered = representative_centered[best_mi]

            traj_10hz = _interpolate_to_10hz(
                collision_pos=collision_pos_np,
                pred_2hz=pred_global,
                t_collision=best_t,
                skip=CVAE_SKIP,
            )
            output_vehicles.append({
                'collision_state': {
                    'x': veh['cx'], 'y': veh['cy'],
                    'vx': col_vx, 'vy': col_vy,
                    'heading': veh['heading'],
                    'length': veh['length'], 'width': veh['width'],
                },
                'collision_timestep': best_t,
                'prob': veh['prob'],
                'position_log_prob': veh['position_log_prob'],
                'trajectory_2hz': pred_centered,
                'trajectory_global': pred_global,
                'trajectory_10hz': traj_10hz,
                'cluster_info': cluster_info,
            })

        return output_vehicles


# =========================================================================
# Interpolation: 2Hz → 10Hz
# =========================================================================

def _interpolate_to_10hz(collision_pos, pred_2hz, t_collision, skip=5):
    """
    Interpolate 2Hz reverse-time predictions to 10Hz.

    Args:
        collision_pos: (2,) collision global position
        pred_2hz: (horizon, 2) predicted positions at 2Hz (reverse time)
        t_collision: collision timestep in 10Hz
        skip: sub-sampling factor

    Returns:
        dict with 't_indices' and 'positions' arrays at 10Hz
    """
    # Key-frame timesteps (10Hz): collision, collision-5, collision-10, ...
    keyframes = [t_collision]
    positions = [collision_pos.copy()]
    for i in range(len(pred_2hz)):
        t = t_collision - (i + 1) * skip
        if t < 0:
            break
        keyframes.append(t)
        positions.append(pred_2hz[i])

    keyframes = np.array(keyframes)
    positions = np.array(positions)

    if len(keyframes) < 2:
        return {'t_indices': keyframes, 'positions': positions}

    # Sort ascending by timestep
    order = np.argsort(keyframes)
    keyframes = keyframes[order]
    positions = positions[order]

    # Interpolate
    all_t = []
    all_pos = []
    for i in range(len(keyframes) - 1):
        t0, t1 = keyframes[i], keyframes[i + 1]
        p0, p1 = positions[i], positions[i + 1]
        for t in range(t0, t1):
            alpha = (t - t0) / (t1 - t0)
            all_t.append(t)
            all_pos.append((1 - alpha) * p0 + alpha * p1)
    # Add last keyframe
    all_t.append(keyframes[-1])
    all_pos.append(positions[-1])

    return {
        't_indices': np.array(all_t),
        'positions': np.array(all_pos),
    }


# =========================================================================
# Export to Reverse_waymo plotter format
# =========================================================================

# StateEstim type → category mapping for vec.pkl
_LANE_TYPE_CODES = {1, 2, 3}               # LANE_FREEWAY/SURFACE/BIKE
_ROAD_LINE_TYPE_CODES = {6, 7, 8, 9, 10, 11, 12, 13}
_ROAD_EDGE_TYPE_CODES = {15, 16}            # ROAD_EDGE_BOUNDARY/MEDIAN
_CROSSWALK_TYPE_CODES = {18}
_AGENT_TYPE_NAMES = {1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST'}
_SENTINEL = 1e9

_REVERSE_WAYMO_ROOT = os.environ.get('REVERSE_WAYMO_ROOT')
_PLOT_SCRIPT = (
    os.path.join(_REVERSE_WAYMO_ROOT, 'scripts', 'plot_waymo_vector_results.py')
    if _REVERSE_WAYMO_ROOT else None
)
# Inserted vehicle colors (per rank)
_INSERT_COLORS = ['#ff0000', '#ff6600', '#ff00ff', '#00ccff', '#ffcc00']


def reverse_waymo_available() -> bool:
    """Return True when optional Reverse Waymo plotting helpers are available."""
    return bool(_PLOT_SCRIPT and os.path.exists(_PLOT_SCRIPT))


def _reverse_waymo_missing_message() -> str:
    if not _REVERSE_WAYMO_ROOT:
        return "REVERSE_WAYMO_ROOT is not set; optional Reverse Waymo plots are disabled."
    return f"Reverse Waymo plotter not found: {_PLOT_SCRIPT}"


def _lane_raw_to_vectors(lane_raw: np.ndarray) -> dict:
    """
    Convert StateEstim ``lane_raw``  *(N, 4)*  to the ``vectors`` dict
    expected by Reverse_waymo's ``<sid>.vec.pkl``.

    Returns dict with keys ``lanes``, ``road_lines``, ``road_edges``,
    ``crosswalks`` – each a list of ``(P, 2) float32`` polylines.
    """
    vectors: Dict[str, list] = {
        'lanes': [], 'road_lines': [], 'road_edges': [], 'crosswalks': [],
    }
    lane_ids = lane_raw[:, 3].astype(int)
    seen, order = set(), []
    for lid in lane_ids:
        if lid not in seen:
            order.append(lid)
            seen.add(lid)
    for lid in order:
        mask = lane_ids == lid
        pts = lane_raw[mask]
        tc = int(pts[0, 2])
        xy = pts[:, :2].astype(np.float32)
        if len(xy) < 2:
            continue
        if tc in _LANE_TYPE_CODES:
            vectors['lanes'].append(xy)
        elif tc in _ROAD_LINE_TYPE_CODES:
            vectors['road_lines'].append(xy)
        elif tc in _ROAD_EDGE_TYPE_CODES:
            vectors['road_edges'].append(xy)
        elif tc in _CROSSWALK_TYPE_CODES:
            vectors['crosswalks'].append(xy)
        else:
            vectors['lanes'].append(xy)
    return vectors


def export_scenario_for_plot(
    scenario_id: str,
    pkl_path: str,
    vehicles: List[Dict],
    best_t: int,
    dump_dir: str,
    vec_root: str,
    vec_subdir: str = 'pipeline',
) -> Optional[Dict]:
    """
    Export pipeline results to Reverse_waymo dump + vec PKL format.

    Creates::

        dump_dir/{scenario_id}.pkl
        vec_root/{vec_subdir}/{scenario_id}.vec.pkl

    Returns metadata dict (paths, inserted indices) or *None* on failure.
    """
    _ensure_pickle_compat()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    all_agent = np.array(data['all_agent'])  # (T, N, 9)
    lane_raw = np.array(data.get('unsampled_lane', data['lane']))
    T_raw, N_orig, _ = all_agent.shape

    # ── Ego trajectory ─────────────────────────────────────────────────
    ego_traj = all_agent[:, 0, :]   # (T, 9)
    ego_valid = ego_traj[:, 8] > 0.5
    valid_idx = np.where(ego_valid)[0]
    if len(valid_idx) < 4:
        print(f"    Export skip: ego valid too short ({len(valid_idx)})")
        return None

    # ── CUT at collision timestep ──────────────────────────────────────
    # Only keep timesteps up to and including best_t so that every
    # trajectory (ego + all neighbours + inserted vehicle) ends at
    # the collision moment.
    valid_idx = valid_idx[valid_idx <= best_t]
    if len(valid_idx) < 3:
        print(f"    Export skip: ego valid before collision too short "
              f"({len(valid_idx)})")
        return None

    ego_xy = ego_traj[valid_idx, :2].astype(np.float32)

    # hist = everything before the collision point
    # fut  = collision point only  (plotter needs non-empty fut)
    hist = ego_xy[:-1]
    fut = ego_xy[-1:]
    total_len = len(ego_xy)          # = len(hist) + len(fut)
    steps = valid_idx                 # absolute timesteps, all ≤ best_t

    # ── Existing neighbours ────────────────────────────────────────────
    neighbors_data: List[np.ndarray] = []
    neighbor_types: List[str] = []
    for ai in range(1, N_orig):
        traj_pts = []
        any_valid_flag = False
        for j, t in enumerate(steps):
            if t < T_raw:
                ag = all_agent[t, ai]
                if ag[8] > 0.5:
                    traj_pts.append(ag[:2].astype(np.float32))
                    any_valid_flag = True
                else:
                    traj_pts.append(
                        np.array([_SENTINEL, _SENTINEL], dtype=np.float32))
            else:
                traj_pts.append(
                    np.array([_SENTINEL, _SENTINEL], dtype=np.float32))
        if any_valid_flag:
            neighbors_data.append(np.array(traj_pts))
            # Resolve type from first valid step
            for t in steps:
                if t < T_raw and all_agent[t, ai, 8] > 0.5:
                    atype = int(all_agent[t, ai, 7])
                    neighbor_types.append(
                        _AGENT_TYPE_NAMES.get(atype, 'VEHICLE'))
                    break
            else:
                neighbor_types.append('VEHICLE')

    # ── Append inserted vehicles (CVAE backward trajectories) ──────────
    inserted_indices: List[int] = []
    for veh in vehicles:
        traj_10hz = veh['trajectory_10hz']
        t_idx_arr = traj_10hz['t_indices']
        positions = traj_10hz['positions']

        ins_traj = np.full((total_len, 2), _SENTINEL, dtype=np.float32)
        for ti, pos in zip(t_idx_arr, positions):
            local = int(np.searchsorted(steps, ti))
            if local < total_len and steps[local] == ti:
                ins_traj[local] = pos.astype(np.float32)

        # Ensure collision point itself is present
        col = veh['collision_state']
        col_local = int(np.searchsorted(steps, veh['collision_timestep']))
        if col_local < total_len and steps[col_local] == veh['collision_timestep']:
            ins_traj[col_local] = np.array(
                [col['x'], col['y']], dtype=np.float32)

        inserted_indices.append(len(neighbors_data))
        neighbors_data.append(ins_traj)
        neighbor_types.append('VEHICLE')

    # Build neighbour array (T_total, K, 2)
    if neighbors_data:
        neighbor = np.stack(neighbors_data, axis=1)
    else:
        neighbor = np.full((total_len, 1, 2), _SENTINEL, dtype=np.float32)
        neighbor_types = ['VEHICLE']

    # ── Collect actual bounding box sizes ─────────────────────────────
    #   ego_size:      (length, width, heading)
    #   neighbor_sizes: {neighbor_index: (length, width, heading)} for
    #                   inserted vehicles (and optionally others)
    ego_at_col = all_agent[steps[-1], 0]
    ego_size = (float(ego_at_col[5]), float(ego_at_col[6]),
                float(ego_at_col[4]))  # (L, W, heading)

    neighbor_sizes: Dict[int, tuple] = {}
    for veh, nidx in zip(vehicles, inserted_indices):
        col = veh['collision_state']
        neighbor_sizes[nidx] = (float(col['length']), float(col['width']),
                                float(col['heading']))

    # ── Save dump PKL ──────────────────────────────────────────────────
    dump = {
        'hist': hist,
        'fut': fut,
        'neighbor': neighbor,
        'neighbor_type': neighbor_types,
        'neighbor_ids': [str(i) for i in range(len(neighbor_types))],
        'inserted_neighbor_index': inserted_indices[0] if inserted_indices else None,
        'pred5': None,
        'pred_many': None,
        'map': scenario_id,
        'angle': None,
        'ego_abs_pos': None,
        'ego_yaw': None,
        # Actual bounding box sizes for correct rendering
        'ego_size': ego_size,
        'neighbor_sizes': neighbor_sizes,
    }
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f"{scenario_id}.pkl")
    with open(dump_path, 'wb') as f:
        pickle.dump(dump, f)

    # ── Save vec PKL ───────────────────────────────────────────────────
    vectors = _lane_raw_to_vectors(lane_raw)
    full_vec_dir = os.path.join(vec_root, vec_subdir)
    os.makedirs(full_vec_dir, exist_ok=True)
    vec_path = os.path.join(full_vec_dir, f"{scenario_id}.vec.pkl")
    with open(vec_path, 'wb') as f:
        pickle.dump({'vectors': vectors}, f)

    return {
        'dump_path': dump_path,
        'vec_path': vec_path,
        'scenario_id': scenario_id,
        'inserted_indices': inserted_indices,
    }


def export_original_scenario_for_plot(
    scenario_id: str,
    pkl_path: str,
    best_t: int,
    dump_dir: str,
    vec_root: str,
    vec_subdir: str = 'original',
) -> Optional[Dict]:
    """Export the *original* scenario (no inserted vehicles) for plotting."""
    _ensure_pickle_compat()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    all_agent = np.array(data['all_agent'])
    lane_raw = np.array(data.get('unsampled_lane', data['lane']))
    T_raw, N_orig, _ = all_agent.shape

    ego_traj = all_agent[:, 0, :]
    ego_valid = ego_traj[:, 8] > 0.5
    valid_idx = np.where(ego_valid)[0]
    valid_idx = valid_idx[valid_idx <= best_t]
    if len(valid_idx) < 3:
        return None

    ego_xy = ego_traj[valid_idx, :2].astype(np.float32)
    hist = ego_xy[:-1]
    fut = ego_xy[-1:]
    total_len = len(ego_xy)
    steps = valid_idx

    neighbors_data: List[np.ndarray] = []
    neighbor_types: List[str] = []
    for ai in range(1, N_orig):
        traj_pts = []
        any_valid_flag = False
        for j, t in enumerate(steps):
            if t < T_raw:
                ag = all_agent[t, ai]
                if ag[8] > 0.5:
                    traj_pts.append(ag[:2].astype(np.float32))
                    any_valid_flag = True
                else:
                    traj_pts.append(
                        np.array([_SENTINEL, _SENTINEL], dtype=np.float32))
            else:
                traj_pts.append(
                    np.array([_SENTINEL, _SENTINEL], dtype=np.float32))
        if any_valid_flag:
            neighbors_data.append(np.array(traj_pts))
            for t in steps:
                if t < T_raw and all_agent[t, ai, 8] > 0.5:
                    atype = int(all_agent[t, ai, 7])
                    neighbor_types.append(
                        _AGENT_TYPE_NAMES.get(atype, 'VEHICLE'))
                    break
            else:
                neighbor_types.append('VEHICLE')

    if neighbors_data:
        neighbor = np.stack(neighbors_data, axis=1)
    else:
        neighbor = np.full((total_len, 1, 2), _SENTINEL, dtype=np.float32)
        neighbor_types = ['VEHICLE']

    ego_at_col = all_agent[steps[-1], 0]
    ego_size = (float(ego_at_col[5]), float(ego_at_col[6]),
                float(ego_at_col[4]))

    dump = {
        'hist': hist,
        'fut': fut,
        'neighbor': neighbor,
        'neighbor_type': neighbor_types,
        'neighbor_ids': [str(i) for i in range(len(neighbor_types))],
        'inserted_neighbor_index': None,
        'pred5': None,
        'pred_many': None,
        'map': scenario_id,
        'angle': None,
        'ego_abs_pos': None,
        'ego_yaw': None,
        'ego_size': ego_size,
        'neighbor_sizes': {},
    }
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f"{scenario_id}.pkl")
    with open(dump_path, 'wb') as f:
        pickle.dump(dump, f)

    vectors = _lane_raw_to_vectors(lane_raw)
    full_vec_dir = os.path.join(vec_root, vec_subdir)
    os.makedirs(full_vec_dir, exist_ok=True)
    vec_path = os.path.join(full_vec_dir, f"{scenario_id}.vec.pkl")
    with open(vec_path, 'wb') as f:
        pickle.dump({'vectors': vectors}, f)

    return {'dump_path': dump_path, 'vec_path': vec_path,
            'scenario_id': scenario_id}


def plot_original_scenario(
    dump_dir: str,
    plot_out_dir: str,
    vec_map_root: str,
    scenario_id: str,
    python_bin: str = sys.executable,
    ego_margin: float = 50.0,
) -> Optional[str]:
    """Plot the original scenario (no inserted vehicles) using Reverse_waymo."""
    if not reverse_waymo_available():
        print(f"    WARNING: {_reverse_waymo_missing_message()}")
        return None

    tmp_dir = tempfile.mkdtemp(prefix='pipeline_orig_plot_')
    dump_path = os.path.join(dump_dir, f"{scenario_id}.pkl")
    if os.path.exists(dump_path):
        shutil.copy2(dump_path, os.path.join(tmp_dir, f"{scenario_id}.pkl"))

    os.makedirs(plot_out_dir, exist_ok=True)

    cmd = [
        python_bin, _PLOT_SCRIPT,
        '--dump_dir', tmp_dir,
        '--out_dir', plot_out_dir,
        '--vec_map_root', vec_map_root,
        '--ego_margin', str(ego_margin),
        '--dot_only',
    ]

    out_png = os.path.join(plot_out_dir, f"{scenario_id}.png")
    try:
        subprocess.check_call(cmd, timeout=120,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        if os.path.exists(out_png):
            print(f"    Original plot saved → {out_png}")
            return out_png
        else:
            print(f"    WARNING: expected output not found: {out_png}")
    except subprocess.CalledProcessError as e:
        print(f"    WARNING: original plotter failed: {e}")
    except subprocess.TimeoutExpired:
        print(f"    WARNING: original plotter timed out")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return None


def plot_with_reverse_waymo(
    dump_dir: str,
    plot_out_dir: str,
    vec_map_root: str,
    scenario_id: str,
    inserted_indices: List[int],
    python_bin: str = sys.executable,
    ego_margin: float = 50.0,
) -> Optional[str]:
    """
    Call ``Reverse_waymo/scripts/plot_waymo_vector_results.py`` for a
    single scenario.

    Returns the output PNG path on success, *None* otherwise.
    """
    if not reverse_waymo_available():
        print(f"    WARNING: {_reverse_waymo_missing_message()}")
        return None

    # Temp dir with just this scenario's dump (--id_colors is global per run)
    tmp_dir = tempfile.mkdtemp(prefix='pipeline_plot_')
    dump_path = os.path.join(dump_dir, f"{scenario_id}.pkl")
    if os.path.exists(dump_path):
        shutil.copy2(dump_path, os.path.join(tmp_dir, f"{scenario_id}.pkl"))

    os.makedirs(plot_out_dir, exist_ok=True)

    # Build --id_colors for every inserted vehicle
    id_parts = []
    for rank, idx in enumerate(inserted_indices):
        c = _INSERT_COLORS[rank % len(_INSERT_COLORS)]
        id_parts.append(f"ID{idx}={c}")

    cmd = [
        python_bin, _PLOT_SCRIPT,
        '--dump_dir', tmp_dir,
        '--out_dir', plot_out_dir,
        '--vec_map_root', vec_map_root,
        '--ego_margin', str(ego_margin),
        '--dot_only',
    ]
    if id_parts:
        cmd += ['--id_colors', ','.join(id_parts)]

    out_png = os.path.join(plot_out_dir, f"{scenario_id}.png")
    try:
        subprocess.check_call(cmd, timeout=120,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        if os.path.exists(out_png):
            print(f"    Plot saved → {out_png}")
            return out_png
        else:
            print(f"    WARNING: expected output not found: {out_png}")
    except subprocess.CalledProcessError as e:
        print(f"    WARNING: plotter failed: {e}")
    except subprocess.TimeoutExpired:
        print(f"    WARNING: plotter timed out")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return None


# =========================================================================
# Cluster visualisation
# =========================================================================

# Distinct cluster colours
_CLUSTER_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#AED6F1", "#D5F5E3", "#FADBD8", "#D6EAF8",
]


def plot_cluster_trajectories(
    vehicles: List[Dict],
    pkl_path: str,
    best_t: int,
    output_path: str,
    ego_margin: float = 50.0,
):
    """
    Plot per-inserted-vehicle cluster visualisation in Reverse_waymo style.

    For each inserted vehicle with ``cluster_info``:
      - Left panel:  CVAE samples (grey) + chosen representative (gold)
      - Right panel: K-means feature clusters coloured + representatives

    Uses the same drawing functions as ``Reverse_waymo/scripts/
    plot_waymo_vector_results.py`` (white background, vector map,
    oriented vehicle boxes, fading polylines).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    (_draw_vector_map, _draw_fading_polyline,
     _draw_oriented_box, _apply_ticks_and_grid) = _get_reverse_waymo_draw_funcs()

    _ensure_pickle_compat()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    all_agent = np.array(data['all_agent'])  # (T, N, 9)
    lane_raw = np.array(data.get('unsampled_lane', data['lane']))

    # Convert lane_raw to Reverse_waymo vector dict for draw_vector_map
    vec_dict = _lane_raw_to_vectors(lane_raw)

    ego_traj = all_agent[:, 0, :]
    ego_valid = ego_traj[:, 8] > 0.5
    valid_idx = np.where(ego_valid)[0]
    valid_idx = valid_idx[valid_idx <= best_t]
    ego_xy = ego_traj[valid_idx, :2]
    ego_heading = float(all_agent[valid_idx[-1], 0, 4]) if len(valid_idx) > 0 else 0.0
    ego_l = float(all_agent[valid_idx[-1], 0, 5]) if len(valid_idx) > 0 else 4.5
    ego_w = float(all_agent[valid_idx[-1], 0, 6]) if len(valid_idx) > 0 else 2.0

    # ── Collect other-agent trajectories (up to best_t) ──
    T_raw, N_agents, _ = all_agent.shape
    other_agents_info: List[Dict] = []
    for ai in range(1, N_agents):
        ag_traj = all_agent[:, ai, :]
        ag_valid = ag_traj[:, 8] > 0.5
        ag_valid_idx = np.where(ag_valid)[0]
        ag_valid_idx = ag_valid_idx[ag_valid_idx <= best_t]
        if len(ag_valid_idx) < 2:
            continue
        # Only keep vehicles (type==1)
        first_valid_t = ag_valid_idx[0]
        ag_type = int(ag_traj[first_valid_t, 7])
        ag_xy = ag_traj[ag_valid_idx, :2]
        last_t = ag_valid_idx[-1]
        ag_heading = float(ag_traj[last_t, 4])
        ag_l = float(ag_traj[last_t, 5])
        ag_w = float(ag_traj[last_t, 6])
        other_agents_info.append({
            'xy': ag_xy,
            'heading': ag_heading,
            'length': max(ag_l, 0.5),
            'width': max(ag_w, 0.3),
            'type': ag_type,
        })

    # Count how many vehicles have cluster_info
    n_veh = sum(1 for v in vehicles if v.get('cluster_info') is not None)
    if n_veh == 0:
        print("    No cluster info to plot.")
        return

    fig, axes = plt.subplots(n_veh, 2, figsize=(24, 12 * n_veh),
                              squeeze=False)

    plot_idx = 0
    for vi, veh in enumerate(vehicles):
        ci = veh.get('cluster_info')
        if ci is None:
            continue

        accepted_fwd = ci['all_accepted_fwd']  # list of (T+1, 2) fwd-time samples
        labels = ci.get('labels', np.zeros(len(accepted_fwd), dtype=int))
        n_clusters = ci.get('n_clusters', 0)
        n_accept = ci['n_accept']
        n_total = ci['n_total']
        chosen_idx = ci.get('chosen_idx', 0)
        cluster_representatives = ci.get('cluster_representatives', {})

        col_state = veh['collision_state']
        collision_pos = np.array([col_state['x'], col_state['y']])
        veh_heading = float(col_state['heading'])
        veh_l = float(col_state['length'])
        veh_w = float(col_state['width'])

        # Viewport
        cx, cy = float(collision_pos[0]), float(collision_pos[1])
        half = ego_margin
        bbox = (cx - half, cy - half, cx + half, cy + half)

        # ── Helper: draw shared scene elements on an axis ──
        def _draw_scene(ax):
            """Draw map, ego, and other agents on *ax*."""
            ax.set_facecolor("white")
            ax.set_aspect("equal")
            _draw_vector_map(ax, vec_dict, bbox, alpha=1.0)

            # Other agents – trajectories + boxes
            _other_color = "#888888"          # neutral grey
            _ped_color = "#a0522d"            # brownish for pedestrians
            _cyc_color = "#228b22"            # greenish for cyclists
            for oa in other_agents_info:
                axy = oa['xy']
                # Crop to viewport (skip agents entirely outside bbox)
                if (axy[:, 0].max() < bbox[0] - 5 or
                    axy[:, 0].min() > bbox[2] + 5 or
                    axy[:, 1].max() < bbox[1] - 5 or
                    axy[:, 1].min() > bbox[3] + 5):
                    continue
                if oa['type'] == 2:
                    c = _ped_color
                elif oa['type'] == 3:
                    c = _cyc_color
                else:
                    c = _other_color
                _draw_fading_polyline(ax, axy, base_color=c,
                                      linewidth=1.5, zorder=450)
                _draw_oriented_box(ax, axy[-1, :2], oa['heading'],
                                   oa['length'], oa['width'],
                                   color=c, alpha=0.55, zorder=500,
                                   edgecolor='#444444', edgewidth=0.8)

            # Ego trajectory + box
            _draw_fading_polyline(ax, ego_xy, base_color="#1a73e8",
                                  linewidth=3.0, zorder=600)
            _draw_oriented_box(ax, ego_xy[-1, :2], ego_heading,
                               ego_l, ego_w, color="#1a73e8", alpha=0.85,
                               zorder=700, edgecolor='black',
                               edgewidth=1.5)

        # ------ Panel 1: all CVAE samples + chosen representative ------
        ax_l = axes[plot_idx, 0]
        _draw_scene(ax_l)

        # All CVAE samples (light grey polylines)
        n_show = min(150, len(accepted_fwd))
        show_idx = np.linspace(0, len(accepted_fwd) - 1,
                                n_show, dtype=int)
        for si in show_idx:
            traj = accepted_fwd[si]
            ax_l.plot(traj[:, 0], traj[:, 1], '-', color="#bbbbbb",
                      linewidth=0.5, alpha=0.25, zorder=400)

        # Chosen representative
        chosen = accepted_fwd[chosen_idx]
        _draw_fading_polyline(ax_l, chosen, base_color="#FFD700",
                              linewidth=3.5, zorder=650)
        # Representative vehicle box at collision point
        _draw_oriented_box(ax_l, collision_pos, veh_heading,
                           veh_l, veh_w, color="#ff0000", alpha=0.85,
                           zorder=710, edgecolor='black', edgewidth=1.5)

        # Collision marker
        ax_l.scatter(*collision_pos, s=200, c="#32CD32", marker="*",
                     zorder=800, edgecolors="black", linewidths=1.0)

        ax_l.set_xlim(bbox[0], bbox[2])
        ax_l.set_ylim(bbox[1], bbox[3])
        _apply_ticks_and_grid(ax_l)

        ax_l.set_title(
            f"Vehicle {vi+1}: {n_accept} refined representatives "
            f"from {n_total} CVAE samples  |  "
            f"Gold = chosen representative",
            fontsize=12, pad=10)

        # Legend
        from matplotlib.lines import Line2D
        leg_elements = [
            Line2D([0], [0], color="#1a73e8", lw=2.5, label="Ego"),
            Line2D([0], [0], color="#888888", lw=1.5, label="Other vehicles"),
            Line2D([0], [0], color="#FFD700", lw=3, label="Chosen representative"),
            Line2D([0], [0], color="#bbbbbb", lw=1, alpha=0.5,
                   label=f"CVAE samples ({n_total})"),
            Line2D([0], [0], marker="*", color="#32CD32", lw=0,
                   markersize=12, label="Collision"),
        ]
        ax_l.legend(handles=leg_elements, loc="upper left", fontsize=9,
                    framealpha=0.85)

        # ------ Panel 2: K-means feature clusters ------
        ax_r = axes[plot_idx, 1]
        _draw_scene(ax_r)

        unique_cls = sorted(set(labels))
        legend_handles = [
            Line2D([0], [0], color="#1a73e8", lw=2.5, label="Ego"),
            Line2D([0], [0], color="#888888", lw=1.5, label="Other vehicles"),
        ]

        n_accepted = len(accepted_fwd)
        # Cluster trajectories + representatives
        for ci_idx, cl in enumerate(unique_cls):
            cl_mask = labels == cl
            cl_idx_arr = np.where(cl_mask)[0]
            color = _CLUSTER_PALETTE[ci_idx % len(_CLUSTER_PALETTE)]
            cl_size = len(cl_idx_arr)

            for idx in cl_idx_arr[:60]:
                if idx >= n_accepted:
                    continue
                traj = accepted_fwd[idx]
                ax_r.plot(traj[:, 0], traj[:, 1], '-', color=color,
                          linewidth=0.7, alpha=0.35, zorder=400)

            # Cluster representative (bold polyline + vehicle box)
            if cl in cluster_representatives:
                rep_i = cluster_representatives[cl]
                if rep_i >= n_accepted:
                    continue
                m = accepted_fwd[rep_i]
                _draw_fading_polyline(ax_r, m, base_color=color,
                                      linewidth=3.0, zorder=650)
                # Vehicle box at collision end of representative
                _draw_oriented_box(ax_r, m[-1, :2], veh_heading,
                                   veh_l, veh_w, color=color, alpha=0.8,
                                   zorder=710, edgecolor='black',
                                   edgewidth=1.2)

            legend_handles.append(
                Line2D([0], [0], color=color, lw=2.5,
                       label=f"C{cl} (n={cl_size})"))

        # Collision marker
        ax_r.scatter(*collision_pos, s=200, c="#32CD32", marker="*",
                     zorder=800, edgecolors="black", linewidths=1.0)
        legend_handles.append(
            Line2D([0], [0], marker="*", color="#32CD32", lw=0,
                   markersize=12, label="Collision"))

        # Inserted vehicle box (red, at collision)
        _draw_oriented_box(ax_r, collision_pos, veh_heading,
                           veh_l, veh_w, color="#ff0000", alpha=0.7,
                           zorder=705, edgecolor='black', edgewidth=1.5)

        ax_r.set_xlim(bbox[0], bbox[2])
        ax_r.set_ylim(bbox[1], bbox[3])
        _apply_ticks_and_grid(ax_r)

        ax_r.set_title(
            f"K-means: {n_clusters} lateral/arc-length clusters  |  "
            f"Chosen = cluster "
            f"{_find_cluster_of(labels, chosen_idx) if n_clusters > 0 else 'N/A'}"
            f" representative",
            fontsize=12, pad=10)
        ax_r.legend(handles=legend_handles, loc="upper left",
                    fontsize=8, framealpha=0.85,
                    ncol=max(1, n_clusters // 6 + 1))

        plot_idx += 1

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Cluster plot → {output_path}")


def _find_cluster_of(labels, idx):
    """Return the cluster label for a given index."""
    if idx < len(labels):
        return int(labels[idx])
    return -1


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='StateEstim → CVAE end-to-end pipeline')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing StateEstim-style scenario PKLs')
    parser.add_argument('--n_scenarios', type=int, default=3)
    parser.add_argument('--state_estim_ckpt', default=None)
    parser.add_argument('--cvae_ckpt', default=None)
    parser.add_argument('--refine_vocab', default=None,
                        help='Path to refine vocab .npz (default: refine/data/vocab_t0_K384.npz or COSTER_REFINE_VOCAB)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max_selected', type=int, default=1)
    parser.add_argument('--out_dir', default=None,
                        help='Output root (default: pipeline_out/ next to script)')
    parser.add_argument('--ego_margin', type=float, default=50.0,
                        help='Half-size (m) around ego for plot viewport')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip all plotting (only do inference + dump)')
    parser.add_argument('--save_meta', action='store_true',
                        help='Save per-scenario metadata JSON (n_clusters, etc.)')
    parser.add_argument('--min_t', type=int, default=0,
                        help='Minimum timestep for collision insertion (10Hz index, default: 0)')
    args = parser.parse_args()

    _ensure_pickle_compat()

    # ── Resolve output root ──
    if args.out_dir is None:
        args.out_dir = os.path.join(str(_THIS_DIR), 'pipeline_out')
    if not args.skip_plots and not reverse_waymo_available():
        print(f"WARNING: {_reverse_waymo_missing_message()}")
        print("WARNING: continuing with --skip_plots behavior.")
        args.skip_plots = True
    dump_dir = os.path.join(args.out_dir, 'dumps')
    vec_root = os.path.join(args.out_dir, 'vec_maps')
    plot_dir = os.path.join(args.out_dir, 'plots')
    snapshot_plot_dir = os.path.join(args.out_dir, 'snapshot_plots')
    orig_dump_dir = os.path.join(args.out_dir, 'original_dumps')
    orig_vec_root = os.path.join(args.out_dir, 'original_vec_maps')
    orig_plot_dir = os.path.join(args.out_dir, 'original_plots')
    for d in (dump_dir, vec_root, plot_dir,
              orig_dump_dir, orig_vec_root, orig_plot_dir):
        os.makedirs(d, exist_ok=True)
    os.makedirs(snapshot_plot_dir, exist_ok=True)

    # Find scenario files
    pkl_files = sorted(glob.glob(os.path.join(args.data_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in '{args.data_dir}'")

    # Filter stationary scenarios
    moving = []
    for p in pkl_files:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        if 'all_agent' in data:
            aa = np.array(data['all_agent'])
            speeds = np.sqrt(aa[:, 0, 2]**2 + aa[:, 0, 3]**2)
            if speeds.mean() > 1.0:
                moving.append(p)
    moving = moving[:args.n_scenarios]
    print(f"Processing {len(moving)} moving scenarios\n")

    # Build pipeline
    pipeline = StateEstimCVAEPipeline(
        state_estim_ckpt=args.state_estim_ckpt,
        cvae_ckpt=args.cvae_ckpt,
        refine_vocab=args.refine_vocab,
        device=args.device,
        max_selected=args.max_selected,
        min_t=args.min_t,
    )

    # Metadata collection
    meta_dir = os.path.join(args.out_dir, 'meta')
    if args.save_meta:
        os.makedirs(meta_dir, exist_ok=True)

    # Process each scenario
    ok = 0
    for i, pkl_path in enumerate(moving):
        name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(moving)}] {name}")
        print(f"{'='*60}")

        vehicles = pipeline.process_scenario(pkl_path)

        if not args.skip_plots:
            snapshot_paths = save_snapshot_plot(
                scenario_id=name,
                snapshot_debug=pipeline.last_stage_debug,
                out_dir=snapshot_plot_dir,
                ego_margin=args.ego_margin,
            )
            if snapshot_paths:
                print(f"  Snapshot plot saved")

        if not vehicles:
            print("  → Skipped (no vehicles)")
            # Save meta even for failed scenarios
            if args.save_meta:
                import json
                meta_path = os.path.join(meta_dir, f"{name}.json")
                with open(meta_path, 'w') as mf:
                    json.dump({'scenario': name, 'n_clusters': 0,
                               'n_vehicles': 0, 'success': False}, mf)
            continue

        # Compute n_clusters from the selected snapshot result
        n_clusters = 0
        if pipeline.last_stage_debug and 'snapshot_result' in pipeline.last_stage_debug:
            ci = pipeline.last_stage_debug['snapshot_result'].get('cluster_info')
            if ci and ci.get('n_clusters', 0) > 0:
                n_clusters = ci['n_clusters']

        # Print summary
        for vi, v in enumerate(vehicles):
            col = v['collision_state']
            t2 = v['trajectory_2hz']
            print(f"  Vehicle {vi+1}:")
            print(f"    Collision: ({col['x']:.1f}, {col['y']:.1f}) "
                  f"v=({col['vx']:.1f}, {col['vy']:.1f}) m/s "
                  f"hdg={np.degrees(col['heading']):.0f}°")
            print(f"    Trajectory: {len(t2)} steps × 0.5s = "
                  f"{len(t2)*0.5:.1f}s backward")
            traj_len = np.linalg.norm(np.diff(v['trajectory_global'], axis=0),
                                       axis=1).sum()
            print(f"    Path length: {traj_len:.1f}m")

        # ── Export dump + vec ──
        best_t = vehicles[0]['collision_timestep']
        export_info = export_scenario_for_plot(
            scenario_id=name,
            pkl_path=pkl_path,
            vehicles=vehicles,
            best_t=best_t,
            dump_dir=dump_dir,
            vec_root=vec_root,
            vec_subdir='pipeline',
        )
        if export_info is None:
            print("  → Export failed")
            if args.save_meta:
                import json
                meta_path = os.path.join(meta_dir, f"{name}.json")
                with open(meta_path, 'w') as mf:
                    json.dump({'scenario': name, 'n_clusters': n_clusters,
                               'n_vehicles': len(vehicles), 'success': False}, mf)
            continue

        if not args.skip_plots:
            # ── Plot original scenario (no inserted vehicles) ──
            orig_export = export_original_scenario_for_plot(
                scenario_id=name,
                pkl_path=pkl_path,
                best_t=best_t,
                dump_dir=orig_dump_dir,
                vec_root=orig_vec_root,
                vec_subdir='original',
            )
            if orig_export is not None:
                plot_original_scenario(
                    dump_dir=orig_dump_dir,
                    plot_out_dir=orig_plot_dir,
                    vec_map_root=orig_vec_root,
                    scenario_id=name,
                    ego_margin=args.ego_margin,
                )

            # ── Plot with Reverse_waymo (collision scenario) ──
            out_png = plot_with_reverse_waymo(
                dump_dir=dump_dir,
                plot_out_dir=plot_dir,
                vec_map_root=vec_root,
                scenario_id=name,
                inserted_indices=export_info['inserted_indices'],
                ego_margin=args.ego_margin,
            )
            if out_png:
                ok += 1

            # ── Cluster visualisation ──
            cluster_plot_dir = os.path.join(args.out_dir, 'cluster_plots')
            os.makedirs(cluster_plot_dir, exist_ok=True)
            cluster_png = os.path.join(cluster_plot_dir, f"{name}_clusters.png")
            plot_cluster_trajectories(
                vehicles=vehicles,
                pkl_path=pkl_path,
                best_t=best_t,
                output_path=cluster_png,
                ego_margin=args.ego_margin,
            )
        else:
            ok += 1

        # ── Save metadata JSON ──
        if args.save_meta:
            import json
            meta_path = os.path.join(meta_dir, f"{name}.json")
            with open(meta_path, 'w') as mf:
                json.dump({
                    'scenario': name,
                    'n_clusters': n_clusters,
                    'n_vehicles': len(vehicles),
                    'success': True,
                    'best_t': best_t,
                    'inserted_indices': export_info.get('inserted_indices', []),
                    'snapshot': {
                        'prob': float(vehicles[0].get('prob', 0.0)),
                        'position_log_prob': float(vehicles[0].get('position_log_prob', 0.0)),
                    },
                }, mf)

    print(f"\n{'='*60}")
    print(f"Done: {ok}/{len(moving)} scenarios processed")
    print(f"Original: {orig_plot_dir}")
    print(f"Plots:    {plot_dir}")
    print(f"Snapshots:{snapshot_plot_dir}")
    print(f"Dumps:    {dump_dir}")
    print(f"Vec maps: {vec_root}")
    if args.save_meta:
        print(f"Meta:     {meta_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

