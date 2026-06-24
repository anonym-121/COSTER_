"""
Unified scenario loader for StateEstim init model.

Loads a .pkl file and directly extracts the three arrays the inference
pipeline needs, regardless of whether the file is in StateEstim format
(produced by trans20.py) or ScenarioNet format (from waymo_pkl/).

Usage:
    from utils.scenario_loader import load_scenario
    lane_raw, agents_t0, traf_t0 = load_scenario('/path/to/scenario.pkl')
"""

import pickle
from enum import Enum

import numpy as np

# =========================================================================
# ScenarioNet type mappings
# =========================================================================

# Map feature type string → integer (matches trans20.py encoding)
_MAP_TYPE = {
    'LANE_FREEWAY':        1,
    'LANE_SURFACE_STREET': 2,
    'LANE_BIKE_LANE':      3,
    'ROAD_LINE_BROKEN_SINGLE_WHITE':   6,
    'ROAD_LINE_SOLID_SINGLE_WHITE':    7,
    'ROAD_LINE_SOLID_DOUBLE_WHITE':    8,
    'ROAD_LINE_BROKEN_SINGLE_YELLOW':  9,
    'ROAD_LINE_BROKEN_DOUBLE_YELLOW':  10,
    'ROAD_LINE_SOLID_SINGLE_YELLOW':   11,
    'ROAD_LINE_SOLID_DOUBLE_YELLOW':   12,
    'ROAD_LINE_PASSING_DOUBLE_YELLOW': 13,
    'ROAD_EDGE_BOUNDARY': 15,
    'ROAD_EDGE_MEDIAN':   16,
    'STOP_SIGN':  17,
    'CROSSWALK':  18,
    'SPEED_BUMP': 19,
}

_TL_STATE = {
    'LANE_STATE_UNKNOWN': 0, 'None': 0,
    'LANE_STATE_ARROW_STOP': 1, 'LANE_STATE_STOP': 1, 'LANE_STATE_FLASHING_STOP': 1,
    'LANE_STATE_ARROW_CAUTION': 2, 'LANE_STATE_CAUTION': 2, 'LANE_STATE_FLASHING_CAUTION': 2,
    'LANE_STATE_ARROW_GO': 3, 'LANE_STATE_GO': 3,
}

_TRACK_TYPE = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'OTHER': 4}

_SAMPLE_NUM = 10  # polyline downsampling interval (trans20.py default)


# =========================================================================
# Pickle compatibility for old StateEstim pkl files
# =========================================================================

class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8


def _ensure_pickle_compat():
    import __main__
    if not hasattr(__main__, 'RoadLineType'):
        __main__.RoadLineType = RoadLineType


# =========================================================================
# ScenarioNet → arrays (direct extraction, no intermediate dict)
# =========================================================================

def _downsample(pts, n=_SAMPLE_NUM, keep_all=False):
    if len(pts) < n or keep_all:
        return pts
    return pts[::n]


def _extract_lane_scenarionet(map_features):
    """map_features dict → lane array [N, 4]  (x, y, type, id)"""
    rows = []
    for fid_str, feat in map_features.items():
        fid = int(fid_str)
        tint = _MAP_TYPE.get(feat['type'])
        if tint is None:
            continue

        if feat['type'] == 'STOP_SIGN':
            pos = feat['position'][:2].astype(np.float64)
            rows.append([pos[0], pos[1], tint, fid])
            continue

        if feat['type'] in ('CROSSWALK', 'SPEED_BUMP'):
            pts = feat['polygon'][:, :2].astype(np.float64)
            pts = _downsample(pts, keep_all=True)
        else:
            pts = feat['polyline'][:, :2].astype(np.float64)
            pts = _downsample(pts)

        for p in pts:
            rows.append([p[0], p[1], tint, fid])

    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4))


def _resolve_timestep_idx(num_steps, timestep):
    """Resolve timestep selector to a valid integer index."""
    if num_steps <= 0:
        return 0
    if timestep == 'mid':
        return num_steps // 2
    if timestep == 'last':
        return num_steps - 1
    if timestep is None:
        return 0
    idx = int(timestep)
    return max(0, min(num_steps - 1, idx))


def _extract_agents_scenarionet(tracks, sdc_id, timestep=0):
    """tracks dict → agents at selected timestep as [N, 9]"""
    ids = list(tracks.keys())
    sid = str(sdc_id)
    if sid in ids:
        ids.remove(sid)
        ids.insert(0, sid)

    n = len(ids)
    out = np.zeros([n, 9], dtype=np.float64)
    any_track = tracks[ids[0]]['state']
    num_steps = int(any_track['position'].shape[0])
    t = _resolve_timestep_idx(num_steps, timestep)
    for i, tid in enumerate(ids):
        s = tracks[tid]['state']
        out[i, 0] = s['position'][t, 0]
        out[i, 1] = s['position'][t, 1]
        out[i, 2] = s['velocity'][t, 0]
        out[i, 3] = s['velocity'][t, 1]
        out[i, 4] = s['heading'][t]
        out[i, 5] = s['length'][t]
        out[i, 6] = s['width'][t]
        out[i, 7] = _TRACK_TYPE.get(tracks[tid]['type'], 0)
        out[i, 8] = float(s['valid'][t])
    return out


def _extract_traf_scenarionet(dynamic_map_states, timestep=0):
    """dynamic_map_states dict → traffic lights at selected timestep as list[ndarray]"""
    traf = []
    for did, d in dynamic_map_states.items():
        if d.get('type') != 'TRAFFIC_LIGHT':
            continue
        states = d['state']['object_state']
        if not states:
            continue
        t = _resolve_timestep_idx(len(states), timestep)
        st = str(states[t])
        si = _TL_STATE.get(st, 0)
        active = 1.0 if st not in ('None', 'LANE_STATE_UNKNOWN') else 0.0
        sp = d['stop_point'][:2].astype(np.float64)
        traf.append(np.array([d['lane'], sp[0], sp[1], 0.0, si, active],
                             dtype=np.float64))
    return traf


# =========================================================================
# Public API
# =========================================================================

def load_scenario(pkl_path, timestep=0):
    """
    Load a .pkl scenario and return the three arrays needed for inference.

    Automatically detects:
      - **ScenarioNet format**: has ``map_features`` key (dict-based)
      - **StateEstim format**: has ``lane`` key (array-based, from trans20.py)

    Args:
        pkl_path: path to the .pkl file.

    Returns:
        lane_raw   – np.ndarray [N, 4]  (x, y, type, lane_id) – global coords
        agents_t0  – np.ndarray [M, 9] at selected timestep
        traf_t0    – list[np.ndarray] traffic light info at selected timestep
    """
    _ensure_pickle_compat()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if 'map_features' in data and 'lane' not in data:
        # ---- ScenarioNet format → direct extraction ----
        meta = data.get('metadata', {})
        sdc_id = meta.get('sdc_id', '0')

        lane_raw = _extract_lane_scenarionet(data['map_features'])
        agents_t0 = _extract_agents_scenarionet(data['tracks'], sdc_id, timestep=timestep)
        traf_t0 = _extract_traf_scenarionet(data.get('dynamic_map_states', {}), timestep=timestep)

    else:
        # ---- StateEstim format → simple array extraction ----
        lane_raw = data['lane']
        aa = data['all_agent']
        t = _resolve_timestep_idx(int(aa.shape[0]), timestep)
        agents_t0 = aa[t]
        tl = data['traffic_light']
        t_tl = _resolve_timestep_idx(len(tl), timestep)
        traf_t0 = tl[t_tl]

    return lane_raw, agents_t0, traf_t0


