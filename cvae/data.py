"""
Dataset for VectorMapCVAE.

Supports both:
  OB_HORIZON >= 1 → observation window = OB_HORIZON + 1 states
  OB_HORIZON  = 0 → single collision state (no extrapolation)

Neighbor data spans (ob+pred+1) steps for decoder social attention.
"""
import os
import sys
import pathlib
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# ── ADV-BMT imports (used as a library) ──
_BMT_ROOT = os.environ.get(
    "ADV_BMT_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent.parent / "Adv-BMT"),
)
if _BMT_ROOT not in sys.path:
    sys.path.insert(0, _BMT_ROOT)

try:
    from bmt.dataset.preprocessor import (
        process_map_and_traffic_light,
        process_track,
    )
except ImportError:
    process_map_and_traffic_light = None
    process_track = None

try:
    from scenarionet import read_dataset_summary, read_scenario
except ImportError:
    read_dataset_summary = None
    read_scenario = None


class ScenarioNetCVAEDataset(Dataset):
    """
    PyTorch Dataset that reads ScenarioNet .pkl files and produces training
    samples for VectorMapCVAE.

    Each sample is a dict with:
        x:            (ob_horizon+1, 6)   ego state [x, y, vx, vy, ax, ay]
        y:            (pred_horizon, 2)    future displacement targets
        neighbor:     (ob_horizon+pred_horizon+1, Nn, 6)
        map_feature:  (M, V, 27)
        map_mask:     (M, V) bool
        map_position: (M, 3)
        map_heading:  (M,)
    """

    def __init__(
        self,
        data_dir,
        ob_horizon: int = 5,
        pred_horizon: int = 10,
        num_skipped_steps: int = 5,
        max_agents: int = 64,
        max_map_features: int = 256,
        max_vectors: int = 128,
        max_traffic_lights: int = 64,
        ob_radius: float = 50.0,
        reverse_time: bool = True,
        max_neighbors: int = 32,
        limit_map_range: bool = True,
        augment: bool = True,
        fraction: Optional[float] = None,
        split: Optional[str] = None,
        split_ratio: float = 0.8,
        seed: int = 42,
        subfolder_filter: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            data_dir: path or list of paths to ScenarioNet data directories.
            split: None (all), "train" (first split_ratio), "val" (rest).
            split_ratio: train/val split ratio (default 0.8).
            subfolder_filter: optional list of subfolder names to restrict loading.
        """
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dirs = data_dir
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.skip = num_skipped_steps
        self.max_agents = max_agents
        self.max_map_features = max_map_features
        self.max_vectors = max_vectors
        self.max_traffic_lights = max_traffic_lights
        self.ob_radius = ob_radius
        self.reverse_time = reverse_time
        self.max_neighbors = max_neighbors
        self.limit_map_range = limit_map_range
        self.augment = augment

        self.total_steps = ob_horizon + pred_horizon
        self.raw_steps_needed = self.total_steps * self.skip

        assert read_dataset_summary is not None, (
            "scenarionet is required. Install with: pip install scenarionet"
        )
        if process_map_and_traffic_light is None or process_track is None:
            raise ImportError(
                "ADV-BMT preprocessing utilities are required for CVAE training. "
                "Set ADV_BMT_ROOT to your Adv-BMT checkout before creating "
                "ScenarioNetCVAEDataset."
            )

        # Discover scenario files across all data directories
        all_summary_list = []
        all_mappings = {}
        _subfolder_set = set(subfolder_filter) if subfolder_filter else None
        for ddir in self.data_dirs:
            summary_dict, summary_list, mapping = read_dataset_summary(ddir)
            loaded = 0
            for fname in summary_list:
                if _subfolder_set is not None:
                    subfolder = mapping.get(fname, "")
                    if subfolder not in _subfolder_set:
                        continue
                all_summary_list.append(fname)
                all_mappings[fname] = (ddir, mapping)
                loaded += 1
            tag = f" (filtered: {subfolder_filter})" if _subfolder_set else ""
            print(f"  loaded {loaded} scenarios from {ddir}{tag}")
        summary_list = all_summary_list
        self._dir_mapping = all_mappings

        # Deterministic split
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(summary_list))
        n_train = int(len(summary_list) * split_ratio)
        if split == "train":
            idx = sorted(perm[:n_train])
            summary_list = [summary_list[i] for i in idx]
        elif split == "val":
            idx = sorted(perm[n_train:])
            summary_list = [summary_list[i] for i in idx]

        if fraction is not None and 0.0 < fraction < 1.0:
            rng2 = np.random.RandomState(seed + 1)
            n_pick = max(1, int(len(summary_list) * fraction))
            idx = rng2.choice(len(summary_list), size=n_pick, replace=False)
            idx.sort()
            summary_list = [summary_list[i] for i in idx]

        self.summary_list = summary_list
        self.rng = np.random.RandomState(seed)

        tag = f" [{split}]" if split else ""
        dirs_str = ", ".join(self.data_dirs)
        print(f"[ScenarioNetCVAEDataset{tag}] {len(self.summary_list)} scenarios from {dirs_str}")

    def __len__(self):
        return len(self.summary_list)

    def __getitem__(self, idx):
        file_name = self.summary_list[idx]
        data_dir, mapping = self._dir_mapping[file_name]
        scenario = read_scenario(
            dataset_path=data_dir,
            mapping=mapping,
            scenario_file_name=file_name,
        )

        try:
            sample = self._process_scenario(scenario)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        return sample

    def _process_scenario(self, scenario):
        """Convert a ScenarioNet scenario dict into a CVAE training sample."""
        from metadrive.scenario.scenario_description import ScenarioDescription as SD

        metadata = scenario[SD.METADATA]
        track_length = scenario[SD.LENGTH]
        sdc_name = metadata["sdc_id"]

        config = _make_preprocess_config(
            max_vectors=self.max_vectors,
            max_map_features=self.max_map_features,
            max_traffic_lights=self.max_traffic_lights,
            max_agents=self.max_agents,
            limit_map_range=self.limit_map_range,
        )

        # ---- Map features ----
        data_dict = {}
        if "current_time_index" in metadata:
            data_dict["metadata/current_time_index"] = metadata["current_time_index"]
        else:
            data_dict["metadata/current_time_index"] = 0

        tracks_to_predict_dict = metadata.get("tracks_to_predict", {})
        track_index_to_predict = np.array(
            [int(v["track_index"]) for v in tracks_to_predict_dict.values()]
        )
        track_name_to_predict = [int(k) for k in tracks_to_predict_dict.keys()]
        try:
            sdc_name_int = int(sdc_name)
        except Exception:
            sdc_name_int = sdc_name
        if sdc_name_int in track_name_to_predict:
            track_name_to_predict.remove(sdc_name_int)
            track_name_to_predict.insert(0, sdc_name_int)
        track_name_to_predict = np.array(track_name_to_predict)

        data_dict.update({
            "in_evaluation": False,
            "metadata/sdc_name": sdc_name,
            "encoder/object_of_interest_name": track_name_to_predict,
            "encoder/object_of_interest_id": track_index_to_predict,
        })

        data_dict = process_map_and_traffic_light(
            data_dict=data_dict,
            scenario=scenario,
            map_feature=scenario[SD.MAP_FEATURES],
            dynamic_map_states=scenario[SD.DYNAMIC_MAP_STATES],
            track_length=track_length,
            max_vectors=config["max_vectors"],
            max_map_features=config["max_map_features"],
            limit_map_range=config["limit_map_range"],
            max_length_per_map_feature=10000,
            max_traffic_lights=config["max_traffic_lights"],
            remove_traffic_light_state=True,
        )

        # ---- Agent features ----
        data_dict = process_track(
            data_dict=data_dict,
            tracks=scenario[SD.TRACKS],
            track_length=track_length,
            sdc_name=sdc_name,
            max_agents=self.max_agents,
        )

        # ---- Extract what we need ----
        map_feature = data_dict["encoder/map_feature"]          # (M, V, 27)
        map_mask = data_dict["encoder/map_feature_valid_mask"]  # (M, V) bool
        map_position = data_dict["encoder/map_position"]        # (M, 3)
        map_heading = data_dict["encoder/map_heading"]          # (M,)

        agent_position = data_dict["encoder/agent_position"]    # (T, N, 3)
        agent_velocity = data_dict["encoder/agent_velocity"]    # (T, N, 2)
        agent_valid = data_dict["encoder/agent_valid_mask"]     # (T, N) bool
        sdc_index = data_dict["encoder/sdc_index"]

        T_full, N_agents = agent_position.shape[:2]

        # Sub-sample every skip steps
        time_indices = np.arange(0, T_full, self.skip)
        if len(time_indices) < self.total_steps + 1:
            time_indices = np.arange(T_full)

        pos_sub = agent_position[time_indices]    # (T_sub, N, 3)
        vel_sub = agent_velocity[time_indices]    # (T_sub, N, 2)
        valid_sub = agent_valid[time_indices]     # (T_sub, N)

        # Choose ego agent (SDC first, then any valid agent)
        ego_idx = sdc_index
        ego_valid = valid_sub[:, ego_idx]
        if ego_valid.sum() < self.total_steps:
            valid_counts = valid_sub.sum(axis=0)
            candidates = np.where(valid_counts >= self.total_steps)[0]
            if len(candidates) == 0:
                raise ValueError("No agent with enough valid steps")
            ego_idx = self.rng.choice(candidates)
            ego_valid = valid_sub[:, ego_idx]

        # Find a valid window
        valid_steps = np.where(ego_valid)[0]
        if len(valid_steps) < self.total_steps + 1:
            raise ValueError("Not enough valid steps for ego")

        max_start = len(valid_steps) - (self.total_steps + 1)
        if max_start <= 0:
            start = 0
        else:
            start = self.rng.randint(0, max_start + 1) if self.augment else 0
        window_valid_indices = valid_steps[start:start + self.total_steps + 1]

        # Extract ego trajectory in this window
        ego_pos = pos_sub[window_valid_indices, ego_idx, :2]    # (total+1, 2)
        ego_vel = vel_sub[window_valid_indices, ego_idx]         # (total+1, 2)

        # Compute acceleration
        ego_v = ego_pos[1:] - ego_pos[:-1]
        ego_a = np.zeros_like(ego_v)
        ego_a[1:] = ego_v[1:] - ego_v[:-1]
        ego_a[0] = ego_a[1] if len(ego_a) > 1 else ego_a[0]

        # Build ego state: [x, y, vx, vy, ax, ay]
        ego_state = np.concatenate([ego_pos[1:], ego_v, ego_a], axis=-1)  # (total, 6)

        # Build neighbor trajectories
        other_agents = [i for i in range(N_agents) if i != ego_idx]
        neighbor_list = []
        for oi in other_agents:
            n_pos = pos_sub[window_valid_indices, oi, :2]
            n_vel = vel_sub[window_valid_indices, oi]
            n_valid = valid_sub[window_valid_indices, oi]
            n_state_pos = np.where(n_valid[:, None], n_pos, 1e9)
            n_state_vel = np.where(n_valid[:, None], n_vel, 1e9)
            n_state = np.concatenate([n_state_pos, n_state_vel], axis=-1)
            n_state_full = np.full((self.total_steps + 1, 6), 1e9, dtype=np.float32)
            n_state_full[:, :4] = n_state
            neighbor_list.append(n_state_full)

        if len(neighbor_list) == 0:
            neighbors = np.full((self.total_steps + 1, 1, 6), 1e9, dtype=np.float32)
        else:
            neighbors = np.stack(neighbor_list, axis=1)

        # Filter neighbors by distance
        ego_pos_expanded = ego_pos[:, None, :]
        n_pos = neighbors[..., :2]
        dist = np.linalg.norm(n_pos - ego_pos_expanded, axis=-1)
        within_radius = np.any(dist <= self.ob_radius, axis=0)
        neighbors = neighbors[:, within_radius]
        if neighbors.shape[1] == 0:
            neighbors = np.full((self.total_steps + 1, 1, 6), 1e9, dtype=np.float32)

        # Limit number of neighbors
        if neighbors.shape[1] > self.max_neighbors:
            avg_dist = np.nanmean(
                np.where(dist[:, within_radius] < 1e8, dist[:, within_radius], np.nan),
                axis=0,
            )
            keep_idx = np.argsort(avg_dist)[:self.max_neighbors]
            neighbors = neighbors[:, keep_idx]

        # ---- Reverse time if needed ----
        if self.reverse_time:
            ego_state = ego_state[::-1].copy()
            neighbors = neighbors[::-1].copy()
            ego_pos = ego_pos[::-1].copy()

        # Split into observation and future
        x_full = np.zeros((self.ob_horizon + 1, 6), dtype=np.float32)
        if self.ob_horizon == 0:
            x_full[0, :2] = ego_pos[0]
            x_full[0, 2:4] = ego_state[0, 2:4]
            x_full[0, 4:6] = ego_state[0, 4:6]
        else:
            x_full[1:] = ego_state[:self.ob_horizon]
            if self.reverse_time:
                x_full[0, :2] = ego_pos[0] + ego_state[0, 2:4]
            else:
                x_full[0, :2] = ego_pos[0]
                x_full[0, 2:4] = ego_state[0, 2:4]
                x_full[0, 4:6] = ego_state[0, 4:6]

        y_future = ego_state[self.ob_horizon:self.ob_horizon + self.pred_horizon, :2]

        # neighbor: (ob+pred+1, Nn, 6)
        n_obs = neighbors[:self.ob_horizon + self.pred_horizon + 1]

        # Data augmentation: random rotation
        if self.augment and self.rng.random() > 0.5:
            angle = self.rng.uniform(0, 2 * np.pi)
            s, c = np.sin(angle), np.cos(angle)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            # Rotate ego
            x_full[..., :2] = (R @ x_full[..., :2, np.newaxis]).squeeze(-1)
            x_full[..., 2:4] = (R @ x_full[..., 2:4, np.newaxis]).squeeze(-1)
            x_full[..., 4:6] = (R @ x_full[..., 4:6, np.newaxis]).squeeze(-1)
            y_future = (R @ y_future[..., np.newaxis]).squeeze(-1)
            # Rotate neighbors
            for dim_pair in [(0, 2), (2, 4)]:
                valid = np.abs(n_obs[..., dim_pair[0]]) < 1e8
                vals = n_obs[..., dim_pair[0]:dim_pair[1]].copy()
                rotated = np.einsum("ij,...j->...i", R, vals)
                n_obs[..., dim_pair[0]:dim_pair[1]] = np.where(
                    valid[..., None], rotated, n_obs[..., dim_pair[0]:dim_pair[1]]
                )
            # Rotate map features (start, end, direction)
            map_feature = map_feature.copy()
            for start_dim in [0, 3, 6]:
                xy = map_feature[..., start_dim:start_dim + 2].copy()
                map_feature[..., start_dim:start_dim + 2] = np.einsum(
                    "ij,...j->...i", R, xy
                )
            dir_xy = map_feature[..., 6:8]
            new_heading = np.arctan2(dir_xy[..., 1], dir_xy[..., 0])
            map_feature[..., 9] = new_heading
            map_feature[..., 10] = np.sin(new_heading)
            map_feature[..., 11] = np.cos(new_heading)

            map_position = map_position.copy()
            map_position[..., :2] = (R @ map_position[..., :2, np.newaxis]).squeeze(-1)
            map_heading = map_heading.copy() + angle

        return {
            "x": np.float32(x_full),
            "y": np.float32(y_future),
            "neighbor": np.float32(n_obs),
            "map_feature": np.float32(map_feature),
            "map_mask": map_mask.astype(bool),
            "map_position": np.float32(map_position),
            "map_heading": np.float32(map_heading),
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate that pads variable-size neighbors and map features."""
        result = {}

        # ---- Fixed-size tensors: x, y ----
        for k in ["x", "y"]:
            result[k] = torch.tensor(np.stack([b[k] for b in batch]), dtype=torch.float32)
            result[k] = result[k].permute(1, 0, *range(2, result[k].dim()))

        # ---- Variable-size: neighbor ----
        max_nn = max(b["neighbor"].shape[1] for b in batch)
        padded_neighbors = []
        for b in batch:
            n = b["neighbor"]
            if n.shape[1] < max_nn:
                pad = np.full((n.shape[0], max_nn - n.shape[1], n.shape[2]), 1e9, dtype=np.float32)
                n = np.concatenate([n, pad], axis=1)
            padded_neighbors.append(n)
        result["neighbor"] = torch.tensor(np.stack(padded_neighbors), dtype=torch.float32)
        result["neighbor"] = result["neighbor"].permute(1, 0, 2, 3)  # (L, B, Nn, 6)

        # ---- Variable-size: map features ----
        max_M = max(b["map_feature"].shape[0] for b in batch)
        max_V = max(b["map_feature"].shape[1] for b in batch)
        padded_map_feat, padded_map_mask = [], []
        padded_map_pos, padded_map_head = [], []
        for b in batch:
            mf = b["map_feature"]
            mm = b["map_mask"]
            mp = b["map_position"]
            mh = b["map_heading"]
            M, V, D = mf.shape
            mf_pad = np.zeros((max_M, max_V, D), dtype=np.float32)
            mm_pad = np.zeros((max_M, max_V), dtype=bool)
            mp_pad = np.zeros((max_M, 3), dtype=np.float32)
            mh_pad = np.zeros((max_M,), dtype=np.float32)
            mf_pad[:M, :V] = mf
            mm_pad[:M, :V] = mm
            mp_pad[:M] = mp
            mh_pad[:M] = mh
            padded_map_feat.append(mf_pad)
            padded_map_mask.append(mm_pad)
            padded_map_pos.append(mp_pad)
            padded_map_head.append(mh_pad)

        result["map_feature"] = torch.tensor(np.stack(padded_map_feat), dtype=torch.float32)
        result["map_mask"] = torch.tensor(np.stack(padded_map_mask), dtype=torch.bool)
        result["map_position"] = torch.tensor(np.stack(padded_map_pos), dtype=torch.float32)
        result["map_heading"] = torch.tensor(np.stack(padded_map_head), dtype=torch.float32)

        return result


def _make_preprocess_config(max_vectors, max_map_features, max_traffic_lights,
                            max_agents, limit_map_range):
    """Create a minimal config dict for the preprocessor functions."""
    return {
        "max_vectors": max_vectors,
        "max_map_features": max_map_features,
        "max_traffic_lights": max_traffic_lights,
        "max_agents": max_agents,
        "limit_map_range": limit_map_range,
    }
