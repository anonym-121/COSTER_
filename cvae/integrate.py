"""
Integration interface: VectorMapCVAE ↔ ADV-BMT scgen_generator.

Replaces GPT_backwardAR() in scgen_generator.py with CVAE_backward().
Input/output format matches ADV-BMT conventions so the rest of the pipeline
(collision point creation, reject sampling, scenario overwrite) stays unchanged.

Supports both:
  OB_HORIZON >= 1 → 2+ state observation (collision + extrapolated)
  OB_HORIZON  = 0 → single collision state (position, velocity, acceleration)

Usage:
    from cvae.integrate import CVAEBackwardGenerator

    gen = CVAEBackwardGenerator(
        ckpt_path="/path/to/ckpt-best",
        config_path="/path/to/config.py",
        device=torch.device("cuda"),
    )
    output = gen.backward(batched_data_dict, adv_id=adv_id)
"""
import os
import importlib.util
from typing import Optional

import torch

try:
    from .model import VectorMapCVAE
except ImportError:  # Allow running this file directly during development.
    from model import VectorMapCVAE


class CVAEBackwardGenerator:
    """
    Wraps a trained VectorMapCVAE model to produce backward trajectories
    in the format expected by ADV-BMT's scgen_generator pipeline.

    The generator:
    1. Extracts the ADV's collision state + scene context from batched_data_dict
    2. Builds observation states (single or multi-state depending on OB_HORIZON)
    3. Gathers neighbor trajectories for the full decode horizon
    4. Runs CVAE reverse-time inference with dynamic map & social attention
    5. Packs results into the same output dict format as GPT_backwardAR
    """

    def __init__(
        self,
        ckpt_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        n_predictions: int = 1,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_predictions = n_predictions

        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.py")
        spec = importlib.util.spec_from_file_location(
            "config", config_path,
            submodule_search_locations=[os.path.dirname(config_path)],
        )
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        self.config = config

        # Build and load model
        self.model = VectorMapCVAE(**config.model)
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            print(f"[CVAEBackwardGenerator] Loaded checkpoint: {ckpt_path}")
            if "ade" in state:
                print(f"  ADE: {state['ade']:.4f}, FDE: {state['fde']:.4f}, epoch: {state.get('epoch', '?')}")
        else:
            print(f"[CVAEBackwardGenerator] WARNING: checkpoint not found at {ckpt_path}")

        self.model.to(self.device)
        self.model.eval()

        self.skip = config.NUM_SKIPPED_STEPS
        self.ob_horizon = config.OB_HORIZON
        self.pred_horizon = config.PRED_HORIZON
        self.dt = self.skip * 0.1

    @torch.no_grad()
    def backward(self, batched_data_dict, adv_id):
        """
        Generate backward trajectories for the ADV agent using the CVAE.

        Args:
            batched_data_dict: dict with ADV-BMT convention tensors.
            adv_id: int — index of the ADV agent in the agent dimension.

        Returns:
            output_dict matching GPT_backwardAR output format.
        """
        device = self.device

        agent_pos = batched_data_dict["decoder/agent_position"]
        agent_heading = batched_data_dict["decoder/agent_heading"]
        agent_vel = batched_data_dict["decoder/agent_velocity"]
        agent_valid = batched_data_dict["decoder/agent_valid_mask"]

        map_feature = batched_data_dict["encoder/map_feature"]
        map_mask = batched_data_dict["encoder/map_feature_valid_mask"]
        map_position = batched_data_dict["encoder/map_position"]
        map_heading_feat = batched_data_dict["encoder/map_heading"]

        B, T, N = agent_pos.shape[:3]

        # Output buffers
        all_recon_pos = agent_pos.clone()
        all_recon_heading = agent_heading.clone()
        all_recon_vel = agent_vel.clone()
        all_recon_valid = agent_valid.clone()

        for b in range(B):
            adv_valid_b = agent_valid[b, :, adv_id]
            valid_steps = adv_valid_b.nonzero(as_tuple=True)[0]
            if len(valid_steps) == 0:
                continue
            col_step = valid_steps[-1].item()

            all_recon_valid[b, :, adv_id] = False
            all_recon_valid[b, col_step, adv_id] = True

            adv_col_pos = agent_pos[b, col_step, adv_id, :2]
            adv_col_vel = agent_vel[b, col_step, adv_id]

            fwd_disp = adv_col_vel * self.dt

            # ── Build observation states ──
            ego_states = torch.zeros(self.ob_horizon + 1, 6, device=device)
            if self.ob_horizon == 0:
                ego_states[0, :2] = adv_col_pos
                ego_states[0, 2:4] = fwd_disp
                ego_states[0, 4:6] = 0.0
            else:
                # past_interpolate: extrapolated reference before collision
                ego_states[0, :2] = adv_col_pos - fwd_disp
                ego_states[0, 2:4] = fwd_disp
                ego_states[0, 4:6] = 0.0
                # State 1: collision point
                ego_states[1, :2] = adv_col_pos
                ego_states[1, 2:4] = fwd_disp
                ego_states[1, 4:6] = 0.0

            # ── Build neighbor trajectories ──
            total_neigh_steps = self.ob_horizon + self.pred_horizon + 1
            other_agents = [i for i in range(N) if i != adv_id]

            rev_sub_steps = []
            for s_idx in range(total_neigh_steps):
                fwd_step = col_step - s_idx * self.skip
                rev_sub_steps.append(max(0, fwd_step))

            neighbor_list = []
            for oi in other_agents:
                n_states = []
                for fwd_step in rev_sub_steps:
                    if 0 <= fwd_step < T and agent_valid[b, fwd_step, oi]:
                        px = agent_pos[b, fwd_step, oi, 0].item()
                        py = agent_pos[b, fwd_step, oi, 1].item()
                        vx = agent_vel[b, fwd_step, oi, 0].item()
                        vy = agent_vel[b, fwd_step, oi, 1].item()
                        n_states.append([px, py, vx, vy, 0.0, 0.0])
                    else:
                        n_states.append([1e9] * 6)
                neighbor_list.append(n_states)

            if len(neighbor_list) == 0:
                neighbors = torch.full(
                    (total_neigh_steps, 1, 6),
                    1e9, dtype=torch.float32, device=device,
                )
            else:
                neighbors = torch.tensor(neighbor_list, dtype=torch.float32, device=device)
                neighbors = neighbors.permute(1, 0, 2)  # (L, Nn, 6)

            # ── Run CVAE inference ──
            x_in = ego_states.unsqueeze(1)   # (ob+1, 1, 6)
            n_in = neighbors.unsqueeze(1)    # (L, 1, Nn, 6)
            mf = map_feature[b:b + 1]
            mm = map_mask[b:b + 1]
            mp = map_position[b:b + 1]
            mh = map_heading_feat[b:b + 1]

            pred = self.model(
                x=x_in, neighbor=n_in,
                map_feature=mf, map_mask=mm,
                map_position=mp, map_heading=mh,
                n_predictions=self.n_predictions,
            )
            if pred.dim() == 4:
                pred = pred[0]
            pred = pred.squeeze(1)  # (horizon, 2)

            # ── Map predicted positions back to full timeline ──
            pred_positions = pred.cpu()

            for step_idx in range(self.pred_horizon):
                fwd_step = col_step - (step_idx + 1) * self.skip
                if fwd_step < 0:
                    break
                all_recon_pos[b, fwd_step, adv_id, 0] = pred_positions[step_idx, 0].item()
                all_recon_pos[b, fwd_step, adv_id, 1] = pred_positions[step_idx, 1].item()
                all_recon_valid[b, fwd_step, adv_id] = True

            # Interpolate between sub-sampled steps
            filled_steps = [col_step]
            for step_idx in range(self.pred_horizon):
                fwd_step = col_step - (step_idx + 1) * self.skip
                if fwd_step >= 0:
                    filled_steps.append(fwd_step)
            filled_steps.sort()

            for i in range(len(filled_steps) - 1):
                t0 = filled_steps[i]
                t1 = filled_steps[i + 1]
                if t1 - t0 <= 1:
                    continue
                p0_x = all_recon_pos[b, t0, adv_id, 0].item()
                p0_y = all_recon_pos[b, t0, adv_id, 1].item()
                p1_x = all_recon_pos[b, t1, adv_id, 0].item()
                p1_y = all_recon_pos[b, t1, adv_id, 1].item()
                for t in range(t0 + 1, t1):
                    alpha = (t - t0) / (t1 - t0)
                    all_recon_pos[b, t, adv_id, 0] = (1 - alpha) * p0_x + alpha * p1_x
                    all_recon_pos[b, t, adv_id, 1] = (1 - alpha) * p0_y + alpha * p1_y
                    all_recon_valid[b, t, adv_id] = True

            # Compute heading and velocity from positions
            for t in range(1, T):
                if all_recon_valid[b, t, adv_id] and all_recon_valid[b, t - 1, adv_id]:
                    dp = all_recon_pos[b, t, adv_id, :2] - all_recon_pos[b, t - 1, adv_id, :2]
                    all_recon_vel[b, t, adv_id] = dp / 0.1
                    heading = torch.atan2(dp[1], dp[0])
                    all_recon_heading[b, t, adv_id] = heading

            first_valid = all_recon_valid[b, :, adv_id].nonzero(as_tuple=True)[0]
            if len(first_valid) > 1:
                t0 = first_valid[0].item()
                t1 = first_valid[1].item()
                all_recon_vel[b, t0, adv_id] = all_recon_vel[b, t1, adv_id]
                all_recon_heading[b, t0, adv_id] = all_recon_heading[b, t1, adv_id]

        return {
            "decoder/reconstructed_position": all_recon_pos[..., :2],
            "decoder/reconstructed_heading": all_recon_heading,
            "decoder/reconstructed_velocity": all_recon_vel,
            "decoder/reconstructed_valid_mask": all_recon_valid,
        }
